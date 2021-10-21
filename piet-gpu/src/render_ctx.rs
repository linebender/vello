use std::borrow::Cow;

use crate::MAX_BLEND_STACK;
use piet::kurbo::{Affine, Insets, PathEl, Point, Rect, Shape};
use piet::{
    Color, Error, FixedGradient, ImageFormat, InterpolationMode, IntoBrush, RenderContext,
    StrokeStyle,
};

use piet_gpu_types::encoder::{Encode, Encoder};
use piet_gpu_types::scene::{
    Clip, CubicSeg, Element, FillColor, FillLinGradient, LineSeg, QuadSeg, SetFillMode,
    SetLineWidth, Transform,
};

use crate::gradient::{LinearGradient, RampCache};
use crate::text::Font;
pub use crate::text::{PathEncoder, PietGpuText, PietGpuTextLayout, PietGpuTextLayoutBuilder};

pub struct PietGpuImage;

pub struct PietGpuRenderContext {
    encoder: Encoder,
    elements: Vec<Element>,
    // Will probably need direct accesss to hal Device to create images etc.
    inner_text: PietGpuText,
    stroke_width: f32,
    fill_mode: FillMode,
    // We're tallying these cpu-side for expedience, but will probably
    // move this to some kind of readback from element processing.
    /// The count of elements that make it through to coarse rasterization.
    path_count: usize,
    /// The count of path segment elements.
    pathseg_count: usize,
    /// The count of transform elements.
    trans_count: usize,

    cur_transform: Affine,
    state_stack: Vec<State>,
    clip_stack: Vec<ClipElement>,

    ramp_cache: RampCache,
}

#[derive(Clone)]
pub enum PietGpuBrush {
    Solid(u32),
    LinGradient(LinearGradient),
}

#[derive(Default)]
struct State {
    /// The transform relative to the parent state.
    rel_transform: Affine,
    /// The transform at the parent state.
    ///
    /// This invariant should hold: transform * rel_transform = cur_transform
    transform: Affine,
    n_clip: usize,
}

struct ClipElement {
    /// Index of BeginClip element in element vec, for bbox fixup.
    begin_ix: usize,
    bbox: Option<Rect>,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FillMode {
    // Fill path according to the non-zero winding rule.
    Nonzero = 0,
    // Fill stroked path.
    Stroke = 1,
}

const TOLERANCE: f64 = 0.25;

impl PietGpuRenderContext {
    pub fn new() -> PietGpuRenderContext {
        let encoder = Encoder::new();
        let elements = Vec::new();
        let font = Font::new();
        let inner_text = PietGpuText::new(font);
        let stroke_width = 0.0;
        PietGpuRenderContext {
            encoder,
            elements,
            inner_text,
            stroke_width,
            fill_mode: FillMode::Nonzero,
            path_count: 0,
            pathseg_count: 0,
            trans_count: 0,
            cur_transform: Affine::default(),
            state_stack: Vec::new(),
            clip_stack: Vec::new(),
            ramp_cache: RampCache::default(),
        }
    }

    pub fn get_scene_buf(&mut self) -> &[u8] {
        const ALIGN: usize = 128;
        let padded_size = (self.elements.len() + (ALIGN - 1)) & ALIGN.wrapping_neg();
        self.elements.resize(padded_size, Element::Nop());
        self.elements.encode(&mut self.encoder);
        self.encoder.buf()
    }

    pub fn path_count(&self) -> usize {
        self.path_count
    }

    pub fn pathseg_count(&self) -> usize {
        self.pathseg_count
    }

    pub fn trans_count(&self) -> usize {
        self.trans_count
    }

    pub fn get_ramp_data(&self) -> Vec<u32> {
        self.ramp_cache.get_ramp_data()
    }

    pub(crate) fn set_fill_mode(&mut self, fill_mode: FillMode) {
        if self.fill_mode != fill_mode {
            self.elements.push(Element::SetFillMode(SetFillMode {
                fill_mode: fill_mode as u32,
            }));
            self.fill_mode = fill_mode;
        }
    }
}

impl RenderContext for PietGpuRenderContext {
    type Brush = PietGpuBrush;
    type Image = PietGpuImage;
    type Text = PietGpuText;
    type TextLayout = PietGpuTextLayout;

    fn status(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn solid_brush(&mut self, color: Color) -> Self::Brush {
        // kernel4 expects colors encoded in alpha-premultiplied sRGB:
        //
        // [α,sRGB(α⋅R),sRGB(α⋅G),sRGB(α⋅B)]
        //
        // See also http://ssp.impulsetrain.com/gamma-premult.html.
        let (r, g, b, a) = color.as_rgba();
        let premul = Color::rgba(
            to_srgb(from_srgb(r) * a),
            to_srgb(from_srgb(g) * a),
            to_srgb(from_srgb(b) * a),
            a,
        );
        PietGpuBrush::Solid(premul.as_rgba_u32())
    }

    fn gradient(&mut self, gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Error> {
        match gradient.into() {
            FixedGradient::Linear(lin) => {
                let lin = self.ramp_cache.add_linear_gradient(&lin);
                Ok(PietGpuBrush::LinGradient(lin))
            }
            _ => todo!("don't do radial gradients yet"),
        }
    }

    fn clear(&mut self, _color: Color) {}

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        let width_f32 = width as f32;
        if self.stroke_width != width_f32 {
            self.elements
                .push(Element::SetLineWidth(SetLineWidth { width: width_f32 }));
            self.stroke_width = width_f32;
        }
        self.set_fill_mode(FillMode::Stroke);
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        // Note: the bbox contribution of stroke becomes more complicated with miter joins.
        self.accumulate_bbox(|| shape.bounding_box() + Insets::uniform(width * 0.5));
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, false);
        self.encode_brush(&brush);
    }

    fn stroke_styled(
        &mut self,
        _shape: impl Shape,
        _brush: &impl IntoBrush<Self>,
        _width: f64,
        _style: &StrokeStyle,
    ) {
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        // Note: we might get a good speedup from using an approximate bounding box.
        // Perhaps that should be added to kurbo.
        self.accumulate_bbox(|| shape.bounding_box());
        let path = shape.path_elements(TOLERANCE);
        self.set_fill_mode(FillMode::Nonzero);
        self.encode_path(path, true);
        self.encode_brush(&brush);
    }

    fn fill_even_odd(&mut self, _shape: impl Shape, _brush: &impl IntoBrush<Self>) {}

    fn clip(&mut self, shape: impl Shape) {
        self.set_fill_mode(FillMode::Nonzero);
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, true);
        let begin_ix = self.elements.len();
        self.elements.push(Element::BeginClip(Clip {
            bbox: Default::default(),
        }));
        if self.clip_stack.len() >= MAX_BLEND_STACK {
            panic!("Maximum clip/blend stack size {} exceeded", MAX_BLEND_STACK);
        }
        self.clip_stack.push(ClipElement {
            bbox: None,
            begin_ix,
        });
        self.path_count += 1;
        if let Some(tos) = self.state_stack.last_mut() {
            tos.n_clip += 1;
        }
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        layout.draw_text(self, pos.into());
    }

    fn save(&mut self) -> Result<(), Error> {
        self.state_stack.push(State {
            rel_transform: Affine::default(),
            transform: self.cur_transform,
            n_clip: 0,
        });
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Error> {
        if let Some(state) = self.state_stack.pop() {
            if state.rel_transform != Affine::default() {
                let a_inv = state.rel_transform.inverse();
                self.encode_transform(to_scene_transform(a_inv));
            }
            self.cur_transform = state.transform;
            for _ in 0..state.n_clip {
                self.pop_clip();
            }
            Ok(())
        } else {
            Err(Error::StackUnbalance)
        }
    }

    fn finish(&mut self) -> Result<(), Error> {
        for _ in 0..self.clip_stack.len() {
            self.pop_clip();
        }
        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.encode_transform(to_scene_transform(transform));
        if let Some(tos) = self.state_stack.last_mut() {
            tos.rel_transform *= transform;
        }
        self.cur_transform *= transform;
    }

    fn make_image(
        &mut self,
        _width: usize,
        _height: usize,
        _buf: &[u8],
        _format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        Ok(PietGpuImage)
    }

    fn draw_image(
        &mut self,
        _image: &Self::Image,
        _rect: impl Into<Rect>,
        _interp: InterpolationMode,
    ) {
    }

    fn draw_image_area(
        &mut self,
        _image: &Self::Image,
        _src_rect: impl Into<Rect>,
        _dst_rect: impl Into<Rect>,
        _interp: InterpolationMode,
    ) {
    }

    fn blurred_rect(&mut self, _rect: Rect, _blur_radius: f64, _brush: &impl IntoBrush<Self>) {}

    fn current_transform(&self) -> Affine {
        self.cur_transform
    }

    fn with_save(&mut self, f: impl FnOnce(&mut Self) -> Result<(), Error>) -> Result<(), Error> {
        self.save()?;
        // Always try to restore the stack, even if `f` errored.
        f(self).and(self.restore())
    }
}

impl PietGpuRenderContext {
    fn encode_line_seg(&mut self, seg: LineSeg) {
        self.elements.push(Element::Line(seg));
        self.pathseg_count += 1;
    }

    fn encode_quad_seg(&mut self, seg: QuadSeg) {
        self.elements.push(Element::Quad(seg));
        self.pathseg_count += 1;
    }

    fn encode_cubic_seg(&mut self, seg: CubicSeg) {
        self.elements.push(Element::Cubic(seg));
        self.pathseg_count += 1;
    }

    fn encode_path(&mut self, path: impl Iterator<Item = PathEl>, is_fill: bool) {
        if is_fill {
            self.encode_path_inner(
                path.flat_map(|el| {
                    match el {
                        PathEl::MoveTo(..) => Some(PathEl::ClosePath),
                        _ => None,
                    }
                    .into_iter()
                    .chain(Some(el))
                })
                .chain(Some(PathEl::ClosePath)),
            )
        } else {
            self.encode_path_inner(path)
        }
    }

    fn encode_path_inner(&mut self, path: impl Iterator<Item = PathEl>) {
        let flatten = false;
        if flatten {
            let mut start_pt = None;
            let mut last_pt = None;
            piet::kurbo::flatten(path, TOLERANCE, |el| {
                match el {
                    PathEl::MoveTo(p) => {
                        let scene_pt = to_f32_2(p);
                        start_pt = Some(scene_pt);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::LineTo(p) => {
                        let scene_pt = to_f32_2(p);
                        let seg = LineSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_pt,
                        };
                        self.encode_line_seg(seg);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::ClosePath => {
                        if let (Some(start), Some(last)) = (start_pt.take(), last_pt.take()) {
                            if last != start {
                                let seg = LineSeg {
                                    p0: last,
                                    p1: start,
                                };
                                self.encode_line_seg(seg);
                            }
                        }
                    }
                    _ => (),
                }
                //println!("{:?}", el);
            });
        } else {
            let mut start_pt = None;
            let mut last_pt = None;
            for el in path {
                match el {
                    PathEl::MoveTo(p) => {
                        let scene_pt = to_f32_2(p);
                        start_pt = Some(scene_pt);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::LineTo(p) => {
                        let scene_pt = to_f32_2(p);
                        let seg = LineSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_pt,
                        };
                        self.encode_line_seg(seg);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::QuadTo(p1, p2) => {
                        let scene_p1 = to_f32_2(p1);
                        let scene_p2 = to_f32_2(p2);
                        let seg = QuadSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_p1,
                            p2: scene_p2,
                        };
                        self.encode_quad_seg(seg);
                        last_pt = Some(scene_p2);
                    }
                    PathEl::CurveTo(p1, p2, p3) => {
                        let scene_p1 = to_f32_2(p1);
                        let scene_p2 = to_f32_2(p2);
                        let scene_p3 = to_f32_2(p3);
                        let seg = CubicSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_p1,
                            p2: scene_p2,
                            p3: scene_p3,
                        };
                        self.encode_cubic_seg(seg);
                        last_pt = Some(scene_p3);
                    }
                    PathEl::ClosePath => {
                        if let (Some(start), Some(last)) = (start_pt.take(), last_pt.take()) {
                            if last != start {
                                let seg = LineSeg {
                                    p0: last,
                                    p1: start,
                                };
                                self.encode_line_seg(seg);
                            }
                        }
                    }
                }
                //println!("{:?}", el);
            }
        }
    }

    fn pop_clip(&mut self) {
        let tos = self.clip_stack.pop().unwrap();
        let bbox = tos.bbox.unwrap_or_default();
        let bbox_f32_4 = rect_to_f32_4(bbox);
        self.elements
            .push(Element::EndClip(Clip { bbox: bbox_f32_4 }));
        self.path_count += 1;
        if let Element::BeginClip(begin_clip) = &mut self.elements[tos.begin_ix] {
            begin_clip.bbox = bbox_f32_4;
        } else {
            unreachable!("expected BeginClip, not found");
        }
        if let Some(bbox) = tos.bbox {
            self.union_bbox(bbox);
        }
    }

    /// Accumulate a bbox.
    ///
    /// The bbox is given lazily as a closure, relative to the current transform.
    /// It's lazy because we don't need to compute it unless we're inside a clip.
    fn accumulate_bbox(&mut self, f: impl FnOnce() -> Rect) {
        if !self.clip_stack.is_empty() {
            let bbox = f();
            let bbox = self.cur_transform.transform_rect_bbox(bbox);
            self.union_bbox(bbox);
        }
    }

    /// Accumulate an absolute bbox.
    ///
    /// The bbox is given already transformed into surface coordinates.
    fn union_bbox(&mut self, bbox: Rect) {
        if let Some(tos) = self.clip_stack.last_mut() {
            tos.bbox = if let Some(old_bbox) = tos.bbox {
                Some(old_bbox.union(bbox))
            } else {
                Some(bbox)
            };
        }
    }

    pub(crate) fn append_path_encoder(&mut self, path: &PathEncoder) {
        let elements = path.elements();
        self.elements.extend(elements.iter().cloned());
        self.pathseg_count += path.n_segs();
    }

    pub(crate) fn fill_glyph(&mut self, rgba_color: u32) {
        let fill = FillColor { rgba_color };
        self.elements.push(Element::FillColor(fill));
        self.path_count += 1;
    }

    /// Bump the path count when rendering a color emoji.
    pub(crate) fn bump_n_paths(&mut self, n_paths: usize) {
        self.path_count += n_paths;
    }

    pub(crate) fn encode_transform(&mut self, transform: Transform) {
        self.elements.push(Element::Transform(transform));
        self.trans_count += 1;
    }

    fn encode_brush(&mut self, brush: &PietGpuBrush) {
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                let fill = FillColor {
                    rgba_color: *rgba_color,
                };
                self.elements.push(Element::FillColor(fill));
                self.path_count += 1;
            }
            PietGpuBrush::LinGradient(lin) => {
                let fill_lin = FillLinGradient {
                    index: lin.ramp_id,
                    p0: lin.start,
                    p1: lin.end,
                };
                self.elements.push(Element::FillLinGradient(fill_lin));
                self.path_count += 1;
            }
        }
    }
}

impl IntoBrush<PietGpuRenderContext> for PietGpuBrush {
    fn make_brush<'b>(
        &'b self,
        _piet: &mut PietGpuRenderContext,
        _bbox: impl FnOnce() -> Rect,
    ) -> std::borrow::Cow<'b, PietGpuBrush> {
        Cow::Borrowed(self)
    }
}

pub(crate) fn to_f32_2(point: Point) -> [f32; 2] {
    [point.x as f32, point.y as f32]
}

fn rect_to_f32_4(rect: Rect) -> [f32; 4] {
    [
        rect.x0 as f32,
        rect.y0 as f32,
        rect.x1 as f32,
        rect.y1 as f32,
    ]
}

fn to_scene_transform(transform: Affine) -> Transform {
    let c = transform.as_coeffs();
    Transform {
        mat: [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32],
        translate: [c[4] as f32, c[5] as f32],
    }
}

fn to_srgb(f: f64) -> f64 {
    if f <= 0.0031308 {
        f * 12.92
    } else {
        let a = 0.055;
        (1. + a) * f64::powf(f, f64::recip(2.4)) - a
    }
}

fn from_srgb(f: f64) -> f64 {
    if f <= 0.04045 {
        f / 12.92
    } else {
        let a = 0.055;
        f64::powf((f + a) * f64::recip(1. + a), 2.4)
    }
}
