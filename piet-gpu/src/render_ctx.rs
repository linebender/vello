use std::{borrow::Cow, ops::RangeBounds};

use piet::{
    HitTestPosition,
    kurbo::{Affine, Insets, PathEl, Point, Rect, Shape, Size}, TextAttribute, TextStorage,
};
use piet::{
    Color, Error, FixedGradient, FontFamily, HitTestPoint, ImageFormat, InterpolationMode,
    IntoBrush, LineMetric, RenderContext, StrokeStyle, Text, TextLayout, TextLayoutBuilder,
};

use piet_gpu_types::encoder::{Encode, Encoder};
use piet_gpu_types::scene::{
    Clip, CubicSeg, Element, FillColor, LineSeg, QuadSeg, SetFillMode, SetLineWidth, Transform,
};

pub struct PietGpuImage;

#[derive(Clone)]
pub struct PietGpuTextLayout;

pub struct PietGpuTextLayoutBuilder;

#[derive(Clone)]
pub struct PietGpuText;

pub struct PietGpuRenderContext {
    encoder: Encoder,
    // The elements are drawn in a mostly front-to-back order, but the API accepts elements in
    // back-to-front order instead. Therefore, in each callback we first buffer the commands first,
    // then reverse it in flush() and store it in elements. In get_scene_buf(), we reverse the order
    // of element again, therefore the overall drawing order is reversed, but the structure of each
    // command remains to be the order specified in individual functions.
    element_tmp: Vec<Element>,
    elements: Vec<Element>,
    // Will probably need direct accesss to hal Device to create images etc.
    inner_text: PietGpuText,
    stroke_width: Option<f32>,
    fill_mode: Option<FillMode>,
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
}

#[derive(Clone)]
pub enum PietGpuBrush {
    Solid(u32),
    Gradient,
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
    encoded_path: Vec<Element>,
}

#[derive(Clone, Copy, PartialEq)]
enum FillMode {
    // Fill path according to the non-zero winding rule.
    Nonzero = 0,
    // Fill stroked path.
    Stroke = 1,
}

const TOLERANCE: f64 = 0.25;

impl PietGpuRenderContext {
    pub fn new() -> PietGpuRenderContext {
        let encoder = Encoder::new();
        let element_tmp = Vec::new();
        let elements = Vec::new();
        let inner_text = PietGpuText;
        PietGpuRenderContext {
            encoder,
            element_tmp,
            elements,
            inner_text,
            stroke_width: None,
            fill_mode: None,
            path_count: 0,
            pathseg_count: 0,
            trans_count: 0,
            cur_transform: Affine::default(),
            state_stack: Vec::new(),
            clip_stack: Vec::new(),
        }
    }

    pub fn get_scene_buf(&mut self) -> &[u8] {
        // TODO: iterator based reverse
        self.elements.reverse();
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

    fn flush(&mut self) {
        self.elements.extend(self.element_tmp.drain(..).rev());
    }
}

fn set_fill_mode(ctx: &mut PietGpuRenderContext, fill_mode: FillMode) {
    if ctx.fill_mode != Some(fill_mode) {
        if let Some(cur_mode) = ctx.fill_mode {
            ctx.element_tmp
                .push(Element::SetFillMode(SetFillMode { fill_mode: cur_mode as u32 }));
        }
        ctx.fill_mode = Some(fill_mode);
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
        let premul = Color::rgba(to_srgb(from_srgb(r) * a), to_srgb(from_srgb(g) * a), to_srgb(from_srgb(b) * a), a);
        PietGpuBrush::Solid(premul.as_rgba_u32())
    }

    fn gradient(&mut self, _gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Error> {
        Ok(Self::Brush::Gradient)
    }

    fn clear(&mut self, _color: Color) {}

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        let width_f32 = width as f32;
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                // Note: the bbox contribution of stroke becomes more complicated with miter joins.
                self.accumulate_bbox(|| shape.bounding_box() + Insets::uniform(width * 0.5));
                let path = shape.path_elements(TOLERANCE);
                self.encode_path(path, false);
                let stroke = FillColor { rgba_color };
                self.element_tmp.push(Element::FillColor(stroke));
                self.path_count += 1;
            }
            _ => (),
        }
        // Push the stroke width for the paths drawn *after* us, not *ours*. The last path is
        // handled by finish().
        if self.stroke_width != Some(width_f32) {
            if let Some(stroke_width) = self.stroke_width {
                self.element_tmp
                    .push(Element::SetLineWidth(SetLineWidth { width: stroke_width }));
            }
            self.stroke_width = Some(width_f32);
        }
        set_fill_mode(self, FillMode::Stroke);
        self.flush();
    }

    fn stroke_styled(
        &mut self,
        _shape: impl Shape,
        _brush: &impl IntoBrush<Self>,
        _width: f64,
        _style: &StrokeStyle,
    ) {}

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        if let PietGpuBrush::Solid(rgba_color) = brush {
            // Note: we might get a good speedup from using an approximate bounding box.
            // Perhaps that should be added to kurbo.
            self.accumulate_bbox(|| shape.bounding_box());
            let path = shape.path_elements(TOLERANCE);
            self.encode_path(path, true);
            let fill = FillColor { rgba_color };
            self.element_tmp.push(Element::FillColor(fill));
            self.path_count += 1;
            set_fill_mode(self, FillMode::Nonzero);
        }
        self.flush();
    }

    fn fill_even_odd(&mut self, _shape: impl Shape, _brush: &impl IntoBrush<Self>) {}

    fn clip(&mut self, shape: impl Shape) {
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, true);
        let encoded_path = std::mem::take(&mut self.element_tmp);
        // WARNING: Be careful to not push anything after Element::Clip
        let begin_ix = self.elements.len();
        self.element_tmp.push(Element::EndClip(Clip {
            bbox: Default::default(),
        }));
        self.clip_stack.push(ClipElement {
            bbox: None,
            begin_ix,
            encoded_path,
        });
        self.path_count += 1;
        if let Some(tos) = self.state_stack.last_mut() {
            tos.n_clip += 1;
        }
        self.flush();
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(&mut self, _layout: &Self::TextLayout, _pos: impl Into<Point>) {}

    fn save(&mut self) -> Result<(), Error> {
        self.state_stack.push(State {
            rel_transform: Affine::default(),
            transform: self.cur_transform,
            n_clip: 0,
        });
        self.flush();
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Error> {
        if let Some(state) = self.state_stack.pop() {
            if state.rel_transform != Affine::default() {
                let a_inv = state.rel_transform.inverse();
                self.element_tmp
                    .push(Element::Transform(to_scene_transform(a_inv)));
                self.trans_count += 1;
            }
            self.cur_transform = state.transform;
            for _ in 0..state.n_clip {
                self.pop_clip();
            }
            self.flush();
            Ok(())
        } else {
            Err(Error::StackUnbalance)
        }
    }

    fn finish(&mut self) -> Result<(), Error> {
        if let Some(stroke_width) = self.stroke_width.take() {
            self.elements
                .push(Element::SetLineWidth(SetLineWidth { width: stroke_width }));
        }
        if let Some(fill_mode) = self.fill_mode.take() {
            self.elements
                .push(Element::SetFillMode(SetFillMode { fill_mode: fill_mode as u32 }));
        }
        for _ in 0..self.state_stack.len() {
            self.restore()?;
        }
        for _ in 0..self.clip_stack.len() {
            self.pop_clip();
        }
        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.element_tmp
            .push(Element::Transform(to_scene_transform(transform)));
        self.trans_count += 1;
        if let Some(tos) = self.state_stack.last_mut() {
            tos.rel_transform *= transform;
        }
        self.cur_transform *= transform;
        self.flush();
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
    ) {}

    fn draw_image_area(
        &mut self,
        _image: &Self::Image,
        _src_rect: impl Into<Rect>,
        _dst_rect: impl Into<Rect>,
        _interp: InterpolationMode,
    ) {}

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
        self.element_tmp.push(Element::Line(seg));
        self.pathseg_count += 1;
    }

    fn encode_quad_seg(&mut self, seg: QuadSeg) {
        self.element_tmp.push(Element::Quad(seg));
        self.pathseg_count += 1;
    }

    fn encode_cubic_seg(&mut self, seg: CubicSeg) {
        self.element_tmp.push(Element::Cubic(seg));
        self.pathseg_count += 1;
    }

    fn encode_path(&mut self, path: impl Iterator<Item=PathEl>, is_fill: bool) {
        if is_fill {
            self.encode_path_inner(path.flat_map(|el| {
                match el {
                    PathEl::MoveTo(..) => {
                        Some(PathEl::ClosePath)
                    }
                    _ => None
                }.into_iter().chain(Some(el))
            }).chain(Some(PathEl::ClosePath)))
        } else {
            self.encode_path_inner(path)
        }
    }

    fn encode_path_inner(&mut self, path: impl Iterator<Item=PathEl>) {
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
        self.element_tmp.extend(tos.encoded_path);
        self.element_tmp
            .push(Element::BeginClip(Clip { bbox: bbox_f32_4 }));
        self.path_count += 1;
        if let Element::EndClip(end_clip) = &mut self.elements[tos.begin_ix] {
            end_clip.bbox = bbox_f32_4;
        } else {
            unreachable!("expected BeginClip, not found");
        }
        set_fill_mode(self, FillMode::Nonzero);
        if let Some(bbox) = tos.bbox {
            self.union_bbox(bbox);
        }
        self.flush();
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
}

impl Text for PietGpuText {
    type TextLayout = PietGpuTextLayout;
    type TextLayoutBuilder = PietGpuTextLayoutBuilder;

    fn load_font(&mut self, _data: &[u8]) -> Result<FontFamily, Error> {
        Ok(FontFamily::default())
    }

    fn new_text_layout(&mut self, _text: impl TextStorage) -> Self::TextLayoutBuilder {
        PietGpuTextLayoutBuilder
    }

    fn font_family(&mut self, _family_name: &str) -> Option<FontFamily> {
        Some(FontFamily::default())
    }
}

impl TextLayoutBuilder for PietGpuTextLayoutBuilder {
    type Out = PietGpuTextLayout;

    fn max_width(self, _width: f64) -> Self {
        self
    }

    fn alignment(self, _alignment: piet::TextAlignment) -> Self {
        self
    }

    fn default_attribute(self, _attribute: impl Into<TextAttribute>) -> Self {
        self
    }

    fn range_attribute(
        self,
        _range: impl RangeBounds<usize>,
        _attribute: impl Into<TextAttribute>,
    ) -> Self {
        self
    }

    fn build(self) -> Result<Self::Out, Error> {
        Ok(PietGpuTextLayout)
    }
}

impl TextLayout for PietGpuTextLayout {
    fn size(&self) -> Size {
        Size::ZERO
    }

    fn image_bounds(&self) -> Rect {
        Rect::ZERO
    }

    fn line_text(&self, _line_number: usize) -> Option<&str> {
        None
    }

    fn line_metric(&self, _line_number: usize) -> Option<LineMetric> {
        None
    }

    fn line_count(&self) -> usize {
        0
    }

    fn hit_test_point(&self, _point: Point) -> HitTestPoint {
        HitTestPoint::default()
    }

    fn hit_test_text_position(&self, _text_position: usize) -> HitTestPosition {
        HitTestPosition::default()
    }

    fn text(&self) -> &str {
        ""
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

fn to_f32_2(point: Point) -> [f32; 2] {
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