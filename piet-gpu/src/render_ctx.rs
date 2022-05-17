// This should match the value in kernel4.comp for correct rendering.
const DO_SRGB_CONVERSION: bool = false;

use std::borrow::Cow;

use crate::encoder::GlyphEncoder;
use crate::stages::{Config, Transform};
use crate::MAX_BLEND_STACK;
use piet::kurbo::{Affine, PathEl, Point, Rect, Shape};
use piet::{
    Color, Error, FixedGradient, ImageFormat, InterpolationMode, IntoBrush, RenderContext,
    StrokeStyle,
};

use piet_gpu_hal::BufWrite;
use piet_gpu_types::encoder::{Encode, Encoder};
use piet_gpu_types::scene::Element;

use crate::gradient::{Colrv1RadialGradient, LinearGradient, RadialGradient, RampCache};
use crate::text::Font;
pub use crate::text::{PietGpuText, PietGpuTextLayout, PietGpuTextLayoutBuilder};
use crate::Blend;

pub struct PietGpuImage;

pub struct PietGpuRenderContext {
    encoder: Encoder,
    elements: Vec<Element>,
    // Will probably need direct accesss to hal Device to create images etc.
    inner_text: PietGpuText,
    stroke_width: f32,
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

    // Fields for new element processing pipeline below
    // TODO: delete old encoder, rename
    new_encoder: crate::encoder::Encoder,
}

#[derive(Clone)]
pub enum PietGpuBrush {
    Solid(u32),
    LinGradient(LinearGradient),
    RadGradient(RadialGradient),
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
    blend: Option<Blend>,
}

const TOLERANCE: f64 = 0.25;

impl PietGpuRenderContext {
    pub fn new() -> PietGpuRenderContext {
        let encoder = Encoder::new();
        let elements = Vec::new();
        let font = Font::new();
        let inner_text = PietGpuText::new(font);
        let stroke_width = -1.0;
        PietGpuRenderContext {
            encoder,
            elements,
            inner_text,
            stroke_width,
            path_count: 0,
            pathseg_count: 0,
            trans_count: 0,
            cur_transform: Affine::default(),
            state_stack: Vec::new(),
            clip_stack: Vec::new(),
            ramp_cache: RampCache::default(),
            new_encoder: crate::encoder::Encoder::new(),
        }
    }

    pub fn stage_config(&self) -> (Config, usize) {
        self.new_encoder.stage_config()
    }

    /// Number of draw objects.
    ///
    /// This is for the new element processing pipeline. It's not necessarily the
    /// same as the number of paths (as in the old pipeline), but it might take a
    /// while to sort that out.
    pub fn n_drawobj(&self) -> usize {
        self.new_encoder.n_drawobj()
    }

    /// Number of paths.
    pub fn n_path(&self) -> u32 {
        self.new_encoder.n_path()
    }

    pub fn n_pathseg(&self) -> u32 {
        self.new_encoder.n_pathseg()
    }

    pub fn n_pathtag(&self) -> usize {
        self.new_encoder.n_pathtag()
    }

    pub fn n_transform(&self) -> usize {
        self.new_encoder.n_transform()
    }

    pub fn n_clip(&self) -> u32 {
        self.new_encoder.n_clip()
    }

    pub fn write_scene(&self, buf: &mut BufWrite) {
        self.new_encoder.write_scene(buf);
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
            FixedGradient::Radial(rad) => {
                let rad = self.ramp_cache.add_radial_gradient(&rad);
                Ok(PietGpuBrush::RadGradient(rad))
            }
            _ => todo!("don't do radial gradients yet"),
        }
    }

    fn clear(&mut self, _color: Color) {}

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        self.encode_linewidth(width.abs() as f32);
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
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
        let path = shape.path_elements(TOLERANCE);
        self.encode_linewidth(-1.0);
        self.encode_path(path, true);
        self.encode_brush(&brush);
    }

    fn fill_even_odd(&mut self, _shape: impl Shape, _brush: &impl IntoBrush<Self>) {}

    fn clip(&mut self, shape: impl Shape) {
        self.encode_linewidth(-1.0);
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, true);
        self.new_encoder.begin_clip(None);
        if self.clip_stack.len() >= MAX_BLEND_STACK {
            panic!("Maximum clip/blend stack size {} exceeded", MAX_BLEND_STACK);
        }
        self.clip_stack.push(ClipElement { blend: None });
        if let Some(tos) = self.state_stack.last_mut() {
            tos.n_clip += 1;
        }
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        self.encode_linewidth(-1.0);
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
                self.encode_transform(Transform::from_kurbo(a_inv));
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
        self.encode_transform(Transform::from_kurbo(transform));
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
    pub fn blend(&mut self, shape: impl Shape, blend: Blend) {
        self.encode_linewidth(-1.0);
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, true);
        self.new_encoder.begin_clip(Some(blend));
        if self.clip_stack.len() >= MAX_BLEND_STACK {
            panic!("Maximum clip/blend stack size {} exceeded", MAX_BLEND_STACK);
        }
        self.clip_stack.push(ClipElement { blend: Some(blend) });
        if let Some(tos) = self.state_stack.last_mut() {
            tos.n_clip += 1;
        }
    }

    pub fn radial_gradient_colrv1(&mut self, rad: &Colrv1RadialGradient) -> PietGpuBrush {
        PietGpuBrush::RadGradient(self.ramp_cache.add_radial_gradient_colrv1(rad))
    }

    pub fn fill_transform(&mut self, shape: impl Shape, brush: &PietGpuBrush, transform: Affine) {
        let path = shape.path_elements(TOLERANCE);
        self.encode_linewidth(-1.0);
        self.encode_path(path, true);
        self.encode_transform(Transform::from_kurbo(transform));
        self.new_encoder.swap_last_tags();
        self.encode_brush(&brush);
        self.encode_transform(Transform::from_kurbo(transform.inverse()));
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
        let mut pe = self.new_encoder.path_encoder();
        for el in path {
            match el {
                PathEl::MoveTo(p) => {
                    let p = to_f32_2(p);
                    pe.move_to(p[0], p[1]);
                }
                PathEl::LineTo(p) => {
                    let p = to_f32_2(p);
                    pe.line_to(p[0], p[1]);
                }
                PathEl::QuadTo(p1, p2) => {
                    let p1 = to_f32_2(p1);
                    let p2 = to_f32_2(p2);
                    pe.quad_to(p1[0], p1[1], p2[0], p2[1]);
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    let p1 = to_f32_2(p1);
                    let p2 = to_f32_2(p2);
                    let p3 = to_f32_2(p3);
                    pe.cubic_to(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]);
                }
                PathEl::ClosePath => pe.close_path(),
            }
        }
        pe.path();
        let n_pathseg = pe.n_pathseg();
        self.new_encoder.finish_path(n_pathseg);
    }

    fn pop_clip(&mut self) {
        let tos = self.clip_stack.pop().unwrap();
        self.new_encoder.end_clip(tos.blend);
    }

    pub(crate) fn encode_glyph(&mut self, glyph: &GlyphEncoder) {
        self.new_encoder.encode_glyph(glyph);
    }

    pub(crate) fn fill_glyph(&mut self, rgba_color: u32) {
        self.new_encoder.fill_color(rgba_color);
    }

    pub(crate) fn encode_transform(&mut self, transform: Transform) {
        self.new_encoder.transform(transform);
    }

    fn encode_linewidth(&mut self, linewidth: f32) {
        if self.stroke_width != linewidth {
            self.new_encoder.linewidth(linewidth);
            self.stroke_width = linewidth;
        }
    }

    fn encode_brush(&mut self, brush: &PietGpuBrush) {
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                self.new_encoder.fill_color(*rgba_color);
            }
            PietGpuBrush::LinGradient(lin) => {
                self.new_encoder
                    .fill_lin_gradient(lin.ramp_id, lin.start, lin.end);
            }
            PietGpuBrush::RadGradient(rad) => {
                self.new_encoder
                    .fill_rad_gradient(rad.ramp_id, rad.start, rad.end, rad.r0, rad.r1);
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

fn to_srgb(f: f64) -> f64 {
    if DO_SRGB_CONVERSION {
        if f <= 0.0031308 {
            f * 12.92
        } else {
            let a = 0.055;
            (1. + a) * f64::powf(f, f64::recip(2.4)) - a
        }
    } else {
        f
    }
}

fn from_srgb(f: f64) -> f64 {
    if DO_SRGB_CONVERSION {
        if f <= 0.04045 {
            f / 12.92
        } else {
            let a = 0.055;
            f64::powf((f + a) * f64::recip(1. + a), 2.4)
        }
    } else {
        f
    }
}
