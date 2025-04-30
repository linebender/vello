use vello_api::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_api::mask::Mask;
use vello_api::paint::PaintType;
use vello_api::peniko::{BlendMode, Fill, Font};
use vello_api::pixmap::Pixmap;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder};
use vello_cpu::RenderContext;

pub(crate) trait Renderer: Sized + GlyphRenderer {
    fn new(width: u16, height: u16) -> Self;
    fn fill_path(&mut self, path: &BezPath);
    fn stroke_path(&mut self, path: &BezPath);
    fn fill_rect(&mut self, rect: &Rect);
    fn stroke_rect(&mut self, rect: &Rect);
    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self>;
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<u8>,
        mask: Option<Mask>,
    );
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_opacity_layer(&mut self, opacity: u8);
    fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    fn render_to_pixmap(&self, pixmap: &mut Pixmap);
}

impl Renderer for RenderContext {
    fn new(width: u16, height: u16) -> Self {
       Self::new(width, height) 
    }

    fn fill_path(&mut self, path: &BezPath) {
       Self::fill_path(self, path) 
    }

    fn stroke_path(&mut self, path: &BezPath) {
       Self::stroke_path(self, path) 
    }

    fn fill_rect(&mut self, rect: &Rect) {
       Self::fill_rect(self, rect) 
    }

    fn stroke_rect(&mut self, rect: &Rect) {
       Self::stroke_rect(self, rect) 
    }

    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
       Self::glyph_run(self, font) 
    }

    fn push_layer(&mut self, clip_path: Option<&BezPath>, blend_mode: Option<BlendMode>, opacity: Option<u8>, mask: Option<Mask>) {
       Self::push_layer(self, clip_path, blend_mode, opacity, mask) 
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
       Self::push_clip_layer(self, path) 
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
       Self::push_blend_layer(self, blend_mode) 
    }

    fn push_opacity_layer(&mut self, opacity: u8) {
       Self::push_opacity_layer(self, opacity) 
    }

    fn push_mask_layer(&mut self, mask: Mask) {
       Self::push_mask_layer(self, mask) 
    }

    fn pop_layer(&mut self) {
       Self::pop_layer(self) 
    }

    fn set_stroke(&mut self, stroke: Stroke) {
       Self::set_stroke(self, stroke) 
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
       Self::set_paint(self, paint) 
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
       Self::set_fill_rule(self, fill_rule) 
    }

    fn set_transform(&mut self, transform: Affine) {
       Self::set_transform(self, transform) 
    }

    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
       Self::render_to_pixmap(self, pixmap) 
    }
}