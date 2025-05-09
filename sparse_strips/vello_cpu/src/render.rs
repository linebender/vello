// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::RenderMode;
use crate::fine::{Fine, FineType};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::coarse::Wide;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::colr::{ColrPainter, ColrRenderer};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Image, Paint, PaintType};
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Gradient, Mix};
use vello_common::peniko::{Font, ImageQuality};
use vello_common::pixmap::Pixmap;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, peniko, strip};

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;
/// A render context.
#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
}

impl RenderContext {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let wide = Wide::new(width, height);

        let alphas = vec![];
        let line_buf = vec![];
        let tiles = Tiles::new();
        let strip_buf = vec![];

        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
        let paint_transform = Affine::IDENTITY;
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        let encoded_paints = vec![];

        Self {
            width,
            height,
            wide,
            alphas,
            line_buf,
            tiles,
            strip_buf,
            transform,
            paint,
            paint_transform,
            fill_rule,
            stroke,
            encoded_paints,
        }
    }

    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => {
                // TODO: Add caching?
                g.encode_into(
                    &mut self.encoded_paints,
                    self.transform * self.paint_transform,
                )
            }
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints,
                self.transform * self.paint_transform,
            ),
        }
    }

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(self.fill_rule, paint);
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(Fill::NonZero, paint);
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Fill a blurred rectangle with the given radius and standard deviation.
    ///
    /// Note that this only works properly if the current paint is set to a solid color.
    /// If not, it will fall back to using black as the fill color.
    pub fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        let color = match self.paint {
            PaintType::Solid(s) => s,
            // Fallback to black when attempting to blur a rectangle with an image/gradient paint
            _ => BLACK,
        };

        let blurred_rect = BlurredRoundedRectangle {
            rect: *rect,
            color,
            radius,
            std_dev,
        };

        // The actual rectangle we paint needs to be larger so that the blurring effect
        // is not cut off.
        // The impulse response of a gaussian filter is infinite.
        // For performance reason we cut off the filter at some extent where the response is close to zero.
        let kernel_size = 2.5 * std_dev;
        let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
        let transform = self.transform * self.paint_transform;

        let paint = blurred_rect.encode_into(&mut self.encoded_paints, transform);
        flatten::fill(&inflated_rect.to_path(0.1), transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, paint);
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Push a new layer with the given properties.
    ///
    /// Note that the mask, if provided, needs to have the same size as the render context. Otherwise,
    /// it will be ignored. In addition to that, the mask will not be affected by the current
    /// transformation matrix in place.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        let clip = if let Some(c) = clip_path {
            flatten::fill(c, self.transform, &mut self.line_buf);
            self.make_strips(self.fill_rule);
            Some((self.strip_buf.as_slice(), self.fill_rule))
        } else {
            None
        };

        let mask = mask.and_then(|m| {
            if m.width() != self.width || m.height() != self.height {
                None
            } else {
                Some(m)
            }
        });

        self.wide.push_layer(
            clip,
            blend_mode.unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
            mask,
            opacity.unwrap_or(1.0),
        );
    }

    /// Push a new clip layer.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None);
    }

    /// Push a new blend layer.
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None);
    }

    /// Push a new opacity layer.
    pub fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(None, None, Some(opacity), None);
    }

    /// Push a new mask layer.
    ///
    /// Note that the mask, if provided, needs to have the same size as the render context. Otherwise,
    /// it will be ignored. In addition to that, the mask will not be affected by the current
    /// transformation matrix in place.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask));
    }

    /// Pop the last-pushed layer.
    pub fn pop_layer(&mut self) {
        self.wide.pop_layer();
    }

    /// Set the current stroke.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the current paint.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
    }

    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.paint_transform = Affine::IDENTITY;
    }

    /// Set the current fill rule.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the current transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the current transform.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Reset the render context.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.alphas.clear();
        self.strip_buf.clear();
        self.wide.reset();
    }

    /// Render the current context into a buffer.
    /// The buffer is expected to be in premultiplied RGBA8 format with length `width * height * 4`
    pub fn render_to_buffer(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        render_mode: RenderMode,
    ) {
        assert!(
            !self.wide.has_layers(),
            "some layers haven't been popped yet"
        );
        assert_eq!(
            buffer.len(),
            (width as usize) * (height as usize) * 4,
            "provided width ({}) and height ({}) do not match buffer size ({})",
            width,
            height,
            buffer.len(),
        );

        match render_mode {
            RenderMode::OptimizeSpeed => {
                let mut fine = Fine::<u8>::new(width, height);
                self.do_fine(buffer, &mut fine);
            }
            RenderMode::OptimizeQuality => {
                let mut fine = Fine::<f32>::new(width, height);
                self.do_fine(buffer, &mut fine);
            }
        }
    }

    fn do_fine<F: FineType>(&self, buffer: &mut [u8], fine: &mut Fine<F>) {
        let width_tiles = self.wide.width_tiles();
        let height_tiles = self.wide.height_tiles();
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let wtile = self.wide.get(x, y);
                fine.set_coords(x, y);

                fine.clear(F::extract_color(&wtile.bg));
                for cmd in &wtile.cmds {
                    fine.run_cmd(cmd, &self.alphas, &self.encoded_paints);
                }
                fine.pack(buffer);
            }
        }
    }

    /// Render the current context into a pixmap.
    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap, render_mode: RenderMode) {
        let width = pixmap.width();
        let height = pixmap.height();
        self.render_to_buffer(pixmap.data_as_u8_slice_mut(), width, height, render_mode);
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.make_strips(fill_rule);
        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }

    fn make_strips(&mut self, fill_rule: Fill) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );
    }
}

impl GlyphRenderer for RenderContext {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::fill(glyph.path, prepared_glyph.transform, &mut self.line_buf);
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
            GlyphType::Bitmap(glyph) => {
                // We need to change the state of the render context
                // to render the bitmap, but don't want to pollute the context,
                // so simulate a `save` and `restore` operation.
                let old_transform = self.transform;
                let old_paint = self.paint.clone();

                // If we scale down by a large factor, fall back to cubic scaling.
                let quality = if prepared_glyph.transform.as_coeffs()[0] < 0.5
                    || prepared_glyph.transform.as_coeffs()[3] < 0.5
                {
                    ImageQuality::High
                } else {
                    ImageQuality::Medium
                };

                let image = Image {
                    pixmap: Arc::new(glyph.pixmap),
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    quality,
                };

                self.set_paint(image);
                self.set_transform(prepared_glyph.transform);
                self.fill_rect(&glyph.area);

                // Restore the state.
                self.set_paint(old_paint);
                self.transform = old_transform;
            }
            GlyphType::Colr(glyph) => {
                // Same as for bitmap glyphs, save the state and restore it later on.
                let old_transform = self.transform;
                let old_paint = self.paint.clone();
                let context_color = match old_paint {
                    PaintType::Solid(s) => s,
                    _ => BLACK,
                };

                let area = glyph.area;

                let glyph_pixmap = {
                    let mut ctx = Self::new(glyph.pix_width, glyph.pix_height);
                    let mut pix = Pixmap::new(glyph.pix_width, glyph.pix_height);

                    let mut colr_painter = ColrPainter::new(glyph, context_color, &mut ctx);
                    colr_painter.paint();

                    ctx.render_to_pixmap(&mut pix, RenderMode::OptimizeQuality);

                    pix
                };

                let image = Image {
                    pixmap: Arc::new(glyph_pixmap),
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    // Since the pixmap will already have the correct size, no need to
                    // use a different image quality here.
                    quality: ImageQuality::Low,
                };

                self.set_paint(image);
                self.set_transform(prepared_glyph.transform);
                self.fill_rect(&area);

                // Restore the state.
                self.set_paint(old_paint);
                self.transform = old_transform;
            }
        }
    }

    fn stroke_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::stroke(
                    glyph.path,
                    &self.stroke,
                    prepared_glyph.transform,
                    &mut self.line_buf,
                );
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
            GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
                // The definitions of COLR and bitmap glyphs can't meaningfully support being stroked.
                // (COLR's imaging model only has fills)
                self.fill_glyph(prepared_glyph);
            }
        }
    }
}

impl ColrRenderer for RenderContext {
    fn push_clip_layer(&mut self, clip: &BezPath) {
        Self::push_clip_layer(self, clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }

    fn fill_solid(&mut self, color: AlphaColor<Srgb>) {
        self.set_paint(color);
        self.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.width),
            f64::from(self.height),
        ));
    }

    fn fill_gradient(&mut self, gradient: Gradient) {
        self.set_paint(gradient);
        self.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(self.width),
            f64::from(self.height),
        ));
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        Self::set_paint_transform(self, affine);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }
}

#[cfg(test)]
mod tests {
    use crate::RenderContext;
    use vello_common::kurbo::{Rect, Shape};

    #[test]
    fn reset_render_context() {
        let mut ctx = RenderContext::new(100, 100);
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        ctx.fill_rect(&rect);

        assert!(!ctx.line_buf.is_empty());
        assert!(!ctx.strip_buf.is_empty());
        assert!(!ctx.alphas.is_empty());

        ctx.reset();

        assert!(ctx.line_buf.is_empty());
        assert!(ctx.strip_buf.is_empty());
        assert!(ctx.alphas.is_empty());
    }

    #[test]
    fn clip_overflow() {
        let mut ctx = RenderContext::new(100, 100);

        ctx.alphas
            .extend(core::iter::repeat_n(255, u16::MAX as usize + 1));

        ctx.push_clip_layer(&Rect::new(20.0, 20.0, 180.0, 180.0).to_path(0.1));
        ctx.pop_layer();
    }
}
