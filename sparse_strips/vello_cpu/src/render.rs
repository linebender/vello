// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::RenderMode;
use crate::dispatch::Dispatcher;
#[cfg(feature = "multithreading")]
use crate::dispatch::multi_threaded::MultiThreadedDispatcher;
use crate::dispatch::single_threaded::SingleThreadedDispatcher;
use crate::kurbo::{PathEl, Point};
use alloc::boxed::Box;
#[cfg(feature = "text")]
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::pixmap::Pixmap;

#[cfg(feature = "text")]
use vello_common::{
    color::{AlphaColor, Srgb},
    colr::{ColrPainter, ColrRenderer},
    glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph},
};

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;
/// A render context.
#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) temp_path: BezPath,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    #[cfg_attr(
        not(feature = "text"),
        allow(dead_code, reason = "used when the `text` feature is enabled")
    )]
    pub(crate) level: Level,
    dispatcher: Box<dyn Dispatcher>,
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The number of worker threads that should be used for rendering. Only has an effect
    /// if the `multithreading` feature is active.
    pub num_threads: u16,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::new(),
            #[cfg(feature = "multithreading")]
            num_threads: std::thread::available_parallelism()
                .unwrap()
                .get()
                .saturating_sub(1) as u16,
            #[cfg(not(feature = "multithreading"))]
            num_threads: 0,
        }
    }
}

impl RenderContext {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let settings = RenderSettings::default();
        Self::new_inner(width, height, settings.num_threads, settings.level)
    }

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: &RenderSettings) -> Self {
        Self::new_inner(width, height, settings.num_threads, settings.level)
    }

    fn new_inner(width: u16, height: u16, num_threads: u16, level: Level) -> Self {
        #[cfg(feature = "multithreading")]
        let dispatcher: Box<dyn Dispatcher> = if num_threads == 0 {
            Box::new(SingleThreadedDispatcher::new(width, height, level))
        } else {
            Box::new(MultiThreadedDispatcher::new(
                width,
                height,
                num_threads,
                level,
            ))
        };

        #[cfg(not(feature = "multithreading"))]
        let dispatcher: Box<dyn Dispatcher> = {
            let _ = num_threads;
            Box::new(SingleThreadedDispatcher::new(width, height, level))
        };

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
        let temp_path = BezPath::new();

        Self {
            width,
            height,
            dispatcher,
            transform,
            paint,
            paint_transform,
            fill_rule,
            level,
            stroke,
            temp_path,
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
        let paint = self.encode_current_paint();
        self.dispatcher
            .fill_path(path, self.fill_rule, self.transform, paint);
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        let paint = self.encode_current_paint();
        self.dispatcher
            .stroke_path(path, &self.stroke, self.transform, paint);
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        // Don't use `rect.to_path` here, because it will perform a new allocation, which
        // profiling showed can become a bottleneck for many small rectangles.
        // TODO: Generalize this so that for example `blurred_rectangle` and other places
        // can also profit from this.
        self.temp_path.truncate(0);
        self.temp_path
            .push(PathEl::MoveTo(Point::new(rect.x0, rect.y0)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x1, rect.y0)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x1, rect.y1)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x0, rect.y1)));
        self.temp_path.push(PathEl::ClosePath);

        let paint = self.encode_current_paint();
        self.dispatcher
            .fill_path(&self.temp_path, self.fill_rule, self.transform, paint);
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
        self.dispatcher.fill_path(
            &inflated_rect.to_path(0.1),
            Fill::NonZero,
            self.transform,
            paint,
        );
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run(&mut self, font: &crate::peniko::Font) -> GlyphRunBuilder<'_, Self> {
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
        let mask = mask.and_then(|m| {
            if m.width() != self.width || m.height() != self.height {
                None
            } else {
                Some(m)
            }
        });

        let blend_mode = blend_mode.unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver));
        let opacity = opacity.unwrap_or(1.0);

        self.dispatcher.push_layer(
            clip_path,
            self.fill_rule,
            self.transform,
            blend_mode,
            opacity,
            mask,
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
        self.dispatcher.pop_layer();
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
        self.dispatcher.reset();
        self.encoded_paints.clear();
        self.reset_transform();
        self.reset_paint_transform();
    }

    /// Flush any pending operations.
    ///
    /// This is a no-op when using the single-threaded render mode, and can be ignored.
    /// For multi-threaded rendering, you _have_ to call this before rasterizing, otherwise
    /// the program will panic.
    pub fn flush(&mut self) {
        self.dispatcher.flush();
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
        // TODO: Maybe we should move those checks into the dispatcher.
        let wide = self.dispatcher.wide();
        assert!(!wide.has_layers(), "some layers haven't been popped yet");
        assert_eq!(
            buffer.len(),
            (width as usize) * (height as usize) * 4,
            "provided width ({}) and height ({}) do not match buffer size ({})",
            width,
            height,
            buffer.len(),
        );

        self.dispatcher
            .rasterize(buffer, render_mode, width, height, &self.encoded_paints);
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
}

#[cfg(feature = "text")]
impl GlyphRenderer for RenderContext {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                let paint = self.encode_current_paint();
                self.dispatcher.fill_path(
                    glyph.path,
                    Fill::NonZero,
                    prepared_glyph.transform,
                    paint,
                );
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
                    crate::peniko::ImageQuality::High
                } else {
                    crate::peniko::ImageQuality::Medium
                };

                let image = vello_common::paint::Image {
                    pixmap: Arc::new(glyph.pixmap),
                    x_extend: crate::peniko::Extend::Pad,
                    y_extend: crate::peniko::Extend::Pad,
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
                    let settings = RenderSettings {
                        level: self.level,
                        num_threads: 0,
                    };

                    let mut ctx = Self::new_with(glyph.pix_width, glyph.pix_height, &settings);
                    let mut pix = Pixmap::new(glyph.pix_width, glyph.pix_height);

                    let mut colr_painter = ColrPainter::new(glyph, context_color, &mut ctx);
                    colr_painter.paint();

                    // Technically not necessary since we always render single-threaded, but just
                    // to be safe.
                    ctx.flush();
                    ctx.render_to_pixmap(&mut pix, RenderMode::OptimizeQuality);

                    pix
                };

                let image = vello_common::paint::Image {
                    pixmap: Arc::new(glyph_pixmap),
                    x_extend: crate::peniko::Extend::Pad,
                    y_extend: crate::peniko::Extend::Pad,
                    // Since the pixmap will already have the correct size, no need to
                    // use a different image quality here.
                    quality: crate::peniko::ImageQuality::Low,
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
                let paint = self.encode_current_paint();
                self.dispatcher.stroke_path(
                    glyph.path,
                    &self.stroke,
                    prepared_glyph.transform,
                    paint,
                );
            }
            GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
                // The definitions of COLR and bitmap glyphs can't meaningfully support being stroked.
                // (COLR's imaging model only has fills)
                self.fill_glyph(prepared_glyph);
            }
        }
    }
}

#[cfg(feature = "text")]
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

    fn fill_gradient(&mut self, gradient: crate::peniko::Gradient) {
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
    use vello_common::tile::Tile;

    #[test]
    fn clip_overflow() {
        let mut ctx = RenderContext::new(100, 100);

        for _ in 0..(usize::from(u16::MAX) + 1).div_ceil(usize::from(Tile::HEIGHT * Tile::WIDTH)) {
            ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
        }

        ctx.push_clip_layer(&Rect::new(20.0, 20.0, 180.0, 180.0).to_path(0.1));
        ctx.pop_layer();
    }
}
