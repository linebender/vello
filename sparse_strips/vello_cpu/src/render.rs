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
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Stroke};
use vello_common::mask::Mask;
#[cfg(feature = "text")]
use vello_common::paint::ImageSource;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Fill};
use vello_common::pixmap::Pixmap;
use vello_common::recording::{PushLayerCommand, Recordable, Recorder, Recording, RenderCommand};
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
#[cfg(feature = "text")]
use vello_common::{
    color::{AlphaColor, Srgb},
    colr::{ColrPainter, ColrRenderer},
    glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph},
};

/// A render context.
#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) mask: Option<Mask>,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
    pub(crate) temp_path: BezPath,
    pub(crate) aliasing_threshold: Option<u8>,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    #[cfg_attr(
        not(feature = "text"),
        allow(dead_code, reason = "used when the `text` feature is enabled")
    )]
    pub(crate) render_settings: RenderSettings,
    dispatcher: Box<dyn Dispatcher>,
    #[cfg(feature = "text")]
    pub(crate) glyph_caches: Option<vello_common::glyph::GlyphCaches>,
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The number of worker threads that should be used for rendering. Only has an effect
    /// if the `multithreading` feature is active.
    pub num_threads: u16,
    /// Whether to prioritize speed or quality when rendering.
    ///
    /// For most cases (especially for real-time rendering), it is highly recommended to set
    /// this to `OptimizeSpeed`. If accuracy is a more significant concern (for example for visual
    /// regression testing), then you can set this to `OptimizeQuality`.
    ///
    /// Currently, the only difference this makes is that when choosing `OptimizeSpeed`, rasterization
    /// will happen using u8/u16, while `OptimizeQuality` will use a f32-based pipeline.
    pub render_mode: RenderMode,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::fallback()),
            #[cfg(feature = "multithreading")]
            num_threads: (std::thread::available_parallelism()
                .unwrap()
                .get()
                .saturating_sub(1) as u16)
                .min(8),
            #[cfg(not(feature = "multithreading"))]
            num_threads: 0,
            render_mode: RenderMode::OptimizeSpeed,
        }
    }
}

impl RenderContext {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        #[cfg(feature = "multithreading")]
        let dispatcher: Box<dyn Dispatcher> = if settings.num_threads == 0 {
            Box::new(SingleThreadedDispatcher::new(width, height, settings.level))
        } else {
            Box::new(MultiThreadedDispatcher::new(
                width,
                height,
                settings.num_threads,
                settings.level,
            ))
        };

        #[cfg(not(feature = "multithreading"))]
        let dispatcher: Box<dyn Dispatcher> =
            { Box::new(SingleThreadedDispatcher::new(width, height, settings.level)) };

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
        let aliasing_threshold = None;

        Self {
            width,
            height,
            dispatcher,
            transform,
            aliasing_threshold,
            blend_mode: BlendMode::default(),
            paint,
            render_settings: settings,
            mask: None,
            paint_transform,
            fill_rule,
            stroke,
            temp_path,
            encoded_paints,
            #[cfg(feature = "text")]
            glyph_caches: Some(Default::default()),
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
        self.dispatcher.fill_path(
            path,
            self.fill_rule,
            self.transform,
            paint,
            self.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        let paint = self.encode_current_paint();
        self.dispatcher.stroke_path(
            path,
            &self.stroke,
            self.transform,
            paint,
            self.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.rect_to_temp_path(rect);
        let paint = self.encode_current_paint();
        self.dispatcher.fill_path(
            &self.temp_path,
            self.fill_rule,
            self.transform,
            paint,
            self.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    fn rect_to_temp_path(&mut self, rect: &Rect) {
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

        self.rect_to_temp_path(&inflated_rect);

        let paint = blurred_rect.encode_into(&mut self.encoded_paints, transform);
        self.dispatcher.fill_path(
            &self.temp_path,
            Fill::NonZero,
            self.transform,
            paint,
            self.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.rect_to_temp_path(rect);
        let paint = self.encode_current_paint();
        self.dispatcher.stroke_path(
            &self.temp_path,
            &self.stroke,
            self.transform,
            paint,
            self.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run(&mut self, font: &crate::peniko::FontData) -> GlyphRunBuilder<'_, Self> {
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

        let blend_mode = blend_mode.unwrap_or_default();
        let opacity = opacity.unwrap_or(1.0);

        self.dispatcher.push_layer(
            clip_path,
            self.fill_rule,
            self.transform,
            blend_mode,
            opacity,
            self.aliasing_threshold,
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

    /// Set the aliasing threshold.
    ///
    /// If set to `None` (which is the recommended option in nearly all cases),
    /// anti-aliasing will be applied.
    ///
    /// If instead set to some value, then a pixel will be fully painted if
    /// the coverage is bigger than the threshold (between 0 and 255), otherwise
    /// it will not be painted at all.
    ///
    /// Note that there is no performance benefit to disabling anti-aliasing and
    /// this functionality is simply provided for compatibility.
    pub fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.aliasing_threshold = aliasing_threshold;
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

    /// Get the current stroke
    pub fn stroke(&self) -> &Stroke {
        &self.stroke
    }

    /// Set the current paint.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
    }

    /// Get the current paint.
    pub fn paint(&self) -> &PaintType {
        &self.paint
    }

    /// Set the blend mode that should be used when drawing objects.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Get the currently active blend mode.
    pub fn blend_mode(&self) -> BlendMode {
        self.blend_mode
    }

    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
    }

    /// Get the current paint transform.
    pub fn paint_transform(&self) -> &Affine {
        &self.paint_transform
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.paint_transform = Affine::IDENTITY;
    }

    /// Set the current fill rule.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    // TODO: Add explanation on how this differs to layer masks.
    /// Set the mask to use for path-painting operations.
    pub fn set_mask(&mut self, mask: Option<Mask>) {
        self.mask = mask;
    }

    /// Get the current fill rule.
    pub fn fill_rule(&self) -> &Fill {
        &self.fill_rule
    }

    /// Set the current transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Get the current transform.
    pub fn transform(&self) -> &Affine {
        &self.transform
    }

    /// Reset the current transform.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Reset the render context.
    pub fn reset(&mut self) {
        self.dispatcher.reset();
        self.encoded_paints.clear();
        self.mask = None;
        self.reset_transform();
        self.reset_paint_transform();
        #[cfg(feature = "text")]
        self.glyph_caches.as_mut().unwrap().maintain();
        self.blend_mode = BlendMode::default();
    }

    // TODO: Explain how this is different to `push_clip_layer`.
    /// Push a new clip path to the clip stack.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        self.dispatcher.push_clip_path(
            path,
            self.fill_rule,
            self.transform,
            self.aliasing_threshold,
        );
    }

    /// Pop a clip path from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clip paths before finishing the rendering operation.
    pub fn pop_clip_path(&mut self) {
        self.dispatcher.pop_clip_path();
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
    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        let width = pixmap.width();
        let height = pixmap.height();
        self.render_to_buffer(
            pixmap.data_as_u8_slice_mut(),
            width,
            height,
            self.render_settings.render_mode,
        );
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Return the render settings used by the `RenderContext`.
    pub fn render_settings(&self) -> &RenderSettings {
        &self.render_settings
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
                    self.blend_mode,
                    self.aliasing_threshold,
                    self.mask.clone(),
                );
            }
            GlyphType::Bitmap(glyph) => {
                // We need to change the state of the render context
                // to render the bitmap, but don't want to pollute the context,
                // so simulate a `save` and `restore` operation.

                use vello_common::peniko::ImageSampler;
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
                    image: ImageSource::Pixmap(Arc::new(glyph.pixmap)),
                    sampler: ImageSampler {
                        x_extend: crate::peniko::Extend::Pad,
                        y_extend: crate::peniko::Extend::Pad,
                        quality,
                        alpha: 1.0,
                    },
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

                use vello_common::peniko::ImageSampler;
                let old_transform = self.transform;
                let old_paint = self.paint.clone();
                let context_color = match old_paint {
                    PaintType::Solid(s) => s,
                    _ => BLACK,
                };

                let area = glyph.area;

                let glyph_pixmap = {
                    let settings = RenderSettings {
                        level: self.render_settings.level,
                        render_mode: self.render_settings.render_mode,
                        num_threads: 0,
                    };

                    let mut ctx = Self::new_with(glyph.pix_width, glyph.pix_height, settings);
                    let mut pix = Pixmap::new(glyph.pix_width, glyph.pix_height);

                    let mut colr_painter = ColrPainter::new(glyph, context_color, &mut ctx);
                    colr_painter.paint();

                    // Technically not necessary since we always render single-threaded, but just
                    // to be safe.
                    ctx.flush();
                    ctx.render_to_pixmap(&mut pix);

                    pix
                };

                let image = vello_common::paint::Image {
                    image: ImageSource::Pixmap(Arc::new(glyph_pixmap)),
                    sampler: ImageSampler {
                        x_extend: crate::peniko::Extend::Pad,
                        y_extend: crate::peniko::Extend::Pad,
                        // Since the pixmap will already have the correct size, no need to
                        // use a different image quality here.
                        quality: crate::peniko::ImageQuality::Low,
                        alpha: 1.0,
                    },
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
                    self.blend_mode,
                    self.aliasing_threshold,
                    self.mask.clone(),
                );
            }
            GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
                // The definitions of COLR and bitmap glyphs can't meaningfully support being stroked.
                // (COLR's imaging model only has fills)
                self.fill_glyph(prepared_glyph);
            }
        }
    }

    fn take_glyph_caches(&mut self) -> vello_common::glyph::GlyphCaches {
        self.glyph_caches.take().unwrap()
    }

    fn restore_glyph_caches(&mut self, cache: vello_common::glyph::GlyphCaches) {
        self.glyph_caches = Some(cache);
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

impl Recordable for RenderContext {
    fn record<F>(&mut self, recording: &mut Recording, f: F)
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recorder = Recorder::new(
            recording,
            self.transform,
            #[cfg(feature = "text")]
            self.take_glyph_caches(),
        );
        f(&mut recorder);
        #[cfg(feature = "text")]
        {
            self.glyph_caches = Some(recorder.take_glyph_caches());
        }
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        let buffers = recording.take_cached_strips();
        let (strip_storage, strip_start_indices) =
            self.generate_strips_from_commands(recording.commands(), buffers);
        recording.set_cached_strips(strip_storage, strip_start_indices);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        let (cached_strips, cached_alphas) = recording.get_cached_strips();
        let adjusted_strips = self.prepare_cached_strips(cached_strips, cached_alphas);

        // Use pre-calculated strip start indices from when we generated the cache.
        let strip_start_indices = recording.get_strip_start_indices();
        let mut range_index = 0;

        // Replay commands in order, using cached strips for geometry.
        for command in recording.commands() {
            match command {
                RenderCommand::FillPath(_)
                | RenderCommand::StrokePath(_)
                | RenderCommand::FillRect(_)
                | RenderCommand::StrokeRect(_) => {
                    self.process_geometry_command(
                        strip_start_indices,
                        range_index,
                        &adjusted_strips,
                    );
                    range_index += 1;
                }
                #[cfg(feature = "text")]
                RenderCommand::FillOutlineGlyph(_) | RenderCommand::StrokeOutlineGlyph(_) => {
                    self.process_geometry_command(
                        strip_start_indices,
                        range_index,
                        &adjusted_strips,
                    );
                    range_index += 1;
                }
                RenderCommand::SetPaint(paint) => {
                    self.set_paint(paint.clone());
                }
                RenderCommand::SetPaintTransform(transform) => {
                    self.set_paint_transform(*transform);
                }
                RenderCommand::ResetPaintTransform => {
                    self.reset_paint_transform();
                }
                RenderCommand::SetTransform(transform) => {
                    self.set_transform(*transform);
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.set_fill_rule(*fill_rule);
                }
                RenderCommand::SetStroke(stroke) => {
                    self.set_stroke(stroke.clone());
                }
                RenderCommand::PushLayer(PushLayerCommand {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                }) => {
                    self.push_layer(clip_path.as_ref(), *blend_mode, *opacity, mask.clone());
                }
                RenderCommand::PopLayer => {
                    self.pop_layer();
                }
            }
        }
    }
}

/// Saved state for recording operations.
#[derive(Debug)]
struct RenderState {
    transform: Affine,
    fill_rule: Fill,
    stroke: Stroke,
    paint: PaintType,
    paint_transform: Affine,
}

/// Recording management implementation.
impl RenderContext {
    /// Generate strips from strip commands and capture ranges.
    ///
    /// Returns:
    /// - `collected_strips`: The generated strips.
    /// - `collected_alphas`: The generated alphas.
    /// - `strip_start_indices`: The start indices of strips for each geometry command.
    fn generate_strips_from_commands(
        &mut self,
        commands: &[RenderCommand],
        buffers: (StripStorage, Vec<usize>),
    ) -> (StripStorage, Vec<usize>) {
        let (mut strip_storage, mut strip_start_indices) = buffers;
        strip_storage.clear();
        strip_storage.set_generation_mode(GenerationMode::Append);
        strip_start_indices.clear();

        let saved_state = self.take_current_state();
        let mut strip_generator =
            StripGenerator::new(self.width, self.height, self.render_settings.level);

        for command in commands {
            let start_index = strip_storage.strips.len();

            match command {
                RenderCommand::FillPath(path) => {
                    strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokePath(path) => {
                    strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillRect(rect) => {
                    self.rect_to_temp_path(rect);
                    strip_generator.generate_filled_path(
                        &self.temp_path,
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeRect(rect) => {
                    self.rect_to_temp_path(rect);
                    strip_generator.generate_stroked_path(
                        &self.temp_path,
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                #[cfg(feature = "text")]
                RenderCommand::FillOutlineGlyph((path, glyph_transform)) => {
                    strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                #[cfg(feature = "text")]
                RenderCommand::StrokeOutlineGlyph((path, glyph_transform)) => {
                    strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::SetTransform(transform) => {
                    self.transform = *transform;
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.fill_rule = *fill_rule;
                }
                RenderCommand::SetStroke(stroke) => {
                    self.stroke = stroke.clone();
                }

                _ => {}
            }
        }

        self.restore_state(saved_state);

        (strip_storage, strip_start_indices)
    }
}

/// Recording management implementation.
impl RenderContext {
    fn process_geometry_command(
        &mut self,
        strip_start_indices: &[usize],
        range_index: usize,
        adjusted_strips: &[Strip],
    ) {
        assert!(
            range_index < strip_start_indices.len(),
            "Strip range index out of bounds"
        );
        let start = strip_start_indices[range_index];
        let end = strip_start_indices
            .get(range_index + 1)
            .copied()
            .unwrap_or(adjusted_strips.len());
        let count = end - start;
        if count == 0 {
            // There are no strips to generate.
            return;
        }
        assert!(
            start < adjusted_strips.len() && count > 0,
            "Invalid strip range"
        );
        let paint = self.encode_current_paint();
        self.dispatcher
            .generate_wide_cmd(&adjusted_strips[start..end], paint, self.blend_mode);
    }

    /// Prepare cached strips for rendering by adjusting indices.
    fn prepare_cached_strips(
        &mut self,
        cached_strips: &[Strip],
        cached_alphas: &[u8],
    ) -> Vec<Strip> {
        // Calculate offset for alpha indices based on current dispatcher's alpha buffer size.
        let alpha_offset = {
            let storage = self.dispatcher.strip_storage_mut();
            let offset = storage.alphas.len() as u32;
            // Extend the dispatcher's alpha buffer with cached alphas.
            storage.alphas.extend(cached_alphas);

            offset
        };
        // Create adjusted strips with corrected alpha indices.
        cached_strips
            .iter()
            .map(move |strip| {
                let mut adjusted_strip = *strip;
                adjusted_strip.set_alpha_idx(adjusted_strip.alpha_idx() + alpha_offset);
                adjusted_strip
            })
            .collect()
    }

    /// Save the current rendering state.
    fn take_current_state(&mut self) -> RenderState {
        RenderState {
            paint: self.paint.clone(),
            paint_transform: self.paint_transform,
            transform: self.transform,
            fill_rule: self.fill_rule,
            stroke: core::mem::take(&mut self.stroke),
        }
    }

    /// Restore the saved rendering state.
    fn restore_state(&mut self, state: RenderState) {
        self.transform = state.transform;
        self.fill_rule = state.fill_rule;
        self.stroke = state.stroke;
        self.paint = state.paint;
        self.paint_transform = state.paint_transform;
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
        ctx.flush();
    }

    #[cfg(feature = "multithreading")]
    #[test]
    fn multithreaded_crash_after_reset() {
        use crate::{Level, RenderMode, RenderSettings};
        use vello_common::pixmap::Pixmap;

        let mut pixmap = Pixmap::new(200, 200);
        let settings = RenderSettings {
            level: Level::try_detect().unwrap_or(Level::fallback()),
            num_threads: 1,
            render_mode: RenderMode::OptimizeQuality,
        };

        let mut ctx = RenderContext::new_with(200, 200, settings);
        ctx.reset();
        ctx.fill_path(&Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1));
        ctx.flush();
        ctx.render_to_pixmap(&mut pixmap);
        ctx.flush();
        ctx.render_to_pixmap(&mut pixmap);
    }
}
