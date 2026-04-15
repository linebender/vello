// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::RenderMode;
use crate::dispatch::Dispatcher;
#[cfg(feature = "text")]
use crate::text::{GlyphAtlasResources, GlyphRunBuilder};
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;

#[cfg(feature = "multithreading")]
use crate::dispatch::multi_threaded::MultiThreadedDispatcher;
use crate::dispatch::single_threaded::SingleThreadedDispatcher;
use crate::kurbo::{PathEl, Point};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{ImageId, ImageResolver, Paint, PaintType, Tint};
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Fill};
use vello_common::pixmap::Pixmap;
use vello_common::recording::{
    PushLayerCommand, Recordable, Recorder, Recording, RenderCommand, RenderState,
};
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::util::is_axis_aligned;

#[cfg(feature = "text")]
pub(crate) const DEFAULT_GLYPH_ATLAS_SIZE: u16 = 4096;
// Why do we need this? The reason is that the way uploaded images work in Vello Hybrid
// is different from how they work in Vello CPU.
//
// In Vello Hybrid, all images, regardless of whether they are user-uploaded
// images or cached glyphs, are stored in an image atlas at a certain location. An image ID then
// uniquely resolves to an atlas page index + a location on that page. Whenever we want to
// cache a new glyph, we simply allocate a location in the image atlas and then return the image
// ID associated with that location.
//
// On Vello CPU, it works differently: An image ID is associated with a complete pixmap.
// If a user uploads an image, instead of blitting it into a bigger image atlas, we just
// store the user-provided pixmap and associate an image ID with the whole pixmap. However,
// for glyph caching to work we need the same semantics as in Vello Hybrid. Therefore, we
// use a marker to determine whether an image ID refers to a normal uploaded image or a cached
// glyph and apply special handling based on that.
//
// All IDs < than this value are reserved for normal images, all IDs >= this value are
// reserved for atlas pages.
pub(crate) const ATLAS_IMAGE_ID_BASE: u32 = u32::MAX / 2;

/// Persistent resources required by Vello CPU for rendering.
#[derive(Debug, Default)]
pub struct Resources {
    pub(crate) image_registry: ImageRegistry,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    // Will be initialized lazily on first use.
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    /// Create a new set of renderer resources.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn before_render(&mut self) {
        #[cfg(feature = "text")]
        self.prepare_glyph_cache();
    }

    pub(crate) fn after_render(&mut self) {
        #[cfg(feature = "text")]
        self.maintain_glyph_cache();
    }
}

/// A render context for CPU-based 2D graphics rendering.
///
/// This is the main entry point for drawing operations. It maintains the current
/// rendering state (transforms, paint, stroke, etc.) and dispatches drawing commands
/// to the underlying rasterization engine.
#[derive(Debug)]
pub struct RenderContext {
    /// Width of the render target in pixels.
    pub(crate) width: u16,
    /// Height of the render target in pixels.
    pub(crate) height: u16,
    /// The current rendering state.
    pub(crate) state: RenderState,
    /// The current mask in place.
    pub(crate) mask: Option<Mask>,
    /// Temporary path buffer to avoid repeated allocations.
    pub(crate) temp_path: BezPath,
    /// Optional threshold for aliasing.
    pub(crate) aliasing_threshold: Option<u8>,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    pub(crate) filter: Option<Filter>,
    #[cfg_attr(
        not(feature = "text"),
        allow(dead_code, reason = "used when the `text` feature is enabled")
    )]
    pub(crate) render_settings: RenderSettings,
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
            level: Level::try_detect().unwrap_or(Level::baseline()),
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

        let encoded_paints = vec![];
        let temp_path = BezPath::new();
        let aliasing_threshold = None;

        Self {
            width,
            height,
            dispatcher,
            state: RenderState::default(),
            aliasing_threshold,
            render_settings: settings,
            mask: None,
            temp_path,
            encoded_paints,
            filter: None,
        }
    }

    fn encode_current_paint(&mut self) -> Paint {
        match self.state.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => {
                // TODO: Add caching?
                g.encode_into(
                    &mut self.encoded_paints,
                    self.state.transform * self.state.paint_transform,
                    None,
                )
            }
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints,
                self.state.transform * self.state.paint_transform,
                self.state.tint,
            ),
        }
    }

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            ctx.dispatcher.fill_path(
                path,
                ctx.state.fill_rule,
                ctx.state.transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
                &ctx.encoded_paints,
            );
        });
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            ctx.dispatcher.stroke_path(
                path,
                &ctx.state.stroke,
                ctx.state.transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
                &ctx.encoded_paints,
            );
        });
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();

            // Fast path: Use optimized rect filling if we have no skew in the path transform
            // and anti-aliasing is enabled.
            // TODO: Maybe also support no anti-aliasing in the fast path
            if is_axis_aligned(&ctx.state.transform) && ctx.aliasing_threshold.is_none() {
                // Transform the rect to screen coordinates.
                let transformed_rect = ctx.state.transform.transform_rect_bbox(*rect);
                ctx.dispatcher.fill_rect_fast(
                    &transformed_rect,
                    paint,
                    ctx.state.blend_mode,
                    ctx.mask.clone(),
                    &ctx.encoded_paints,
                );
            } else {
                // Fall back to path-based rendering for rotated/skewed transforms.
                ctx.rect_to_temp_path(rect);
                ctx.dispatcher.fill_path(
                    &ctx.temp_path,
                    ctx.state.fill_rule,
                    ctx.state.transform,
                    paint,
                    ctx.state.blend_mode,
                    ctx.aliasing_threshold,
                    ctx.mask.clone(),
                    &ctx.encoded_paints,
                );
            }
        });
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.with_optional_filter(|ctx| {
            ctx.rect_to_temp_path(rect);
            let paint = ctx.encode_current_paint();
            ctx.dispatcher.stroke_path(
                &ctx.temp_path,
                &ctx.state.stroke,
                ctx.state.transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
                &ctx.encoded_paints,
            );
        });
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

    /// Fill a blurred rectangle with the given corner radius and standard deviation.
    ///
    /// Note that this only works properly if the current paint is set to a solid color.
    /// If not, it will fall back to using black as the fill color.
    pub fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        let color = match self.state.paint {
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
        let transform = self.state.transform * self.state.paint_transform;

        self.rect_to_temp_path(&inflated_rect);

        let paint = blurred_rect.encode_into(&mut self.encoded_paints, transform, None);
        self.dispatcher.fill_path(
            &self.temp_path,
            Fill::NonZero,
            self.state.transform,
            paint,
            self.state.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
            &self.encoded_paints,
        );
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run<'a>(
        &'a mut self,
        resources: &'a mut Resources,
        font: &crate::peniko::FontData,
    ) -> GlyphRunBuilder<'a> {
        glifo::GlyphRunBuilder::new(
            font.clone(),
            self.state.transform,
            crate::text::CpuGlyphRunBackend {
                ctx: self,
                resources,
                atlas_cache_enabled: false,
            },
        )
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
        filter: Option<Filter>,
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
            self.state.fill_rule,
            self.state.transform,
            blend_mode,
            opacity,
            self.aliasing_threshold,
            mask,
            filter,
        );
    }

    /// Push a new clip layer.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_path`.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None, None);
    }

    /// Push a new blend layer.
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }

    /// Push a new opacity layer.
    pub fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(None, None, Some(opacity), None, None);
    }

    /// Push a new mask layer. The mask needs to have the same dimensions as the
    /// render context. The mask will not be affected by the current transform
    /// in place.
    ///
    /// See the explanation in the [masking](https://github.com/linebender/vello/tree/main/sparse_strips/masking/examples)
    /// example for how this method differs from `set_mask`.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a filter layer that affects all subsequent drawing operations.
    ///
    /// WARNING: Note that filters are currently incomplete and experimental. In
    /// particular, they will lead to a panic when used in combination with
    /// multi-threaded rendering.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
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

    /// Pop the last-pushed layer.
    pub fn pop_layer(&mut self) {
        self.dispatcher.pop_layer();
    }

    /// Set the current stroke.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.state.stroke = stroke;
    }

    /// Get the current stroke.
    pub fn stroke(&self) -> &Stroke {
        &self.state.stroke
    }

    /// Get a mutable reference to the current stroke.
    #[cfg(feature = "text")]
    pub(crate) fn stroke_mut(&mut self) -> &mut Stroke {
        &mut self.state.stroke
    }

    /// Set the current paint.
    ///
    /// If the paint is an image with `ImageSource::OpaqueId`, it will be
    /// resolved to the corresponding pixmap at rasterization time.
    /// Make sure to register images with [`Resources::register_image`] first.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.state.paint = paint.into();
    }

    /// Get the current paint.
    pub fn paint(&self) -> &PaintType {
        &self.state.paint
    }

    /// Set the tint for subsequent image paint operations.
    pub fn set_tint(&mut self, tint: Option<Tint>) {
        self.state.tint = tint;
    }

    /// Clear the tint, so subsequent image paints are drawn without tinting.
    pub fn reset_tint(&mut self) {
        self.state.tint = None;
    }

    /// Set the blend mode that should be used when drawing objects.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.state.blend_mode = blend_mode;
    }

    /// Get the currently active blend mode.
    pub fn blend_mode(&self) -> BlendMode {
        self.state.blend_mode
    }

    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.state.paint_transform = paint_transform;
    }

    /// Get the current paint transform.
    pub fn paint_transform(&self) -> &Affine {
        &self.state.paint_transform
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.state.paint_transform = Affine::IDENTITY;
    }

    /// Set the current fill rule.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.state.fill_rule = fill_rule;
    }

    /// Set the mask to use for path-painting operations. The mask needs to
    /// have the same dimensions as the render context. The mask will not be
    /// affected by the current transform in place.
    ///
    /// See the explanation in the [masking](https://github.com/linebender/vello/tree/main/sparse_strips/masking/examples)
    /// example for how this method differs from `push_mask_layer`.
    pub fn set_mask(&mut self, mask: Mask) {
        self.mask = Some(mask);
    }

    /// Reset the mask that is used for path-painting operations.
    pub fn reset_mask(&mut self) {
        self.mask = None;
    }

    /// Get the current fill rule.
    pub fn fill_rule(&self) -> &Fill {
        &self.state.fill_rule
    }

    /// Set the current transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.state.transform = transform;
    }

    /// Get the current transform.
    pub fn transform(&self) -> &Affine {
        &self.state.transform
    }

    /// Reset the current transform.
    pub fn reset_transform(&mut self) {
        self.state.transform = Affine::IDENTITY;
    }

    /// Apply filter to the current paint (affects next drawn elements).
    ///
    /// This sets a filter that will be applied to the next drawn element.
    /// To apply a filter to multiple elements, use `push_filter_layer` instead.
    pub fn set_filter_effect(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        self.filter = None;
    }

    /// Reset the render context.
    pub fn reset(&mut self) {
        self.dispatcher.reset();
        self.encoded_paints.clear();
        self.mask = None;
        self.state.reset();
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        self.dispatcher.push_clip_path(
            path,
            self.state.fill_rule,
            self.state.transform,
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
        self.dispatcher.flush(&self.encoded_paints);
    }

    /// Render the current context into a buffer.
    /// The buffer is expected to be in premultiplied RGBA8 format with length `width * height * 4`
    pub fn render_to_buffer(
        &self,
        resources: &mut Resources,
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

        resources.before_render();

        self.dispatcher.rasterize(
            buffer,
            render_mode,
            width,
            height,
            &self.encoded_paints,
            &resources.image_registry,
        );
        // TODO: We need to figure something out here API-wise. At the moment, the user can
        // theoretically rasterize the same `RenderContext` multiple times without resetting in-between.
        // However, if glyph caching is enabled, this method call could now evict that were previously
        // assumed to exist in `RenderContext`, meaning that if the user rasterizes the same `RenderContext`
        // again without resetting it, some of the cached glyphs might be stale and not exist anymore.
        resources.after_render();
    }

    /// Render the current context into a pixmap.
    pub fn render_to_pixmap(&self, resources: &mut Resources, pixmap: &mut Pixmap) {
        let width = pixmap.width();
        let height = pixmap.height();
        self.render_to_buffer(
            resources,
            pixmap.data_as_u8_slice_mut(),
            width,
            height,
            self.render_settings.render_mode,
        );
    }

    /// Composite the current context into a region of a pixmap.
    ///
    /// The context's content (sized `self.width × self.height`) is composited
    /// directly to the destination pixmap starting at `(dst_x, dst_y)`.
    /// If the region extends beyond the pixmap bounds, it is clipped.
    ///
    /// Unlike [`render_to_pixmap`](Self::render_to_pixmap), this method composites on top of
    /// existing pixmap content rather than clearing it first, allowing multiple
    /// renders to accumulate.
    ///
    /// This is useful for rendering individual elements (like glyphs) into
    /// a spritesheet at specific coordinates.
    ///
    /// # Panics
    ///
    /// This method is only supported with the single-threaded dispatcher and will
    /// **panic** if called on a `RenderContext` using the multi-threaded dispatcher.
    pub fn composite_to_pixmap_at_offset(
        &self,
        resources: &Resources,
        pixmap: &mut Pixmap,
        dst_x: u16,
        dst_y: u16,
    ) {
        let dst_buffer_width = pixmap.width();
        let dst_buffer_height = pixmap.height();
        self.dispatcher.composite_at_offset(
            pixmap.data_as_u8_slice_mut(),
            self.width,
            self.height,
            dst_x,
            dst_y,
            dst_buffer_width,
            dst_buffer_height,
            self.render_settings.render_mode,
            &self.encoded_paints,
            &resources.image_registry,
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

    /// Execute a drawing operation, optionally wrapping it in a filter layer.
    fn with_optional_filter<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self),
    {
        if let Some(filter) = self.filter.clone() {
            self.push_filter_layer(filter);
            f(self);
            self.pop_layer();
        } else {
            f(self);
        }
    }

    /// Take current rendering state and reset the existing state to its default.
    pub fn take_current_state(&mut self) -> RenderState {
        core::mem::take(&mut self.state)
    }

    /// Save a copy of the current rendering state.
    pub fn save_current_state(&mut self) -> RenderState {
        self.state.clone()
    }

    /// Restore rendering state.
    pub fn restore_state(&mut self, state: RenderState) {
        self.state = state;
    }
}

/// Image registry implementation.
impl Resources {
    /// Register a pixmap in the image registry and return its [`ImageId`].
    pub fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        self.image_registry.register(pixmap)
    }

    /// Remove an image from the registry.
    pub fn destroy_image(&mut self, id: ImageId) -> bool {
        self.image_registry.destroy(id)
    }

    /// Resolve an `ImageId` to its pixmap data.
    pub fn resolve_image(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.image_registry.resolve(id)
    }

    /// Clear the image registry.
    pub fn clear_images(&mut self) {
        self.image_registry.clear();
    }
}

impl Recordable for RenderContext {
    fn record<F>(&mut self, recording: &mut Recording, f: F)
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recorder = Recorder::new(recording, self.state.transform);
        f(&mut recorder);
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
                RenderCommand::SetTint(tint) => {
                    self.set_tint(*tint);
                }
                RenderCommand::SetFilterEffect(filter) => {
                    self.set_filter_effect(filter.clone());
                }
                RenderCommand::ResetFilterEffect => {
                    self.reset_filter_effect();
                }
                RenderCommand::PushLayer(PushLayerCommand {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    filter,
                }) => {
                    self.push_layer(
                        clip_path.as_ref(),
                        *blend_mode,
                        *opacity,
                        mask.clone(),
                        filter.clone(),
                    );
                }
                RenderCommand::PopLayer => {
                    self.pop_layer();
                }
            }
        }
    }
}

/// Registry that maps opaque [`ImageId`]s to [`Pixmap`] data.
///
/// Used by [`RenderContext`] to resolve `ImageSource::OpaqueId` at rasterization time.
#[derive(Debug, Default)]
pub(crate) struct ImageRegistry {
    images: HashMap<u32, Arc<Pixmap>>,
    next_id: u32,
}

impl ImageRegistry {
    fn register(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        let id = self.next_id;
        assert!(
            id < ATLAS_IMAGE_ID_BASE,
            "image registry exhausted non-atlas image IDs"
        );

        self.next_id += 1;
        self.images.insert(id, pixmap);
        ImageId::new(id)
    }

    #[cfg(feature = "text")]
    pub(crate) fn register_atlas_page(&mut self, page_index: u32, pixmap: Arc<Pixmap>) {
        self.images.insert(
            ImageId::new(ATLAS_IMAGE_ID_BASE + page_index).as_u32(),
            pixmap,
        );
    }

    pub(crate) fn destroy(&mut self, id: ImageId) -> bool {
        self.images.remove(&id.as_u32()).is_some()
    }

    #[cfg(feature = "text")]
    pub(crate) fn destroy_atlas_page(&mut self, page_index: u32) -> bool {
        self.destroy(ImageId::new(ATLAS_IMAGE_ID_BASE + page_index))
    }

    fn resolve(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.images.get(&id.as_u32()).cloned()
    }

    fn clear(&mut self) {
        self.images.clear();
        self.next_id = 0;
    }
}

impl ImageResolver for ImageRegistry {
    fn resolve(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.images.get(&id.as_u32()).cloned()
    }
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
                        self.state.fill_rule,
                        self.state.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        self.dispatcher.current_clip_path(),
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokePath(path) => {
                    strip_generator.generate_stroked_path(
                        path,
                        &self.state.stroke,
                        self.state.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        self.dispatcher.current_clip_path(),
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillRect(rect) => {
                    self.rect_to_temp_path(rect);
                    strip_generator.generate_filled_path(
                        &self.temp_path,
                        self.state.fill_rule,
                        self.state.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        self.dispatcher.current_clip_path(),
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeRect(rect) => {
                    self.rect_to_temp_path(rect);
                    strip_generator.generate_stroked_path(
                        &self.temp_path,
                        &self.state.stroke,
                        self.state.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        self.dispatcher.current_clip_path(),
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::SetTransform(transform) => {
                    self.state.transform = *transform;
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.state.fill_rule = *fill_rule;
                }
                RenderCommand::SetStroke(stroke) => {
                    self.state.stroke = stroke.clone();
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
        self.dispatcher.generate_wide_cmd(
            &adjusted_strips[start..end],
            paint,
            self.state.blend_mode,
            &self.encoded_paints,
        );
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
}

#[cfg(test)]
mod tests {
    use crate::RenderContext;
    #[cfg(feature = "text")]
    use crate::peniko::{Blob, FontData};
    #[cfg(feature = "text")]
    use alloc::sync::Arc;
    #[cfg(feature = "text")]
    use glifo::Glyph;
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
            level: Level::try_detect().unwrap_or(Level::baseline()),
            num_threads: 1,
            render_mode: RenderMode::OptimizeQuality,
        };

        let mut resources = crate::Resources::new();
        let mut ctx = RenderContext::new_with(200, 200, settings);
        ctx.reset();
        ctx.fill_path(&Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1));
        ctx.flush();
        ctx.render_to_pixmap(&mut resources, &mut pixmap);
        ctx.flush();
        ctx.render_to_pixmap(&mut resources, &mut pixmap);
    }

    #[cfg(feature = "text")]
    #[test]
    fn glyph_atlas_resources_are_lazy() {
        const ROBOTO_FONT: &[u8] =
            include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

        let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);
        let glyphs = [Glyph {
            id: 1,
            x: 0.0,
            y: 0.0,
        }];

        let mut resources = crate::Resources::new();
        let mut ctx = RenderContext::new(100, 100);

        ctx.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        ctx.fill_path(&Rect::new(10.0, 10.0, 20.0, 20.0).to_path(0.1));
        ctx.glyph_run(&mut resources, &font)
            .fill_glyphs(glyphs.into_iter());

        assert!(resources.glyph_resources.is_none());

        ctx.glyph_run(&mut resources, &font)
            .atlas_cache(true)
            .fill_glyphs(glyphs.into_iter());

        assert!(resources.glyph_resources.is_some());
    }
}
