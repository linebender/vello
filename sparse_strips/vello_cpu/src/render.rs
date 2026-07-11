// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::RenderMode;
use crate::dispatch::Dispatcher;
#[cfg(feature = "multithreading")]
use crate::dispatch::multi_threaded::MultiThreadedDispatcher;
#[cfg(feature = "text")]
use crate::text::{GlyphAtlasResources, GlyphRunBuilder};
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;

use crate::dispatch::single_threaded::SingleThreadedDispatcher;
use crate::kurbo::{PathEl, Point};
use crate::record::FilterData;
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
use vello_common::pixmap::{Pixmap, PixmapMut};
use vello_common::render_state::RenderState;
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

    pub(crate) fn before_render(&mut self, render_mode: RenderMode) {
        #[cfg(feature = "text")]
        self.prepare_glyph_cache(render_mode);

        #[cfg(not(feature = "text"))]
        let _ = render_mode;
    }

    pub(crate) fn after_render(&mut self) {
        #[cfg(feature = "text")]
        self.maintain_glyph_cache();
    }
}

/// The composition mode that should be used when rendering into a pixmap.
///
/// For performance reason it is _highly_ recommended that you use `CompositeMode::Replace`, even
/// if you know that the pixmap is already cleared. Only use `SrcOver` if you really have to
/// preserve existing contents.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CompositeMode {
    /// Clear the destination pixmap and render the scene into it.
    #[default]
    Replace,
    /// Render the scene into the pixmap using src-over compositing.
    SrcOver,
}

/// The pixel format to assume for the destination pixmap.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PixelFormat {
    /// Premultiplied RGBA8.
    #[default]
    Rgba8,
}

/// Settings used when rasterizing a scene into a pixmap.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RasterizerSettings {
    /// Whether to prioritize speed or quality when rendering.
    ///
    /// For most cases (especially for real-time rendering), it is highly recommended to set
    /// this to [`RenderMode::OptimizeSpeed`]. If color accuracy is a more significant concern,
    /// then you can set this to [`RenderMode::OptimizeQuality`].
    ///
    /// Currently, the only difference this makes is that when choosing [`RenderMode::OptimizeSpeed`],
    /// rasterization will happen using u8/u16,
    /// while [`RenderMode::OptimizeQuality`] will use a f32-based pipeline.
    pub render_mode: RenderMode,
    /// How rendered content is composited into the destination.
    pub composite_mode: CompositeMode,
    /// Pixel format of the destination.
    pub pixel_format: PixelFormat,
    /// Offset in destination pixels where the render context origin is placed.
    ///
    /// See [`RenderContext::render_with`] for more information.
    pub offset: (u16, u16),
}

impl Default for RasterizerSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::OptimizeSpeed,
            composite_mode: CompositeMode::Replace,
            pixel_format: PixelFormat::Rgba8,
            offset: (0, 0),
        }
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
    /// Stack of root transforms.
    root_transforms: Vec<Affine>,
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
            root_transforms: vec![Affine::IDENTITY],
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
                let transform = self.effective_paint_transform();
                // TODO: Add caching?
                g.encode_into(&mut self.encoded_paints, transform, None)
            }
            PaintType::Image(i) => {
                let transform = self.effective_paint_transform();
                i.encode_into(&mut self.encoded_paints, transform, self.state.tint)
            }
        }
    }

    fn root_transform(&self) -> Affine {
        *self
            .root_transforms
            .last()
            .expect("root transform stack should never be empty")
    }

    fn effective_path_transform(&self) -> Affine {
        self.root_transform() * self.state.transform
    }

    // Unlike `effective_path_transform`, we are not applying the root transform here
    // because clipping handles this separately. See the `clip` module for more information.
    fn clip_path_transform(&self) -> Affine {
        self.state.transform
    }

    fn effective_paint_transform(&self) -> Affine {
        self.effective_path_transform() * self.state.paint_transform
    }

    pub(crate) fn push_root_transform(&mut self, relative_transform: Affine) {
        self.root_transforms
            .push(relative_transform * self.root_transform());
    }

    pub(crate) fn pop_root_transform(&mut self) {
        self.root_transforms.pop();
    }

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            let transform = ctx.effective_path_transform();
            ctx.dispatcher.fill_path(
                path,
                ctx.state.fill_rule,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
            );
        });
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            let transform = ctx.effective_path_transform();
            ctx.dispatcher.stroke_path(
                path,
                &ctx.state.stroke,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
            );
        });
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            let transform = ctx.effective_path_transform();

            // Fast path: Use optimized rect filling if we have no skew in the path transform
            // and anti-aliasing is enabled.
            // TODO: Maybe also support no anti-aliasing in the fast path
            if is_axis_aligned(&transform) && ctx.aliasing_threshold.is_none() {
                // Transform the rect to screen coordinates.
                let transformed_rect = transform.transform_rect_bbox(*rect);
                ctx.dispatcher.fill_rect_fast(
                    &transformed_rect,
                    paint,
                    ctx.state.blend_mode,
                    ctx.mask.clone(),
                );
            } else {
                // Fall back to path-based rendering for rotated/skewed transforms.
                ctx.rect_to_temp_path(rect);
                ctx.dispatcher.fill_path(
                    &ctx.temp_path,
                    ctx.state.fill_rule,
                    transform,
                    paint,
                    ctx.state.blend_mode,
                    ctx.aliasing_threshold,
                    ctx.mask.clone(),
                );
            }
        });
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.with_optional_filter(|ctx| {
            ctx.rect_to_temp_path(rect);
            let paint = ctx.encode_current_paint();
            let transform = ctx.effective_path_transform();
            ctx.dispatcher.stroke_path(
                &ctx.temp_path,
                &ctx.state.stroke,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
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
    /// When `invert` is `true`, the inverse (`1 - alpha`) of the blur coverage is painted: the
    /// paint is fully opaque outside the blurred rectangle and fades to transparent inside it. This
    /// can be used to implement inset box shadows.
    ///
    /// Note that this only works properly if the current paint is set to a solid color.
    /// If not, it will fall back to using black as the fill color.
    pub fn fill_blurred_rounded_rect(
        &mut self,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
        invert: bool,
    ) {
        let rect = rect.abs();
        let color = match self.state.paint {
            PaintType::Solid(s) => s,
            // Fallback to black when attempting to blur a rectangle with an image/gradient paint
            _ => BLACK,
        };

        let blurred_rect = BlurredRoundedRectangle {
            rect,
            color,
            radius,
            std_dev,
            invert,
        };

        // The actual rectangle we paint needs to be larger so that the blurring effect
        // is not cut off.
        // The impulse response of a gaussian filter is infinite.
        // For performance reason we cut off the filter at some extent where the response is close to zero.
        let kernel_size = 2.5 * std_dev;
        let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
        let transform = self.effective_path_transform();
        let paint_transform = self.effective_paint_transform();

        self.rect_to_temp_path(&inflated_rect);

        let paint = blurred_rect.encode_into(&mut self.encoded_paints, paint_transform, None);
        self.dispatcher.fill_path(
            &self.temp_path,
            Fill::NonZero,
            transform,
            paint,
            self.state.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
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
            self.state.paint_transform,
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
        let layer_transform = self.effective_path_transform();
        let filter_plan = filter.map(|filter| FilterData::new(filter, layer_transform));

        // The important part! Let's say we have an element placed in a way such that
        // its drop shadow starts at (0, 0). In order for it to render correctly, we would
        // have to render parts of the shape that at negative viewport coordinates, which is
        // not supported. Therefore, we instead shift everything down such that we can assume
        // everything left/above (0, 0) is not needed for correct rendering, and simply
        // shift everything back when actually compositing the rendered filter layer.
        self.push_root_transform(
            filter_plan
                .as_ref()
                .map_or(Affine::IDENTITY, |filter_plan| {
                    let (shift_x, shift_y) = filter_plan.source_shift();

                    Affine::translate((f64::from(shift_x), f64::from(shift_y)))
                }),
        );

        self.dispatcher.push_layer(
            clip_path,
            self.state.fill_rule,
            layer_transform,
            blend_mode,
            opacity,
            self.aliasing_threshold,
            mask,
            filter_plan,
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
        self.pop_root_transform();
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

    /// Reset the render context and update the scene size.
    pub fn reset_and_resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;

        self.reset();
    }

    /// Reset the render context.
    pub fn reset(&mut self) {
        self.dispatcher.reset(self.width, self.height);
        self.encoded_paints.clear();
        self.mask = None;
        self.root_transforms.clear();
        self.root_transforms.push(Affine::IDENTITY);
        self.state.reset();
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        let transform = self.clip_path_transform();
        self.dispatcher.push_clip_path(
            path,
            self.state.fill_rule,
            transform,
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

    /// Render the current context into a target using default rasterizer settings.
    ///
    /// See the documentation of [`RenderContext::render_with`] for more information.
    pub fn render<'a>(&self, target: impl Into<PixmapMut<'a>>, resources: &mut Resources) {
        self.render_with(target, resources, RasterizerSettings::default());
    }

    /// Render the current context into a target using custom rasterizer settings.
    ///
    /// See the documentation of [`RasterizerSettings`] to understand the tunable parameters for
    /// rasterization.
    ///
    /// There is an important note to make about render sizes. [`RenderContext`] can be configured with
    /// a specific width/height, but so can [`Pixmap`]. In the vast majority of cases, you will simply
    /// want to configure them both to have the same size. However, it _is_ very much possible for them
    /// to have different sizes, which can be useful in certain situations. In principle, the size
    /// that you specify when creating a [`RenderContext`] defines the bound of the scene itself. Any
    /// content that is to the top/left of (0, 0) and to the right/bottom of (width/height) will be
    /// removed. However, the offset in [`RasterizerSettings`] as well as the width/height of
    /// the [`PixmapMut`] define at which location the scene will be rasterized into, and allows
    /// for further clipping certain parts of the scene away. The semantics are defined as follows:
    ///
    /// 1. [`RasterizerSettings::offset`] defines the where the top-left corner will be positioned
    ///    on the pixmap, assuming a y-down coordinate system. In most cases (0, 0) will be the
    ///    appropriate choice, but other values are certainly sensible. For example, if you want to
    ///    implement a custom glyph-atlas, you can construct the scene assuming (0, 0) as the origin
    ///    and then position the glyphs at rasterization time using this feature.
    ///
    /// 2. In case the pixmap width/height is larger than the offset plus the width/height of the
    ///    [`RenderContext`], any remaining rows/columns are simply treated as padding (**however**,
    ///    when using [`CompositeMode::Replace`], then the _whole_ destination pixmap will
    ///    be cleared, not just the area covered by the scene). One potential reason for doing this
    ///    is that certain platforms, for example macOS, require a specific byte stride for buffers.
    ///    For example, let's say that a byte stride of 128 is imposed by the platform, but the actual
    ///    size of the scene you are drawing is only 20x20. In this case, you can create a pixmap
    ///    of size 32x20, and the last 12 columns are essentially treated as padding.
    ///
    /// 3. In case the width/height of the pixmap is _smaller_ than the offset + width/height of the
    ///    scene, then anything that exceeds the pixmap boundaries is simply cut off. This can be useful
    ///    if for some reason you only want to rasterize a small cut-out of the original scene.
    pub fn render_with<'a>(
        &self,
        target: impl Into<PixmapMut<'a>>,
        resources: &mut Resources,
        settings: RasterizerSettings,
    ) {
        // TODO: Maybe we should move those checks into the dispatcher.
        assert!(
            !self.dispatcher.has_layers(),
            "some layers haven't been popped yet"
        );

        resources.before_render(settings.render_mode);
        let mut target = target.into();
        let target_fully_covered = settings.offset == (0, 0)
            && self.width >= target.width()
            && self.height >= target.height();
        // If the scene covers the whole pixmap than packing will take care
        // of clearing everything anyway, so no reason to clear it explicitly
        // here.
        if settings.composite_mode == CompositeMode::Replace && !target_fully_covered {
            target.data_mut().fill(0);
        }

        self.dispatcher.rasterize(
            target,
            self.width,
            self.height,
            settings,
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

    /// Return the width of the scene.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the scene.
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

    /// Whether rendering is currently configured to run in multi-threaded mode.
    pub fn is_multi_threaded(&self) -> bool {
        self.dispatcher.is_multi_threaded()
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "text")]
    use crate::peniko::{Blob, FontData};
    use crate::{CompositeMode, RasterizerSettings, RenderContext, Resources};
    #[cfg(feature = "text")]
    use alloc::sync::Arc;
    use alloc::vec;
    #[cfg(feature = "text")]
    use glifo::Glyph;
    use vello_common::color::PremulRgba8;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::kurbo::{Rect, Shape};
    use vello_common::pixmap::{Pixmap, PixmapMut};
    use vello_common::tile::Tile;

    const GRAY: PremulRgba8 = PremulRgba8 {
        r: 9,
        g: 10,
        b: 11,
        a: 255,
    };

    fn red_pixel() -> PremulRgba8 {
        RED.premultiply().to_rgba8()
    }

    fn blue_pixel() -> PremulRgba8 {
        BLUE.premultiply().to_rgba8()
    }

    fn transparent_pixel() -> PremulRgba8 {
        PremulRgba8::from_u32(0)
    }

    fn solid_pixmap(width: u16, height: u16, color: PremulRgba8) -> Pixmap {
        Pixmap::from_parts(
            vec![color; usize::from(width) * usize::from(height)],
            width,
            height,
        )
    }

    fn red_rect_context(width: u16, height: u16, rect: Rect) -> RenderContext {
        let mut ctx = RenderContext::new(width, height);
        ctx.set_paint(RED);
        ctx.fill_rect(&rect);
        ctx.flush();
        ctx
    }

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

    #[test]
    fn render_with_offset_clears_pixels_outside_scene() {
        let ctx = red_rect_context(2, 2, Rect::new(0.0, 0.0, 2.0, 2.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 3, GRAY);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                offset: (1, 1),
                ..Default::default()
            },
        );

        for y in 0..3 {
            for x in 0..4 {
                let expected = if (1..=2).contains(&x) && (1..=2).contains(&y) {
                    red_pixel()
                } else {
                    transparent_pixel()
                };

                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn render_clips_scene_to_target_bounds() {
        let ctx = red_rect_context(3, 3, Rect::new(0.0, 0.0, 3.0, 3.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 4, GRAY);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                offset: (2, 1),
                ..Default::default()
            },
        );

        for y in 0..4 {
            for x in 0..4 {
                let expected = if (2..=3).contains(&x) && (1..=3).contains(&y) {
                    red_pixel()
                } else {
                    transparent_pixel()
                };
                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn render_into_padded_pixmap() {
        let ctx = red_rect_context(2, 2, Rect::new(0.0, 0.0, 2.0, 2.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 2, GRAY);

        ctx.render(&mut pixmap, &mut resources);

        for y in 0..2 {
            for x in 0..4 {
                let expected = if x < 2 {
                    red_pixel()
                } else {
                    transparent_pixel()
                };
                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn reset_and_resize_updates_scene_size() {
        let mut ctx = RenderContext::new(8, 4);
        let mut resources = Resources::new();
        let mut pixmap = Pixmap::new(4, 8);

        ctx.reset_and_resize(4, 8);
        assert_eq!(ctx.width(), 4);
        assert_eq!(ctx.height(), 8);

        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 8.0));
        ctx.flush();
        ctx.render(&mut pixmap, &mut resources);

        for y in 0..8 {
            for x in 0..4 {
                assert_eq!(pixmap.sample(x, y), blue_pixel(), "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn render_into_raw_buffer() {
        let ctx = red_rect_context(2, 1, Rect::new(0.0, 0.0, 2.0, 1.0));
        let mut resources = Resources::new();
        let mut buffer = vec![0; 3 * 2 * 4];
        for pixel in buffer.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[GRAY.r, GRAY.g, GRAY.b, GRAY.a]);
        }

        {
            let pixmap = PixmapMut::new(3, 2, &mut buffer).unwrap();
            ctx.render_with(
                pixmap,
                &mut resources,
                RasterizerSettings {
                    offset: (1, 1),
                    ..Default::default()
                },
            );
        }

        let expected = [
            transparent_pixel(),
            transparent_pixel(),
            transparent_pixel(),
            transparent_pixel(),
            red_pixel(),
            red_pixel(),
        ];
        for (pixel, expected) in buffer.chunks_exact(4).zip(expected) {
            assert_eq!(pixel, [expected.r, expected.g, expected.b, expected.a]);
        }
    }

    #[test]
    fn pixmap_mut_validates_buffer_length() {
        let mut short_buffer = vec![0; 3 * 2 * 4 - 1];
        assert!(PixmapMut::new(3, 2, &mut short_buffer).is_none());

        let mut exact_buffer = vec![0; 3 * 2 * 4];
        assert!(PixmapMut::new(3, 2, &mut exact_buffer).is_some());
    }

    #[test]
    fn render_src_over_opaque() {
        let ctx = red_rect_context(2, 1, Rect::new(0.0, 0.0, 1.0, 1.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(2, 1, blue_pixel());

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                composite_mode: CompositeMode::SrcOver,
                ..Default::default()
            },
        );

        assert_eq!(pixmap.sample(0, 0), red_pixel());
        assert_eq!(pixmap.sample(1, 0), blue_pixel());
    }

    #[test]
    fn render_src_over_transparent() {
        let mut ctx = RenderContext::new(1, 1);
        ctx.set_paint(RED.with_alpha(0.5));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
        ctx.flush();

        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(1, 1, blue_pixel());

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                composite_mode: CompositeMode::SrcOver,
                ..Default::default()
            },
        );

        assert_eq!(
            pixmap.sample(0, 0),
            PremulRgba8 {
                r: 128,
                g: 0,
                b: 127,
                a: 255,
            }
        );
    }

    #[cfg(feature = "multithreading")]
    #[test]
    fn multithreaded_crash_after_reset() {
        use crate::{Level, RasterizerSettings, RenderMode, RenderSettings};

        let mut pixmap = Pixmap::new(200, 200);
        let settings = RenderSettings {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            num_threads: 1,
        };
        let rasterizer_settings = RasterizerSettings {
            render_mode: RenderMode::OptimizeQuality,
            ..Default::default()
        };

        let mut resources = Resources::new();
        let mut ctx = RenderContext::new_with(200, 200, settings);
        ctx.reset();
        ctx.fill_path(&Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1));
        ctx.flush();
        ctx.render_with(&mut pixmap, &mut resources, rasterizer_settings);
        ctx.flush();
        ctx.render_with(&mut pixmap, &mut resources, rasterizer_settings);
    }

    #[cfg(feature = "multithreading")]
    #[test]
    fn multithreaded_reset_with_pending_tasks() {
        use crate::RenderSettings;

        let mut ctx = RenderContext::new_with(
            100,
            100,
            RenderSettings {
                num_threads: 4,
                ..Default::default()
            },
        );
        
        // Note: This test only works if we draw enough rectangles
        // to trigger a batch send.
        for _ in 0..300 {
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100., 100.0));
        }

        ctx.reset();
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

        let mut resources = Resources::new();
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
