// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

#[cfg(feature = "text")]
use crate::Resources;
use crate::sampling::SampleRect;
#[cfg(feature = "text")]
use crate::text::GlyphRunBuilder;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::Range;
use vello_common::TextureId;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::clip::PathDataRef;
use vello_common::encode::{EncodeExt, EncodedExternalTexture, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter::FilterData;
use vello_common::filter_effects::Filter;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Paint, PaintType, Tint};
#[cfg(feature = "text")]
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Extend, Fill, ImageQuality, ImageSampler};
use vello_common::record::{CommandRecorder, Drawable, LayerClip, LayerProps, PoppedLayer};
use vello_common::render_state::RenderState;
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::transforms::{RootTransforms, Transforms};
use vello_common::util::{RectExt, is_axis_aligned, strip_bbox};
use vello_common::viewport::ViewportState;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// Path or rectangle draw retained by the scene recorder.
#[derive(Debug)]
pub(crate) enum RecordedDraw {
    /// Path represented by generated strips.
    Path(RecordedPath),
    /// Rectangle retained for the direct GPU rectangle path.
    Rect(RecordedRect),
}

/// Recorded path strips and their paint.
#[derive(Debug)]
pub(crate) struct RecordedPath {
    /// Range selecting the path's strips from scene strip storage.
    pub(crate) strips: Range<usize>,
    /// Paint applied to the path.
    pub(crate) paint: Paint,
}

/// Recorded rectangle and its paint.
#[derive(Debug)]
pub(crate) struct RecordedRect {
    /// Rectangle in scene coordinates.
    pub(crate) rect: Rect,
    /// Paint applied to the rectangle.
    pub(crate) paint: Paint,
}

impl RecordedDraw {
    fn new_path(strips: Range<usize>, paint: Paint) -> Self {
        Self::Path(RecordedPath { strips, paint })
    }

    fn new_rect(rect: Rect, paint: Paint) -> Self {
        Self::Rect(RecordedRect { rect, paint })
    }
}

impl Drawable for RecordedDraw {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "recorded fast rectangles are clipped to the u16 viewport"
    )]
    fn bbox(&self, strips: &[Strip]) -> Option<RectU16> {
        match self {
            // TODO: The bbox trait should take a reference to _all_ strips, and then we select
            // the correct strips here. Otherwise, callers of this method always have to make sure
            // to slice the strips correctly themselves.
            Self::Path(_) => strip_bbox(strips),
            Self::Rect(rect) => {
                let rect = rect.rect;

                (!rect.is_zero_area()).then(|| {
                    RectU16::new(
                        rect.x0.floor() as u16,
                        rect.y0.floor() as u16,
                        rect.x1.ceil() as u16,
                        rect.y1.ceil() as u16,
                    )
                    .snap_to_tile_coordinates()
                })
            }
        }
    }
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// Configuration for GPU memory used while rendering.
    pub memory_settings: MemorySettings,
}

/// Settings controlling usage of GPU memory.
#[derive(Copy, Clone, Debug, Default)]
pub struct MemorySettings {
    /// Configuration for the atlas holding uploaded images.
    pub image_atlas_config: AtlasConfig,
    /// Configuration for intermediate layer and scratch textures.
    pub layers_config: LayersConfig,
}

/// Configuration for temporary textures used to render layers and effects.
///
/// Layers, blend modes, and filters need temporary GPU textures before their results can be
/// composited into the main scene. This configuration allows you to specify some limits to prevent
/// excessive allocations for complex or adverserial scenes.
///
/// In general, Vello Hybrid has the following priorities in that specific order:
/// - Render as many scenes as possible successfully.
/// - Use as little memory as possible.
/// - Be as performant as possible.
///
/// Vello Hybrid will always prefer minimizing memory usage, even if at the cost of worse
/// performance. However, since the exact tradeoff is very application-dependent,
/// this configuration still allows you to tune the exact parameters such that better performance
/// is accepted at the cost of larger memory consumption.
#[derive(Copy, Clone, Debug)]
pub struct LayersConfig {
    /// Maximum number of intermediate textures that may be allocated.
    ///
    /// `None` permits growth up to device limits. A finite value allows you to impose a hard limit,
    /// but might result in certain scenes being rejected. The exact texture requirements for
    /// different types of scenes are considered an implementation detail, but in general:
    ///
    /// - `0` for scenes without layers, including layers introduced by COLR glyphs.
    /// - `1` for scenes with only single-child layers and no blend modes / filters.
    /// - `4` for scenes with only single-child layers with potentially blend modes
    ///   or simple filters.
    ///
    /// Any other type of complex nested scene might need more intermediate textures depending on
    /// the exact contents.
    pub max_textures: Option<usize>,
    /// Minimum width and height of each allocated intermediate texture.
    ///
    /// By default, textures only grow to fit the largest required layer,
    /// up to [`Self::max_texture_size`], and do not currently shrink between frames. By increasing
    /// the minimum texture size, you can force Vello Hybrid to more aggressively batch layer draws,
    /// at the cost of larger memory consumption.
    ///
    /// Desktop GPUs may benefit from a larger minimum when rendering scenes with many layers. On
    /// mobile GPUs, experiments have shown that keeping intermediate textures smaller is
    /// more important for both memory usage and performance, even if it requires more render passes.
    /// Therefore, you should in general avoid making this value too large. The default of
    /// 512 by 512 is intended as a conservative starting point. Setting this to 1024 by 1024 is a
    /// reasonable choice, too. Higher values are only recommended if benchmarks have proven
    /// that your target workload profits from this and you don't plan on targeting mobile devices.
    pub min_texture_size: SizeU16,
    /// Maximum width and height of each allocated intermediate texture.
    ///
    /// This should be at least the size of the main scene (and potentially more to account for
    /// padding of large filters), unless you explicitly want to reject too large layers.
    pub max_texture_size: SizeU16,
}

impl Default for LayersConfig {
    fn default() -> Self {
        Self {
            max_textures: None,
            min_texture_size: SizeU16::new(512),
            max_texture_size: SizeU16::new(4096),
        }
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            memory_settings: MemorySettings::default(),
        }
    }
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    /// Width of the rendering surface in pixels.
    pub(crate) width: u16,
    /// Height of the rendering surface in pixels.
    pub(crate) height: u16,
    /// Viewport-dependent path and clip generation state.
    viewport_state: ViewportState,
    pub(crate) render_state: RenderState,
    /// Root transform stack.
    root_transforms: RootTransforms,
    pub(crate) aliasing_threshold: Option<u8>,
    // The reason we use `RefCell` here is that during `render`, we need
    // mutable access so we can store additional encoded paints for filtered layers,
    // if applicable.
    /// Storage for encoded non-solid paint data.
    pub(crate) encoded_paints: RefCell<Vec<EncodedPaint>>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Current filter effect applied to individual draw operations.
    filter: Option<Filter>,
    /// The command recorder.
    pub(crate) recorder: CommandRecorder<RecordedDraw>,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        Self {
            width,
            height,
            viewport_state: ViewportState::new(width, height, settings.level),
            render_state: RenderState::default(),
            root_transforms: RootTransforms::default(),
            aliasing_threshold: None,
            encoded_paints: RefCell::new(vec![]),
            paint_visible: true,
            strip_storage: RefCell::new(StripStorage::new(GenerationMode::Append)),
            filter: None,
            recorder: CommandRecorder::new(width, height),
        }
    }

    fn active_width(&self) -> u16 {
        self.viewport_state.width()
    }

    fn active_height(&self) -> u16 {
        self.viewport_state.height()
    }

    fn active_rect(&self) -> Rect {
        Rect::new(
            0.0,
            0.0,
            f64::from(self.active_width()),
            f64::from(self.active_height()),
        )
    }

    fn transforms(&self) -> &Transforms {
        &self.render_state.transforms
    }

    fn transforms_mut(&mut self) -> &mut Transforms {
        &mut self.render_state.transforms
    }

    fn effective_path_transform(&self) -> Affine {
        self.root_transforms
            .effective_path_transform(self.transforms())
    }

    fn effective_paint_transform(&self) -> Affine {
        self.root_transforms
            .effective_paint_transform(self.transforms())
    }

    /// Encode the current paint into a `Paint` that can be used for rendering.
    ///
    /// For solid colors, this is a simple conversion. For gradients and images,
    /// this encodes the paint data into the `encoded_paints` buffer and returns
    /// a `Paint` that references that data. The combined transform (geometry + paint)
    /// is applied during encoding.
    fn encode_current_paint(&mut self) -> Paint {
        // Note: In vello_cpu, during fine rasterization we apply a 0.5 offset to the location
        // to account for the fact that we want to sample the pixel center instead of the top-left
        // corner. For vello_hybrid, we don't need this, because the GPU itself already applies
        // this shift automatically.
        match self.render_state.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => g.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.effective_paint_transform(),
                None,
            ),
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.effective_paint_transform(),
                self.render_state.tint,
            ),
        }
    }

    /// Encode the current external texture into a [`Paint`] that can be used for rendering.
    fn encode_external_texture_paint(
        &mut self,
        texture_id: TextureId,
        source_region: RectU16,
        quality: ImageQuality,
        x_extend: Extend,
        y_extend: Extend,
        transform: Affine,
    ) -> Paint {
        let idx = self.encoded_paints.borrow().len();
        let encoded = EncodedExternalTexture {
            texture_id,
            source_region,
            sampler: ImageSampler {
                x_extend,
                y_extend,
                quality,
                alpha: 1.0,
            },
            // TODO: Make this configurable by the user, and also take `ImageSampler` alpha into
            // account.
            may_have_transparency: true,
            transform: transform.inverse(),
            tint: self.render_state.tint,
        };
        self.encoded_paints
            .borrow_mut()
            .push(EncodedPaint::ExternalTexture(encoded));
        Paint::Indexed(vello_common::paint::IndexedPaint::new(idx))
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let paint = ctx.encode_current_paint();
            ctx.fill_path_with(
                path,
                ctx.effective_path_transform(),
                ctx.render_state.fill_rule,
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    /// Build strips for a filled path with the given properties and record the draw.
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        self.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
            strip_generator.generate_filled_path(
                path,
                fill_rule,
                transform,
                aliasing_threshold,
                strip_storage,
                clip_path,
            );
        });
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        let transform = self.transforms().clip_path_transform();
        self.viewport_state.push_clip(
            path,
            self.render_state.fill_rule,
            transform,
            self.aliasing_threshold,
        );
    }

    /// Pop a clip path from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clip paths before finishing the rendering operation.
    pub fn pop_clip_path(&mut self) {
        self.viewport_state.pop_clip();
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let paint = ctx.encode_current_paint();
            ctx.stroke_path_with(
                path,
                ctx.effective_path_transform(),
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    /// Build strips for a stroked path with the given properties and record the draw.
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let stroke = self.render_state.stroke.clone();
        self.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
            strip_generator.generate_stroked_path(
                path,
                &stroke,
                transform,
                aliasing_threshold,
                strip_storage,
                clip_path,
            );
        });
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

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let paint = ctx.encode_current_paint();

            if let Some(bounds) = ctx.fast_rect_bounds(rect) {
                ctx.recorder
                    .push_draw(RecordedDraw::new_rect(bounds, paint), &[]);

                return;
            }

            let transform = ctx.effective_path_transform();
            if is_axis_aligned(&transform) && ctx.aliasing_threshold.is_none() {
                let transformed_rect = transform.transform_rect_bbox(*rect);
                ctx.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
                    strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        clip_path,
                    );
                });
            } else {
                // TODO: Use a temporary storage for rect paths, like in `vello_cpu`.
                ctx.fill_path_with(
                    &rect.to_path(DEFAULT_TOLERANCE),
                    transform,
                    ctx.render_state.fill_rule,
                    paint,
                    ctx.aliasing_threshold,
                );
            }
        });
    }

    /// Sample rectangular regions from an externally bound texture and draw them with the
    /// corresponding transforms.
    ///
    /// The per-rect transforms are composed with the current
    /// [scene transform][`Self::set_transform`]. This transform is relative to the local region
    /// defined by each [`SampleRect`]: i.e., the origin of each [`SampleRect`] is used only to
    /// determine the region to sample in the source [`TextureId`], and is ignored for determining
    /// the destination. Note that the [`paint transform`](Self::set_paint_transform) has no impact
    /// on this method.
    ///
    /// A texture with the given [`TextureId`] must be supplied at render time. The given
    /// [source regions][`SampleRect::source_region`] must be within bounds of that texture. The
    /// texture is treated as premultiplied alpha in the render target's color space. See the
    /// backend's binding type for more information on texture requirements.
    pub fn draw_texture_rects(
        &mut self,
        texture_id: TextureId,
        quality: ImageQuality,
        rects: impl IntoIterator<Item = SampleRect>,
    ) {
        // This API currently doesn't take extend mode parameters: as of writing, the
        // `render.wgsl` shader does not use extend modes to sample across boundaries, i.e.,
        // sampling near a boundary doesn't take extend modes into account when determining where
        // the sample should be taken.
        //
        // Because in this API the destination drawn is always the transformed input rect, this
        // means extend modes don't currently materially impact rendering. In general drawing with
        // an external texture brush, extend modes would matter, so we still encode them.
        let x_extend = Extend::Pad;
        let y_extend = Extend::Pad;

        self.with_optional_filter_or_blend_layer(|ctx| {
            let use_fast_rect = ctx.can_emit_fast_strips();

            for rect in rects {
                if rect.source_region.is_empty() {
                    continue;
                }

                let w = f64::from(rect.source_region.width());
                let h = f64::from(rect.source_region.height());
                let transform = ctx.effective_path_transform() * rect.transform;

                if use_fast_rect && is_axis_aligned(&transform) {
                    let dst_rect = Rect::new(0., 0., w, h);
                    let transformed_rect = transform
                        .transform_rect_bbox(dst_rect)
                        .intersect(ctx.active_rect());

                    // Skip mirrored or zero-sized rectangles.
                    if transformed_rect.is_zero_area() {
                        continue;
                    }

                    let paint = ctx.encode_external_texture_paint(
                        texture_id,
                        rect.source_region,
                        quality,
                        x_extend,
                        y_extend,
                        transform,
                    );

                    ctx.recorder
                        .push_draw(RecordedDraw::new_rect(transformed_rect, paint), &[]);
                } else {
                    let paint = ctx.encode_external_texture_paint(
                        texture_id,
                        rect.source_region,
                        quality,
                        x_extend,
                        y_extend,
                        transform,
                    );
                    let dst_rect = Rect::new(0., 0., w, h);
                    ctx.fill_path_with(
                        &dst_rect.to_path(DEFAULT_TOLERANCE),
                        transform,
                        ctx.render_state.fill_rule,
                        paint,
                        ctx.aliasing_threshold,
                    );
                }
            }
        });
    }

    fn record_generated_path<F>(&mut self, paint: Paint, generate: F)
    where
        F: FnOnce(&mut StripGenerator, &mut StripStorage, Option<PathDataRef<'_>>),
    {
        let strips = {
            let mut strip_storage = self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.viewport_state
                .with_generator_and_clip(|strip_generator, clip_path| {
                    generate(strip_generator, &mut strip_storage, clip_path);
                });
            strip_start..strip_storage.strips.len()
        };

        let draw = RecordedDraw::new_path(strips.clone(), paint);
        let strip_storage = self.strip_storage.borrow();
        self.recorder.push_draw(draw, &strip_storage.strips[strips]);
    }

    #[inline]
    fn can_emit_fast_strips(&self) -> bool {
        self.viewport_state.clip().is_none() && self.aliasing_threshold.is_none()
    }

    fn fast_rect_bounds(&self, rect: &Rect) -> Option<Rect> {
        if !self.can_emit_fast_strips() {
            return None;
        }

        // TODO: Either bail out or properly implement the case where `aliasing_threshold` is set.

        // We can't handle skewed rectangles.
        // TODO: Maybe support rotated rectangles (https://github.com/linebender/vello/pull/1482#discussion_r2881223621)
        let transform = self.effective_path_transform();
        if !is_axis_aligned(&transform) {
            return None;
        }

        let transformed_rect = transform
            .transform_rect_bbox(*rect)
            .intersect(self.active_rect());

        // Can't handle mirrored or zero-sized rectangles.
        if transformed_rect.is_zero_area() {
            return None;
        }

        Some(transformed_rect)
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Fill a blurred rectangle with the given corner radius and standard deviation.
    ///
    /// When `invert` is `true`, the inverse (`1 - alpha`) of the blur coverage is painted: the
    /// paint is fully opaque outside the blurred rectangle and fades to transparent inside it. This
    /// can be used to implement inset box shadows.
    ///
    /// This operation uses the current transform and paint transform. Like Vello CPU, it only
    /// uses solid paints; non-solid paints fall back to black.
    pub fn fill_blurred_rounded_rect(
        &mut self,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
        invert: bool,
    ) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let rect = rect.abs();
            let color = match ctx.render_state.paint {
                PaintType::Solid(s) => s,
                _ => BLACK,
            };
            let blurred_rect = BlurredRoundedRectangle {
                rect,
                color,
                radius,
                std_dev,
                invert,
            };

            let kernel_size = 2.5 * std_dev;
            let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
            let transform = ctx.effective_paint_transform();
            let paint =
                blurred_rect.encode_into(&mut ctx.encoded_paints.borrow_mut(), transform, None);

            if let Some(bounds) = ctx.fast_rect_bounds(&inflated_rect) {
                ctx.recorder
                    .push_draw(RecordedDraw::new_rect(bounds, paint), &[]);
                return;
            }

            let path_transform = ctx.effective_path_transform();
            if is_axis_aligned(&path_transform) && ctx.aliasing_threshold.is_none() {
                let transformed_rect = path_transform.transform_rect_bbox(inflated_rect);
                ctx.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
                    strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        clip_path,
                    );
                });
            } else {
                ctx.fill_path_with(
                    &inflated_rect.to_path(DEFAULT_TOLERANCE),
                    path_transform,
                    Fill::NonZero,
                    paint,
                    ctx.aliasing_threshold,
                );
            }
        });
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run<'a>(
        &'a mut self,
        resources: &'a mut Resources,
        font: &FontData,
    ) -> GlyphRunBuilder<'a> {
        glifo::GlyphRunBuilder::new(
            font.clone(),
            self.transforms().scene_transform(),
            *self.transforms().paint_transform(),
            crate::text::HybridGlyphRunBackend {
                scene: self,
                resources,
                atlas_cache_enabled: false,
            },
        )
    }

    /// Push a new layer with the given properties.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        if mask.is_some() {
            unimplemented!("mask layers are currently not supported");
        }

        let blend_mode = blend_mode.unwrap_or_default();
        let layer_transform = self.effective_path_transform();
        let filter_data = filter.map(|filter| FilterData::new(filter, layer_transform));
        let relative_root_transform =
            filter_data
                .as_ref()
                .map_or(Affine::IDENTITY, |filter_data| {
                    let (shift_x, shift_y) = filter_data.source_shift();
                    Affine::translate((f64::from(shift_x), f64::from(shift_y)))
                });
        self.root_transforms.push_root(relative_root_transform);
        if let Some(filter_plan) = &filter_data {
            self.viewport_state.push_root_viewport(filter_plan);
        }

        let clip_path = clip_path.map(|path| {
            let mut strip_storage = self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.viewport_state
                .with_generator_and_clip(|strip_generator, existing_clip| {
                    strip_generator.generate_filled_path(
                        path,
                        self.render_state.fill_rule,
                        layer_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        existing_clip,
                    );

                    let strip_range = strip_start..strip_storage.strips.len();
                    LayerClip {
                        bbox: strip_bbox(&strip_storage.strips[strip_range.clone()])
                            .unwrap_or(RectU16::ZERO),
                        strip_range,
                        thread_idx: 0,
                    }
                })
        });

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity: opacity.unwrap_or(1.0),
                mask: None,
                clip_path,
            },
            filter_data,
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

    /// Push a new mask layer.
    ///
    /// Note that masks are not yet supported in `vello_hybrid`.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a new filter layer.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        if self.recorder.pop_layer() == PoppedLayer::Filter {
            self.viewport_state.pop_root_viewport();
        }
        self.root_transforms.pop_root();
    }

    /// Set the blend mode for subsequent rendering operations.
    ///
    /// # Panics
    ///
    /// Panics if `blend_mode` is destructive, which is currently not supported for
    /// non-isolated blends. You need to use clip layers instead if you need that behavior.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        assert!(
            !blend_mode.is_destructive(),
            "destructive blend modes are currently not supported"
        );

        self.render_state.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.render_state.stroke = stroke;
    }

    /// Get the current stroke.
    pub fn stroke(&self) -> &Stroke {
        &self.render_state.stroke
    }

    /// Get a mutable reference to the current stroke.
    #[cfg(feature = "text")]
    pub(crate) fn stroke_mut(&mut self) -> &mut Stroke {
        &mut self.render_state.stroke
    }

    /// Set the paint for subsequent rendering operations.
    // TODO: This API is not final. Supporting images from a pixmap is explicitly out of scope.
    //       Instead images should be passed via a backend-agnostic opaque id, and be hydrated at
    //       render time into a texture usable by the renderer backend.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.render_state.paint = paint.into();
        self.set_paint_visible();
    }

    fn set_paint_visible(&mut self) {
        self.paint_visible = match &self.render_state.paint {
            PaintType::Solid(color) => color.components[3] != 0.0,
            _ => true,
        };
    }

    /// Set the tint for subsequent image paint operations.
    pub fn set_tint(&mut self, tint: Option<Tint>) {
        self.render_state.tint = tint;
    }

    /// Clear the tint, so subsequent image paints are drawn without tinting.
    pub fn reset_tint(&mut self) {
        self.render_state.tint = None;
    }

    /// Get the current paint.
    pub fn paint(&self) -> &PaintType {
        &self.render_state.paint
    }
    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.transforms_mut().set_paint_transform(paint_transform);
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.transforms_mut().reset_paint_transform();
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.render_state.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transforms_mut().set_transform(transform);
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transforms_mut().reset_transform();
    }

    /// Apply filter to the current paint (affects next drawn element).
    pub fn set_filter_effect(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        self.filter = None;
    }

    fn with_optional_filter_or_blend_layer<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let blend_mode = self.render_state.blend_mode;
        let blend_mode = (blend_mode != BlendMode::default()).then_some(blend_mode);
        let filter = self.filter.clone();

        if blend_mode.is_some() || filter.is_some() {
            self.push_layer(None, blend_mode, None, None, filter);
            let result = f(self);
            self.pop_layer();
            result
        } else {
            f(self)
        }
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.viewport_state.reset(self.width, self.height);
        {
            let mut ss = self.strip_storage.borrow_mut();
            ss.clear();
        }
        self.encoded_paints.borrow_mut().clear();

        self.root_transforms.reset();
        self.render_state.reset();

        self.recorder.reset(self.width, self.height);
        self.filter = None;
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Take current rendering state and reset the existing state to its default.
    pub fn take_current_state(&mut self) -> RenderState {
        let state = core::mem::take(&mut self.render_state);
        self.set_paint_visible();

        state
    }

    /// Save a copy of the current rendering state.
    pub fn save_current_state(&mut self) -> RenderState {
        self.render_state.clone()
    }

    /// Restore rendering state.
    pub fn restore_state(&mut self, state: RenderState) {
        self.render_state = state;
        self.set_paint_visible();
    }
}

#[cfg(test)]
mod tests {
    use super::RecordedDraw;
    #[cfg(feature = "text")]
    use super::Scene;
    #[cfg(feature = "text")]
    use crate::resources::Resources;
    #[cfg(feature = "text")]
    use alloc::sync::Arc;
    #[cfg(feature = "text")]
    use glifo::Glyph;
    #[cfg(feature = "text")]
    use vello_common::kurbo::BezPath;
    use vello_common::kurbo::Rect;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::color::palette::css::BLUE;
    #[cfg(feature = "text")]
    use vello_common::peniko::{Blob, FontData};
    use vello_common::record::Drawable;

    #[test]
    fn recorded_rect_bbox_rounds_fractional_outward() {
        let draw = RecordedDraw::new_rect(
            Rect::new(0.0, 0.0, 4.5, 4.5),
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
        );

        assert_eq!(
            draw.bbox(&[]),
            Some(vello_common::geometry::RectU16::new(0, 0, 8, 8))
        );
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

        let mut scene = Scene::new(200, 200);
        let mut resources = Resources::new();
        let mut triangle = BezPath::new();
        triangle.move_to((10.0, 10.0));
        triangle.line_to((90.0, 50.0));
        triangle.line_to((10.0, 90.0));
        triangle.close_path();

        scene.fill_rect(&Rect::new(10.0, 10.0, 50.0, 50.0));
        scene.fill_path(&triangle);
        scene
            .glyph_run(&mut resources, &font)
            .fill_glyphs(glyphs.into_iter());

        assert!(resources.glyph_resources.is_none());

        scene
            .glyph_run(&mut resources, &font)
            .atlas_cache(true)
            .fill_glyphs(glyphs.into_iter());

        assert!(resources.glyph_resources.is_some());
    }
}
