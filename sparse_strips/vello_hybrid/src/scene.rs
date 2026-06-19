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
use vello_common::clip::ClipState;
use vello_common::coarse::{MODE_HYBRID, Wide, WideTilesBbox};
use vello_common::encode::{EncodeExt, EncodedExternalTexture, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter::FilterData;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Paint, PaintType, Tint};
#[cfg(feature = "text")]
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Extend, Fill, ImageQuality, ImageSampler, Mix};
use vello_common::record::{CommandRecorder, Drawable, LayerClip, LayerProps, PoppedLayer};
use vello_common::render_graph::{RenderGraph, RenderNodeKind};
use vello_common::render_state::RenderState;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::util::{control_point_bbox_u16, is_axis_aligned, strip_bbox};

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// The pipeline mode for strip rendering.
///
/// Determines whether strips are sent directly to the GPU (fast path),
/// go through coarse rasterization, or a mix of both.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StripPathMode {
    /// No layers have been pushed. All strips go directly to the fast buffer,
    /// bypassing coarse rasterization entirely.
    ///
    /// `StripStorage` is in `Append` mode.
    #[default]
    FastOnly,
    /// This mode is activated if there has been a `push_layer` call, but the user indicated
    /// that they will only use src-over blending.
    ///
    /// In this case, we will alternate between render fast strips and coarse-rasterized
    /// layers. Which of the two modes is active is dependent on whether `wide.has_layers()` is
    /// true.
    ///
    /// `StripStorage` alternates between `Append` (for the root level) and
    /// `ReplaceAfter(n)` (inside a layer).
    Interleaved,
    /// This mode is activated if the user indicated not src-over blends might happen,
    /// and there has been at least one `push_layer` call. All previous strips will be
    /// retroactively coarse-rasterized, and from now on we always go through coarse
    /// rasterization.
    ///
    /// `StripStorage` is in `Replace` mode.
    CoarseOnly,
}

/// Metadata for a single path stored in the fast strips buffer.
#[derive(Debug)]
pub(crate) struct FastStripsPath {
    /// The range of strips for this path in the `strips` buffer.
    pub(crate) strips: Range<usize>,
    /// The paint of the path.
    pub(crate) paint: Paint,
}

/// A rectangle stored in the fast-path buffer.
#[derive(Debug)]
pub(crate) struct FastPathRect {
    pub(crate) x0: f32,
    pub(crate) y0: f32,
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) paint: Paint,
}

/// A command in the fast strips buffer.
#[derive(Debug)]
pub(crate) enum FastStripCommand {
    /// A path rendered via the normal strip pipeline.
    Path(FastStripsPath),
    /// A rectangle.
    Rect(FastPathRect),
}

/// A draw command recorded for the new scheduler.
#[derive(Debug)]
pub(crate) enum RecordedDraw {
    /// A path rendered via the normal strip pipeline.
    Path(RecordedPath),
    /// A rectangle.
    Rect(RecordedRect),
}

/// A recorded path draw for the new scheduler.
#[derive(Debug)]
pub(crate) struct RecordedPath {
    /// The range of strips for this path in [`Scene::strip_storage`].
    pub(crate) strips: Range<usize>,
    /// The paint of the path.
    pub(crate) paint: Paint,
    /// The draw bounds in viewport coordinates.
    bbox: RectU16,
}

/// A recorded rectangle draw for the new scheduler.
#[derive(Debug)]
pub(crate) struct RecordedRect {
    /// The rectangle data.
    pub(crate) rect: FastPathRect,
    /// The draw bounds in viewport coordinates.
    bbox: RectU16,
}

impl RecordedDraw {
    fn path(
        strips: Range<usize>,
        strip_storage: &StripStorage,
        viewport_width: u16,
        paint: Paint,
    ) -> Self {
        let bbox = strip_bbox(&strip_storage.strips[strips.clone()], viewport_width);
        Self::Path(RecordedPath {
            strips,
            paint,
            bbox,
        })
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Rectangles are clipped to the u16 viewport before recording."
    )]
    fn rect(rect: FastPathRect) -> Self {
        let bbox = RectU16::new(
            rect.x0.floor() as u16,
            rect.y0.floor() as u16,
            rect.x1.ceil() as u16,
            rect.y1.ceil() as u16,
        );
        Self::Rect(RecordedRect { rect, bbox })
    }
}

impl Drawable for RecordedDraw {
    fn bbox(&self) -> RectU16 {
        match self {
            Self::Path(path) => path.bbox,
            Self::Rect(rect) => rect.bbox,
        }
    }
}

/// A buffer that collects strips from paths that are rendered directly to the surface,
/// bypassing coarse rasterization.
///
/// Strip data itself lives in `strip_storage`. Each `FastStripsPath` records the range of strips
/// for one path within that storage.
#[derive(Debug, Default)]
pub(crate) struct FastStripsBuffer {
    /// All commands in the buffer.
    pub(crate) commands: Vec<FastStripCommand>,
}

impl FastStripsBuffer {
    #[inline(always)]
    fn clear(&mut self) {
        self.commands.clear();
    }
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The configuration for the texture atlas.
    ///
    /// This controls how images are managed in GPU memory through texture atlases.
    /// The atlas system packs multiple images into larger textures to reduce the
    /// number of GPU texture bindings. This config allows customizing atlas parameters such as:
    /// - The number and size of atlases
    /// - How images are allocated across multiple atlases
    /// - Whether new atlases are automatically created when needed
    ///
    /// Adjusting these settings can affect memory usage and rendering performance
    /// depending on your application's image usage patterns.
    pub atlas_config: AtlasConfig,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            atlas_config: AtlasConfig::default(),
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
    /// Wide coarse rasterizer for generating binned draw commands.
    pub(crate) wide: Wide<MODE_HYBRID>,
    clip_context: ClipState,
    level: Level,
    root_transforms: Vec<Affine>,
    pub(crate) render_state: RenderState,
    pub(crate) aliasing_threshold: Option<u8>,
    // The reason we use `RefCell` here is that during `render`, we need
    // mutable access so we can store additional encoded paints for filtered layers,
    // if applicable.
    /// Storage for encoded non-solid paint data.
    pub(crate) encoded_paints: RefCell<Vec<EncodedPaint>>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Generator for converting paths to strips.
    pub(crate) strip_generator: StripGenerator,
    strip_generator_stack: Vec<StripGenerator>,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Counter for generating unique layer IDs.
    layer_id_next: u32,
    /// Dependency graph for managing layer rendering order and filter effects.
    pub(crate) render_graph: RenderGraph,
    /// Current filter effect applied to individual draw operations.
    filter: Option<Filter>,
    /// Recorded command stream consumed by the new scheduler.
    pub(crate) recorder: CommandRecorder<RecordedDraw>,
    /// A buffer that stores the strips of path drawing calls that are rendered directly
    /// to the surface, bypassing coarse rasterization.
    pub(crate) fast_strips_buffer: FastStripsBuffer,
    /// The current strip rendering pipeline mode.
    pub(crate) strip_path_mode: StripPathMode,
    /// Split points in `fast_strips_buffer.paths` that mark boundaries where we must
    /// process one coarse batch before processing another fast path strip batch.
    /// Only meaningful in [`StripPathMode::Interleaved`] mode.
    pub(crate) coarse_batch_splits: Vec<usize>,
}

// We use this macro instead of a method to avoid borrowing issues in the corresponding methods.
//
// When the fast path is active AND we're at the top level (no layers pushed),
// strip_storage is in `Append` mode, so `$strip_start` (captured before generation)
// and the current length delimit the range for this path.
//
// When the fast path is inactive or we're inside a layer, `strip_storage` is in `Replace`
// or `ReplaceAfter` mode where each generation starts with a clear/truncate, so the
// relevant portion of the buffer is the current path's strips.
#[allow(unused_macros)]
macro_rules! submit_strips {
    ($self:ident, $strip_storage:expr, $strip_start:expr, $paint:expr) => {
        if $self.strip_path_mode != StripPathMode::CoarseOnly && !$self.wide.has_layers() {
            $self
                .fast_strips_buffer
                .commands
                .push(FastStripCommand::Path(FastStripsPath {
                    strips: $strip_start..$strip_storage.strips.len(),
                    paint: $paint,
                }));
        } else {
            // In `ReplaceAfter(n)` mode the fast path prefix lives at `[0..n]`
            // and must not be fed into the coarse rasterizer.
            let coarse_start = match $strip_storage.generation_mode() {
                GenerationMode::ReplaceAfter(n) => n,
                _ => 0,
            };
            $self.wide.generate(
                &$strip_storage.strips[coarse_start..],
                $paint,
                $self.render_state.blend_mode,
                0,
                None,
                &$self.encoded_paints.borrow(),
            );
        }
    };
}

const DEFAULT_BLEND_MODE: BlendMode = BlendMode::new(Mix::Normal, Compose::SrcOver);

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        let mut render_graph = RenderGraph::new();

        let wide = Wide::<MODE_HYBRID>::new(width, height, true);

        // Create root node (layer_id 0) as the first node (will be node 0).
        // This ensures the root layer is always rendered last in the execution order.
        let wtile_bbox = WideTilesBbox::new(0, 0, wide.width_tiles(), wide.height_tiles());
        let _ = render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox,
        });

        Self {
            width,
            height,
            wide,
            clip_context: ClipState::new(),
            level: settings.level,
            root_transforms: vec![Affine::IDENTITY],
            render_state: RenderState::default(),
            aliasing_threshold: None,
            encoded_paints: RefCell::new(vec![]),
            paint_visible: true,
            strip_generator: StripGenerator::new(width, height, settings.level),
            strip_generator_stack: Vec::new(),
            // Start strip storage in `Append` mode since we enable the fast path by default.
            strip_storage: RefCell::new(StripStorage::new(GenerationMode::Append)),
            layer_id_next: 0,
            render_graph,
            filter: None,
            recorder: CommandRecorder::new(),
            fast_strips_buffer: FastStripsBuffer::default(),
            strip_path_mode: StripPathMode::FastOnly,
            coarse_batch_splits: Vec::new(),
        }
    }

    fn root_transform(&self) -> Affine {
        *self
            .root_transforms
            .last()
            .expect("root transform stack should never be empty")
    }

    fn effective_path_transform(&self) -> Affine {
        self.root_transform() * self.render_state.transform
    }

    fn effective_paint_transform(&self) -> Affine {
        self.effective_path_transform() * self.render_state.paint_transform
    }

    fn push_root_transform(&mut self, relative_transform: Affine) {
        self.root_transforms
            .push(relative_transform * self.root_transform());
    }

    fn pop_root_transform(&mut self) {
        self.root_transforms.pop();
    }

    fn active_width(&self) -> u16 {
        self.strip_generator.width()
    }

    fn active_height(&self) -> u16 {
        self.strip_generator.height()
    }

    fn push_filter_surface(&mut self, filter_data: &FilterData) {
        let padding = filter_data.source_padding;
        let width = self
            .strip_generator
            .width()
            .saturating_add(padding.x0)
            .saturating_add(padding.x1);
        let height = self
            .strip_generator
            .height()
            .saturating_add(padding.y0)
            .saturating_add(padding.y1);
        let filter_generator = StripGenerator::new(width, height, self.level);
        let parent_generator = core::mem::replace(&mut self.strip_generator, filter_generator);
        self.strip_generator_stack.push(parent_generator);

        self.clip_context
            .push_filter_viewport(filter_data.source_shift(), &mut self.strip_generator);
    }

    fn pop_filter_surface(&mut self) {
        self.strip_generator = self
            .strip_generator_stack
            .pop()
            .expect("filter viewport stack underflow");
        self.clip_context
            .pop_filter_viewport(&mut self.strip_generator);
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

        self.with_optional_filter(|ctx| {
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

    /// Build strips for a filled path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a path
    /// and submits them to the coarse rasterizer. The path is first converted
    /// to strips by the strip generator, then the strips are processed by the
    /// wide coarse rasterizer to generate binned draw commands.
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let (strips, draw) = {
            let strip_storage = &mut self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.strip_generator.generate_filled_path(
                path,
                fill_rule,
                transform,
                aliasing_threshold,
                strip_storage,
                self.clip_context.get(),
            );

            let strips = strip_start..strip_storage.strips.len();
            let draw = RecordedDraw::path(
                strips.clone(),
                strip_storage,
                self.active_width(),
                paint.clone(),
            );
            (strips, draw)
        };

        self.record_path(strips, paint, draw);
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            self.render_state.fill_rule,
            self.render_state.transform,
            self.aliasing_threshold,
        );
    }

    /// Pop a clip path from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clip paths before finishing the rendering operation.
    pub fn pop_clip_path(&mut self) {
        self.clip_context.pop_clip();
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter(|ctx| {
            let paint = ctx.encode_current_paint();
            ctx.stroke_path_with(
                path,
                ctx.effective_path_transform(),
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    /// Build strips for a stroked path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a stroked path
    /// and submits them to the coarse rasterizer. The path is first stroked and
    /// converted to strips by the strip generator, then the strips are processed by
    /// the wide coarse rasterizer to generate binned draw commands.
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let (strips, draw) = {
            let strip_storage = &mut self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.strip_generator.generate_stroked_path(
                path,
                &self.render_state.stroke,
                transform,
                aliasing_threshold,
                strip_storage,
                self.clip_context.get(),
            );

            let strips = strip_start..strip_storage.strips.len();
            let draw = RecordedDraw::path(
                strips.clone(),
                strip_storage,
                self.active_width(),
                paint.clone(),
            );
            (strips, draw)
        };

        self.record_path(strips, paint, draw);
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

        if self.try_fast_rect(rect) {
            return;
        }

        let transform = self.effective_path_transform();
        if is_axis_aligned(&transform) && self.aliasing_threshold.is_none() {
            self.with_optional_filter(|ctx| {
                let paint = ctx.encode_current_paint();
                let transformed_rect = ctx.effective_path_transform().transform_rect_bbox(*rect);
                let (strips, draw) = {
                    let strip_storage = &mut ctx.strip_storage.borrow_mut();
                    let strip_start = strip_storage.strips.len();
                    ctx.strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        ctx.clip_context.get(),
                    );

                    let strips = strip_start..strip_storage.strips.len();
                    let draw = RecordedDraw::path(
                        strips.clone(),
                        strip_storage,
                        ctx.active_width(),
                        paint.clone(),
                    );
                    (strips, draw)
                };

                ctx.record_path(strips, paint, draw);
            });
        } else {
            // TODO: Use a temporary storage for rect paths, like in `vello_cpu`.
            self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
        }
    }

    fn try_fast_rect(&mut self, rect: &Rect) -> bool {
        let Some(bounds) = self.fast_rect_bounds(rect) else {
            return false;
        };

        let paint = self.encode_current_paint();
        self.push_fast_rect(bounds, paint);
        true
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→f32 truncation is acceptable for pixel coordinates"
    )]
    pub fn draw_texture_rects(
        &mut self,
        texture_id: TextureId,
        quality: ImageQuality,
        rects: impl IntoIterator<Item = SampleRect>,
    ) {
        // This API currently doesn't take extend mode parameters: as of writing, the
        // `render_strips.wgsl` shader does not use extend modes to sample across boundaries, i.e.,
        // sampling near a boundary doesn't take extend modes into account when determining where
        // the sample should be taken.
        //
        // Because in this API the destination drawn is always the transformed input rect, this
        // means extend modes don't currently materially impact rendering. In general drawing with
        // an external texture brush, extend modes would matter, so we still encode them.
        let x_extend = Extend::Pad;
        let y_extend = Extend::Pad;

        if self.can_emit_fast_strips() {
            for rect in rects {
                if rect.source_region.is_empty() {
                    continue;
                }

                let w = f64::from(rect.source_region.width());
                let h = f64::from(rect.source_region.height());
                let transform = self.effective_path_transform() * rect.transform;

                if !is_axis_aligned(&transform) {
                    // Non-axis-aligned rects fall back to the strip path (still
                    // in the fast buffer since we checked the global conditions).
                    let paint = self.encode_external_texture_paint(
                        texture_id,
                        rect.source_region,
                        quality,
                        x_extend,
                        y_extend,
                        transform,
                    );
                    let dst_rect = Rect::new(0., 0., w, h);
                    self.fill_path_with(
                        &dst_rect.to_path(DEFAULT_TOLERANCE),
                        transform,
                        self.render_state.fill_rule,
                        paint,
                        self.aliasing_threshold,
                    );
                    continue;
                }

                let dst_rect = Rect::new(0., 0., w, h);
                let transformed_rect = transform.transform_rect_bbox(dst_rect);

                let x0 = transformed_rect
                    .x0
                    .max(0.)
                    .min(f64::from(self.active_width()));
                let y0 = transformed_rect
                    .y0
                    .max(0.)
                    .min(f64::from(self.active_height()));
                let x1 = transformed_rect
                    .x1
                    .max(0.)
                    .min(f64::from(self.active_width()));
                let y1 = transformed_rect
                    .y1
                    .max(0.)
                    .min(f64::from(self.active_height()));

                // Skip mirrored or zero-sized rectangles.
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }

                let paint = self.encode_external_texture_paint(
                    texture_id,
                    rect.source_region,
                    quality,
                    x_extend,
                    y_extend,
                    transform,
                );

                self.record_rect(FastPathRect {
                    x0: x0 as f32,
                    y0: y0 as f32,
                    x1: x1 as f32,
                    y1: y1 as f32,
                    paint,
                });
            }
        } else {
            self.with_optional_filter(|ctx| {
                for rect in rects {
                    if rect.source_region.is_empty() {
                        continue;
                    }

                    let w = f64::from(rect.source_region.width());
                    let h = f64::from(rect.source_region.height());
                    let transform = ctx.effective_path_transform() * rect.transform;
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
            });
        }
    }

    /// Whether we're in a state that allows pushing commands directly into
    /// [`Self::fast_strips_buffer`], bypassing coarse rasterization.
    #[inline]
    fn can_emit_fast_strips(&self) -> bool {
        self.strip_path_mode != StripPathMode::CoarseOnly
            && !self.wide.has_layers()
            && self.filter.is_none()
            && self.clip_context.get().is_none()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→f32 truncation is acceptable for pixel coordinates"
    )]
    fn push_fast_rect(&mut self, bounds: Rect, paint: Paint) {
        self.record_rect(FastPathRect {
            x0: bounds.x0 as f32,
            y0: bounds.y0 as f32,
            x1: bounds.x1 as f32,
            y1: bounds.y1 as f32,
            paint,
        });
    }

    fn record_path(&mut self, strips: Range<usize>, paint: Paint, draw: RecordedDraw) {
        self.fast_strips_buffer
            .commands
            .push(FastStripCommand::Path(FastStripsPath {
                strips: strips.clone(),
                paint: paint.clone(),
            }));
        self.push_recorded_draw(draw);
    }

    fn record_rect(&mut self, rect: FastPathRect) {
        self.fast_strips_buffer
            .commands
            .push(FastStripCommand::Rect(FastPathRect {
                x0: rect.x0,
                y0: rect.y0,
                x1: rect.x1,
                y1: rect.y1,
                paint: rect.paint.clone(),
            }));
        self.push_recorded_draw(RecordedDraw::rect(rect));
    }

    fn push_recorded_draw(&mut self, draw: RecordedDraw) {
        let blend_mode = self.render_state.blend_mode;
        if blend_mode == DEFAULT_BLEND_MODE {
            self.recorder.push_draw(draw);
            return;
        }

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity: 1.0,
                mask: None,
                clip_path: None,
            },
            None,
        );
        self.recorder.push_draw(draw);
        self.recorder.pop_layer();
    }

    fn fast_rect_bounds(&self, rect: &Rect) -> Option<Rect> {
        if !self.can_emit_fast_strips() {
            return None;
        }

        // TODO: Either bail out or properly implement the case where `aliasing_threshold` is set.
        // Also update the code in `flush_fast_path`.

        // We can't handle skewed rectangles.
        // TODO: Maybe support rotated rectangles (https://github.com/linebender/vello/pull/1482#discussion_r2881223621)
        let transform = self.effective_path_transform();
        if !is_axis_aligned(&transform) {
            return None;
        }

        let transformed_rect = transform.transform_rect_bbox(*rect);

        let x0 = transformed_rect
            .x0
            .max(0.0)
            .min(f64::from(self.active_width()));
        let y0 = transformed_rect
            .y0
            .max(0.0)
            .min(f64::from(self.active_height()));
        let x1 = transformed_rect
            .x1
            .max(0.0)
            .min(f64::from(self.active_width()));
        let y1 = transformed_rect
            .y1
            .max(0.0)
            .min(f64::from(self.active_height()));

        // Can't handle mirrored or zero-sized rectangles.
        if x1 <= x0 || y1 <= y0 {
            return None;
        }

        Some(Rect::new(x0, y0, x1, y1))
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Fill a blurred rectangle with the given corner radius and standard deviation.
    ///
    /// This operation uses the current transform and paint transform. Like Vello CPU, it only
    /// uses solid paints; non-solid paints fall back to black.
    pub fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter(|ctx| {
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
            };

            let kernel_size = 2.5 * std_dev;
            let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
            let transform = ctx.effective_paint_transform();
            let paint =
                blurred_rect.encode_into(&mut ctx.encoded_paints.borrow_mut(), transform, None);

            if let Some(bounds) = ctx.fast_rect_bounds(&inflated_rect) {
                ctx.push_fast_rect(bounds, paint);
                return;
            }

            let path_transform = ctx.effective_path_transform();
            if is_axis_aligned(&path_transform) && ctx.aliasing_threshold.is_none() {
                let transformed_rect = path_transform.transform_rect_bbox(inflated_rect);
                let (strips, draw) = {
                    let strip_storage = &mut ctx.strip_storage.borrow_mut();
                    let strip_start = strip_storage.strips.len();
                    ctx.strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        ctx.clip_context.get(),
                    );

                    let strips = strip_start..strip_storage.strips.len();
                    let draw = RecordedDraw::path(
                        strips.clone(),
                        strip_storage,
                        ctx.active_width(),
                        paint.clone(),
                    );
                    (strips, draw)
                };

                ctx.record_path(strips, paint, draw);
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
            self.render_state.transform,
            self.render_state.paint_transform,
            crate::text::HybridGlyphRunBackend {
                scene: self,
                resources,
                atlas_cache_enabled: false,
            },
        )
    }

    /// Flush the fast path buffer through the normal coarse rasterization pipeline.
    ///
    /// This retroactively generates wide tile commands for all strips that have been generated
    /// using the fast path.
    ///
    /// After this call, `strip_storage` is switched back to `Replace` mode.
    #[allow(dead_code)]
    fn flush_fast_path(&mut self) {
        if self.strip_path_mode == StripPathMode::CoarseOnly {
            return;
        }

        let mut strip_storage = self.strip_storage.borrow_mut();
        for cmd in self.fast_strips_buffer.commands.drain(..) {
            match cmd {
                FastStripCommand::Path(path) => {
                    self.wide.generate(
                        &strip_storage.strips[path.strips],
                        path.paint,
                        BlendMode::default(),
                        0,
                        None,
                        &self.encoded_paints.borrow(),
                    );
                }
                FastStripCommand::Rect(r) => {
                    let rect = Rect::new(
                        f64::from(r.x0),
                        f64::from(r.y0),
                        f64::from(r.x1),
                        f64::from(r.y1),
                    );
                    let strip_start = strip_storage.strips.len();
                    self.strip_generator
                        .generate_filled_rect_fast(&rect, &mut strip_storage, None);
                    self.wide.generate(
                        &strip_storage.strips[strip_start..],
                        r.paint,
                        BlendMode::default(),
                        0,
                        None,
                        &self.encoded_paints.borrow(),
                    );
                }
            }
        }

        strip_storage.set_generation_mode(GenerationMode::Replace);
        self.strip_path_mode = StripPathMode::CoarseOnly;
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
            unimplemented!("mask layers are not supported by the new vello_hybrid scheduler yet");
        }

        let blend_mode = blend_mode.unwrap_or(DEFAULT_BLEND_MODE);
        let layer_transform = self.effective_path_transform();
        let filter_plan = filter.map(|filter| FilterData::new(filter, layer_transform));
        self.push_root_transform(
            filter_plan
                .as_ref()
                .map_or(Affine::IDENTITY, |filter_plan| {
                    let (shift_x, shift_y) = filter_plan.source_shift();
                    Affine::translate((f64::from(shift_x), f64::from(shift_y)))
                }),
        );
        if let Some(filter_plan) = &filter_plan {
            self.push_filter_surface(filter_plan);
        }

        let clip_path = clip_path.map(|path| {
            let existing_clip = self.clip_context.get();
            let mut bbox = control_point_bbox_u16(path.iter(), layer_transform);
            if let Some(existing_clip) = existing_clip {
                bbox = bbox.intersect(existing_clip.bbox);
            }

            let mut strip_storage = self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.strip_generator.generate_filled_path(
                path,
                self.render_state.fill_rule,
                layer_transform,
                self.aliasing_threshold,
                &mut strip_storage,
                existing_clip,
            );

            let strip_range = strip_start..strip_storage.strips.len();
            LayerClip {
                strip_range,
                thread_idx: 0,
                bbox,
            }
        });

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity: opacity.unwrap_or(1.0),
                mask: None,
                clip_path,
            },
            filter_plan,
        );
    }

    #[allow(dead_code)]
    fn push_layer_old(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        let blend_mode_val = blend_mode.unwrap_or(DEFAULT_BLEND_MODE);

        self.layer_id_next += 1;

        let strip_offset = 0;
        self.flush_fast_path();

        let mut strip_storage = self.strip_storage.borrow_mut();

        let clip = if let Some(c) = clip_path {
            self.strip_generator.generate_filled_path(
                c,
                self.render_state.fill_rule,
                self.render_state.transform,
                self.aliasing_threshold,
                &mut strip_storage,
                self.clip_context.get(),
            );

            Some(&strip_storage.strips[strip_offset..])
        } else {
            None
        };

        // Mask is unsupported. Blend is partially supported.
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            self.layer_id_next,
            clip,
            blend_mode_val,
            None,
            opacity.unwrap_or(1.),
            filter,
            self.render_state.transform,
            &mut self.render_graph,
            0,
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
    ///
    /// Note that filters are currently ignored in `vello_hybrid`.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        if self.recorder.pop_layer() == PoppedLayer::Filter {
            self.pop_filter_surface();
        }
        self.pop_root_transform();
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
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
        self.render_state.paint_transform = paint_transform;
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.render_state.paint_transform = Affine::IDENTITY;
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.render_state.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.render_state.transform = transform;
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.render_state.transform = Affine::IDENTITY;
    }

    /// Apply filter to the current paint (affects next drawn element).
    pub fn set_filter_effect(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        self.filter = None;
    }

    fn with_optional_filter<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        if let Some(filter) = self.filter.clone() {
            self.push_filter_layer(filter);
            f(self);
            self.pop_layer();
        } else {
            f(self);
        }
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator = StripGenerator::new(self.width, self.height, self.level);
        self.strip_generator_stack.clear();
        self.clip_context.reset();
        self.root_transforms.clear();
        self.root_transforms.push(Affine::IDENTITY);
        // Set the strip storage back to `Append` mode since the fast path is re-enabled on reset.
        {
            let mut ss = self.strip_storage.borrow_mut();
            ss.clear();
            ss.set_generation_mode(GenerationMode::Append);
        }
        self.encoded_paints.borrow_mut().clear();

        self.render_state.reset();

        self.fast_strips_buffer.clear();
        self.recorder.reset();
        self.strip_path_mode = StripPathMode::FastOnly;
        self.coarse_batch_splits.clear();

        self.layer_id_next = 0;
        self.render_graph.clear();
        let wtile_bbox =
            WideTilesBbox::new(0, 0, self.wide.width_tiles(), self.wide.height_tiles());
        self.render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox,
        });
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
    use super::*;
    #[cfg(feature = "text")]
    use crate::resources::Resources;
    #[cfg(feature = "text")]
    use alloc::sync::Arc;
    use core::f64::consts::PI;
    #[cfg(feature = "text")]
    use glifo::Glyph;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::Color;
    #[cfg(feature = "text")]
    use vello_common::peniko::{Blob, FontData};

    // These tests serve the purpose of ensuring that the logic for selecting fast paths
    // works correctly.

    fn unconstrained() -> Scene {
        Scene::new(200, 200)
    }

    fn small_rect() -> Rect {
        Rect::new(10.0, 10.0, 50.0, 50.0)
    }

    fn triangle_path() -> BezPath {
        let mut path = BezPath::new();
        path.move_to((10.0, 10.0));
        path.line_to((90.0, 50.0));
        path.line_to((10.0, 90.0));
        path.close_path();
        path
    }

    fn is_rect(cmd: &FastStripCommand) -> bool {
        matches!(cmd, FastStripCommand::Rect(_))
    }

    fn is_path(cmd: &FastStripCommand) -> bool {
        matches!(cmd, FastStripCommand::Path(_))
    }

    #[test]
    fn fast_only_single_rect() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.fill_rect(&small_rect());

        assert_eq!(scene.strip_path_mode, StripPathMode::FastOnly);
        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_rect(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn fast_only_single_path() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.fill_path(&triangle_path());

        assert_eq!(scene.strip_path_mode, StripPathMode::FastOnly);
        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_path(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn fast_only_mixed_commands() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.fill_rect(&small_rect());
        scene.fill_path(&triangle_path());
        scene.fill_rect(&Rect::new(60.0, 60.0, 90.0, 90.0));

        assert_eq!(scene.strip_path_mode, StripPathMode::FastOnly);
        let cmds = &scene.fast_strips_buffer.commands;
        assert_eq!(cmds.len(), 3);
        assert!(is_rect(&cmds[0]));
        assert!(is_path(&cmds[1]));
        assert!(is_rect(&cmds[2]));
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

        let mut scene = unconstrained();
        let mut resources = Resources::new();

        scene.fill_rect(&small_rect());
        scene.fill_path(&triangle_path());
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

    #[test]
    fn fast_only_stroke_is_path() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.set_stroke(Stroke::new(2.0));
        scene.stroke_rect(&small_rect());

        assert_eq!(scene.strip_path_mode, StripPathMode::FastOnly);
        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_path(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn rect_rejected_by_skew_transform() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.set_transform(Affine::new([1.0, 0.5, 0.0, 1.0, 0.0, 0.0]));
        scene.fill_rect(&small_rect());

        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_path(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn rect_rejected_by_rotation() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.set_transform(Affine::rotate_about(
            45.0 * PI / 180.0,
            Point::new(30.0, 30.0),
        ));
        scene.fill_rect(&small_rect());

        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_path(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn rect_accepted_with_translation() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.set_transform(Affine::translate((5.0, 5.0)));
        scene.fill_rect(&small_rect());

        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_rect(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn rect_accepted_with_scale() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.set_transform(Affine::scale(2.0));
        scene.fill_rect(&Rect::new(5.0, 5.0, 20.0, 20.0));

        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_rect(&scene.fast_strips_buffer.commands[0]));
    }

    #[test]
    fn rect_rejected_by_clip_path() {
        let mut scene = unconstrained();
        scene.set_paint(Color::from_rgba8(255, 0, 0, 255));
        scene.push_clip_path(&triangle_path());
        scene.fill_rect(&small_rect());

        assert_eq!(scene.fast_strips_buffer.commands.len(), 1);
        assert!(is_path(&scene.fast_strips_buffer.commands[0]));
    }
}
