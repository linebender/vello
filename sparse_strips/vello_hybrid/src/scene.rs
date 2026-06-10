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
use vello_common::clip::ClipContext;
use vello_common::encode::{EncodeExt, EncodedExternalTexture, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Paint, PaintType, Tint};
#[cfg(feature = "text")]
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Extend, Fill, ImageQuality, ImageSampler};
use vello_common::render_state::RenderState;
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::tile::Tile;
use vello_common::util::{RectExt, control_point_bbox_u16, is_axis_aligned};

/// Default tolerance for curve flattening.
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// Identifier for a recorded root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RootId(usize);

impl RootId {
    #[inline]
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

/// Identifier for a recorded layer.
#[allow(
    dead_code,
    reason = "Opacity layer metadata is currently consumed by the GPU backends."
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordedLayerId(usize);

impl RecordedLayerId {
    #[inline]
    #[allow(
        dead_code,
        reason = "Opacity layer metadata is currently consumed by the GPU backends."
    )]
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

/// Metadata for a single path stored in the fast strips buffer.
#[derive(Clone, Debug)]
pub(crate) struct FastStripsPath {
    /// The range of strips for this path in the `strips` buffer.
    pub(crate) strips: Range<usize>,
    /// Coarse bounds of the path in viewport coordinates.
    pub(crate) bbox: RectU16,
    /// The paint of the path.
    pub(crate) paint: Paint,
}

/// A rectangle stored in the fast-path buffer.
#[derive(Clone, Debug)]
pub(crate) struct FastPathRect {
    pub(crate) x0: f32,
    pub(crate) y0: f32,
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) paint: Paint,
}

/// A command that can be lowered directly into GPU strips.
#[derive(Clone, Debug)]
pub(crate) enum FastStripCommand {
    /// A path rendered via the normal strip pipeline.
    Path(FastStripsPath),
    /// A rectangle.
    Rect(FastPathRect),
}

/// A command in a recorded root.
#[allow(
    dead_code,
    reason = "Layer commands are materialized by the GPU backends before direct rendering."
)]
#[derive(Clone, Debug)]
pub(crate) enum RecordedCommand {
    /// A drawable command.
    Draw(FastStripCommand),
    /// A previously recorded layer sampled back into the current root.
    Layer(RecordedLayerId),
}

/// A recorded root command stream.
#[derive(Debug, Default)]
pub(crate) struct RecordedRoot {
    /// Commands recorded for this root.
    pub(crate) commands: Vec<RecordedCommand>,
}

impl RecordedRoot {
    pub(crate) fn direct_commands_without_layers(&self) -> Vec<FastStripCommand> {
        self.commands
            .iter()
            .map(|command| match command {
                RecordedCommand::Draw(command) => command.clone(),
                RecordedCommand::Layer(_) => {
                    panic!("recorded root must not contain layers in this direct rendering path")
                }
            })
            .collect()
    }
}

/// A recorded layer.
#[allow(
    dead_code,
    reason = "Layer metadata is currently consumed by the GPU backends."
)]
#[derive(Debug)]
pub(crate) struct RecordedLayer {
    /// Root containing this layer's commands.
    pub(crate) root_id: RootId,
    /// Nesting depth. The root has depth 0; direct child layers have depth 1.
    pub(crate) depth: usize,
    /// Blend mode used when compositing the layer into its parent.
    pub(crate) blend_mode: BlendMode,
    /// Opacity applied when compositing the layer into its parent.
    pub(crate) opacity: f32,
    /// Clip path applied when compositing the layer into its parent.
    pub(crate) clip: Option<LayerClip>,
    /// Filter applied to this layer, if any.
    pub(crate) filter: Option<FilterData>,
    /// Placement metadata for sampling a filtered layer back into its parent.
    pub(crate) filter_placement: FilterLayerPlacement,
    /// Bounds of the rendered layer contents in its parent root coordinate space.
    pub(crate) bbox: RectU16,
    /// Bounds affected when the layer is composited into its parent.
    pub(crate) output_bbox: RectU16,
}

/// A clip path associated with a recorded layer.
#[derive(Debug)]
pub(crate) struct LayerClip {
    /// Strip range for the clip path.
    pub(crate) strips: Range<usize>,
    /// Coarse clip path bounds in viewport coordinates.
    pub(crate) bbox: RectU16,
}

#[derive(Debug)]
struct LayerStackEntry {
    layer_id: RecordedLayerId,
    parent_root_id: RootId,
    previous_root_transform: Affine,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FilterLayerPlacement {
    /// Bounds of the pixmap that receives the unfiltered source and final filtered output.
    pub(crate) pixmap_bbox: RectU16,
    /// Bounds in the parent layer affected by compositing this filtered output.
    pub(crate) composite_bbox: RectU16,
    /// Source x offset used when sampling from the filtered pixmap.
    pub(crate) src_x: u16,
    /// Source y offset used when sampling from the filtered pixmap.
    pub(crate) src_y: u16,
}

impl FilterLayerPlacement {
    pub(crate) const EMPTY: Self = Self {
        pixmap_bbox: RectU16::INVERTED,
        composite_bbox: RectU16::INVERTED,
        src_x: 0,
        src_y: 0,
    };

    fn new(bbox: RectU16, filter_data: &FilterData) -> Self {
        let pixmap_bbox = bbox
            .expand(filter_data.filter_padding)
            .snap_to_tile_coordinates();
        let (shift_x, shift_y) = filter_data.source_shift();
        let src_x = shift_x.saturating_sub(pixmap_bbox.x0);
        let src_y = shift_y.saturating_sub(pixmap_bbox.y0);
        let composite_bbox = pixmap_bbox.relative_to_origin((shift_x, shift_y));

        Self {
            pixmap_bbox,
            composite_bbox,
            src_x,
            src_y,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FilterData {
    pub(crate) filter: Filter,
    /// The transform that was active when this filter layer was pushed.
    pub(crate) transform: Affine,
    /// Padding required by the filtered output.
    pub(crate) filter_padding: RectU16,
    /// Padding required by the filter source input.
    pub(crate) source_padding: RectU16,
}

impl FilterData {
    fn new(filter: Filter, transform: Affine) -> Self {
        fn expansion_padding(expansion: Rect) -> RectU16 {
            let expansion = expansion.snap_to_tile_coordinates();

            RectU16::new(
                (-expansion.x0) as u16,
                (-expansion.y0) as u16,
                expansion.x1 as u16,
                expansion.y1 as u16,
            )
        }

        let source_padding = expansion_padding(filter.source_expansion(&transform));
        let filter_padding = expansion_padding(filter.filter_expansion(&transform));

        Self {
            filter,
            transform,
            filter_padding,
            source_padding,
        }
    }

    pub(crate) fn source_shift(&self) -> (u16, u16) {
        (self.source_padding.x0, self.source_padding.y0)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Rectangle bounds are clamped to u16 before casting."
)]
fn fast_rect_bbox(rect: &FastPathRect) -> RectU16 {
    let x0 = rect.x0.floor().clamp(0.0, f32::from(u16::MAX)) as u16;
    let y0 = rect.y0.floor().clamp(0.0, f32::from(u16::MAX)) as u16;
    let x1 = rect.x1.ceil().clamp(0.0, f32::from(u16::MAX)) as u16;
    let y1 = rect.y1.ceil().clamp(0.0, f32::from(u16::MAX)) as u16;
    RectU16::new(x0, y0, x1, y1)
}

fn strip_bbox(strips: &[Strip], viewport_width: u16) -> RectU16 {
    let mut bbox = RectU16::INVERTED;
    if strips.len() < 2 {
        return bbox;
    }

    for pair in strips.windows(2) {
        let strip = pair[0];
        let next_strip = pair[1];
        if strip.is_sentinel() {
            continue;
        }

        let strip_y = strip.strip_y();
        let row_y = strip_y.saturating_mul(Tile::HEIGHT);
        let row_y1 = row_y.saturating_add(Tile::HEIGHT);
        let strip_width = strip.width_to(&next_strip);
        let strip_x1 = strip.x.saturating_add(strip_width);

        if strip_width > 0 {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() {
            let fill_x1 = if next_strip.is_sentinel() {
                viewport_width
            } else {
                next_strip.x
            };
            if strip_x1 < fill_x1 {
                bbox.union(RectU16::new(strip_x1, row_y, fill_x1, row_y1));
            }
        }
    }

    bbox
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The configuration for the texture atlas.
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
#[derive(Debug)]
pub struct Scene {
    /// Width of the rendering surface in pixels.
    pub(crate) width: u16,
    /// Height of the rendering surface in pixels.
    pub(crate) height: u16,
    clip_context: ClipContext,
    pub(crate) render_state: RenderState,
    root_transform: Affine,
    level: Level,
    pub(crate) aliasing_threshold: Option<u8>,
    /// Storage for encoded non-solid paint data.
    pub(crate) encoded_paints: RefCell<Vec<EncodedPaint>>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Generator for converting paths to strips.
    pub(crate) strip_generator: StripGenerator,
    strip_generator_stack: Vec<StripGenerator>,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Current filter effect applied to individual draw operations.
    filter: Option<Filter>,
    /// Recorded roots. Root `0` is the final scene root.
    pub(crate) roots: Vec<RecordedRoot>,
    /// Recorded layers.
    pub(crate) layers: Vec<RecordedLayer>,
    active_root_id: RootId,
    layer_stack: Vec<LayerStackEntry>,
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
            clip_context: ClipContext::new(),
            render_state: RenderState::default(),
            root_transform: Affine::IDENTITY,
            level: settings.level,
            aliasing_threshold: None,
            encoded_paints: RefCell::new(vec![]),
            paint_visible: true,
            strip_generator: StripGenerator::new(width, height, settings.level),
            strip_generator_stack: Vec::new(),
            strip_storage: RefCell::new(StripStorage::new(GenerationMode::Append)),
            filter: None,
            roots: vec![RecordedRoot::default()],
            layers: Vec::new(),
            active_root_id: RootId(0),
            layer_stack: Vec::new(),
        }
    }

    /// The final scene root.
    #[inline]
    pub(crate) fn root_id(&self) -> RootId {
        RootId(0)
    }

    /// Get a recorded root.
    #[inline]
    pub(crate) fn root(&self, root_id: RootId) -> &RecordedRoot {
        &self.roots[root_id.as_usize()]
    }

    /// Get all recorded layers.
    #[inline]
    #[allow(
        dead_code,
        reason = "Opacity layer metadata is currently consumed by the GPU backends."
    )]
    pub(crate) fn layers(&self) -> &[RecordedLayer] {
        &self.layers
    }

    #[inline]
    fn active_commands(&mut self) -> &mut Vec<RecordedCommand> {
        &mut self.roots[self.active_root_id.as_usize()].commands
    }

    fn record_draw_command(&mut self, command: FastStripCommand) {
        self.active_commands().push(RecordedCommand::Draw(command));
    }

    #[inline]
    fn effective_transform(&self) -> Affine {
        self.root_transform * self.render_state.transform
    }

    fn push_filter_viewport(&mut self, padding: RectU16) {
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
    }

    fn pop_filter_viewport(&mut self) {
        self.strip_generator = self
            .strip_generator_stack
            .pop()
            .expect("filter viewport stack underflowed");
    }

    /// Encode the current paint into a `Paint` that can be used for rendering.
    fn encode_current_paint(&mut self) -> Paint {
        match self.render_state.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => g.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.effective_transform() * self.render_state.paint_transform,
                None,
            ),
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.effective_transform() * self.render_state.paint_transform,
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
                ctx.effective_transform(),
                ctx.render_state.fill_rule,
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let strips = {
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
            strip_start..strip_storage.strips.len()
        };

        let bbox = strip_bbox(
            &self.strip_storage.borrow().strips[strips.clone()],
            self.strip_generator.width(),
        );
        self.record_draw_command(FastStripCommand::Path(FastStripsPath {
            strips,
            bbox,
            paint,
        }));
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        let transform = self.effective_transform();
        self.clip_context.push_clip(
            path.iter(),
            &mut self.strip_generator,
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
                ctx.effective_transform(),
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let strips = {
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
            strip_start..strip_storage.strips.len()
        };

        let bbox = strip_bbox(
            &self.strip_storage.borrow().strips[strips.clone()],
            self.strip_generator.width(),
        );
        self.record_draw_command(FastStripCommand::Path(FastStripsPath {
            strips,
            bbox,
            paint,
        }));
    }

    /// Set the aliasing threshold.
    pub fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.aliasing_threshold = aliasing_threshold;
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        if !self.paint_visible {
            return;
        }

        self.with_optional_filter(|ctx| {
            if ctx.try_fast_rect(rect) {
                return;
            }

            let transform = ctx.effective_transform();
            if is_axis_aligned(&transform) && ctx.aliasing_threshold.is_none() {
                let paint = ctx.encode_current_paint();
                let transformed_rect = transform.transform_rect_bbox(*rect);
                let strips = {
                    let strip_storage = &mut ctx.strip_storage.borrow_mut();
                    let strip_start = strip_storage.strips.len();
                    ctx.strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        ctx.clip_context.get(),
                    );
                    strip_start..strip_storage.strips.len()
                };

                let bbox = strip_bbox(
                    &ctx.strip_storage.borrow().strips[strips.clone()],
                    ctx.strip_generator.width(),
                );
                ctx.record_draw_command(FastStripCommand::Path(FastStripsPath {
                    strips,
                    bbox,
                    paint,
                }));
            } else {
                let paint = ctx.encode_current_paint();
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
    pub fn draw_texture_rects(
        &mut self,
        texture_id: TextureId,
        quality: ImageQuality,
        rects: impl IntoIterator<Item = SampleRect>,
    ) {
        self.with_optional_filter(|ctx| {
            let x_extend = Extend::Pad;
            let y_extend = Extend::Pad;

            for rect in rects {
                if rect.source_region.is_empty() {
                    continue;
                }

                let w = f64::from(rect.source_region.width());
                let h = f64::from(rect.source_region.height());
                let transform = ctx.effective_transform() * rect.transform;
                let paint = ctx.encode_external_texture_paint(
                    texture_id,
                    rect.source_region,
                    quality,
                    x_extend,
                    y_extend,
                    transform,
                );
                let dst_rect = Rect::new(0., 0., w, h);

                if is_axis_aligned(&transform)
                    && ctx.aliasing_threshold.is_none()
                    && ctx.clip_context.get().is_none()
                {
                    let transformed_rect = transform.transform_rect_bbox(dst_rect);
                    let x0 = transformed_rect.x0.max(0.0).min(f64::from(ctx.width));
                    let y0 = transformed_rect.y0.max(0.0).min(f64::from(ctx.height));
                    let x1 = transformed_rect.x1.max(0.0).min(f64::from(ctx.width));
                    let y1 = transformed_rect.y1.max(0.0).min(f64::from(ctx.height));

                    // Skip mirrored or zero-sized rectangles.
                    if x1 <= x0 || y1 <= y0 {
                        continue;
                    }

                    ctx.push_fast_rect(Rect::new(x0, y0, x1, y1), paint);
                    continue;
                }

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

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 to f32 truncation is acceptable for pixel coordinates"
    )]
    fn push_fast_rect(&mut self, bounds: Rect, paint: Paint) {
        self.record_draw_command(FastStripCommand::Rect(FastPathRect {
            x0: bounds.x0 as f32,
            y0: bounds.y0 as f32,
            x1: bounds.x1 as f32,
            y1: bounds.y1 as f32,
            paint,
        }));
    }

    fn fast_rect_bounds(&self, rect: &Rect) -> Option<Rect> {
        if self.aliasing_threshold.is_some() || self.clip_context.get().is_some() {
            return None;
        }

        // We can't handle skewed rectangles.
        // TODO: Maybe support rotated rectangles (https://github.com/linebender/vello/pull/1482#discussion_r2881223621)
        let transform = self.effective_transform();
        if !is_axis_aligned(&transform) {
            return None;
        }

        let transformed_rect = transform.transform_rect_bbox(*rect);
        let bounds = if self.active_root_id == self.root_id() {
            Rect::new(
                transformed_rect.x0.max(0.0).min(f64::from(self.width)),
                transformed_rect.y0.max(0.0).min(f64::from(self.height)),
                transformed_rect.x1.max(0.0).min(f64::from(self.width)),
                transformed_rect.y1.max(0.0).min(f64::from(self.height)),
            )
        } else if transformed_rect.x0 < 0.0 || transformed_rect.y0 < 0.0 {
            return None;
        } else {
            transformed_rect
        };

        // Can't handle mirrored or zero-sized rectangles.
        if bounds.x1 <= bounds.x0 || bounds.y1 <= bounds.y0 {
            return None;
        }

        Some(bounds)
    }

    fn root_bbox(&self, root: &RecordedRoot) -> RectU16 {
        let mut bbox = RectU16::INVERTED;
        for command in &root.commands {
            let command_bbox = match command {
                RecordedCommand::Draw(FastStripCommand::Path(path)) => path.bbox,
                RecordedCommand::Draw(FastStripCommand::Rect(rect)) => fast_rect_bbox(rect),
                RecordedCommand::Layer(layer_id) => self.layers[layer_id.as_usize()].output_bbox,
            };
            bbox.union(command_bbox);
        }
        bbox
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Fill a blurred rectangle with the given corner radius and standard deviation.
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
            let transform = ctx.effective_transform() * ctx.render_state.paint_transform;
            let paint =
                blurred_rect.encode_into(&mut ctx.encoded_paints.borrow_mut(), transform, None);

            if let Some(bounds) = ctx.fast_rect_bounds(&inflated_rect) {
                ctx.push_fast_rect(bounds, paint);
                return;
            }

            let path_transform = ctx.effective_transform();
            if is_axis_aligned(&path_transform) && ctx.aliasing_threshold.is_none() {
                let transformed_rect = path_transform.transform_rect_bbox(inflated_rect);
                let strips = {
                    let strip_storage = &mut ctx.strip_storage.borrow_mut();
                    let strip_start = strip_storage.strips.len();
                    ctx.strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        ctx.clip_context.get(),
                    );
                    strip_start..strip_storage.strips.len()
                };

                let bbox = strip_bbox(
                    &ctx.strip_storage.borrow().strips[strips.clone()],
                    ctx.strip_generator.width(),
                );
                ctx.record_draw_command(FastStripCommand::Path(FastStripsPath {
                    strips,
                    bbox,
                    paint,
                }));
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
            panic!("vello_hybrid layer masks are temporarily unsupported");
        }
        let layer_transform = self.effective_transform();
        let filter_data = filter.map(|filter| FilterData::new(filter, layer_transform));
        if let Some(filter_data) = &filter_data {
            self.push_filter_viewport(filter_data.source_padding);
        }
        let clip = clip_path.map(|clip_path| {
            let existing_clip = self.clip_context.get();
            let mut bbox = control_point_bbox_u16(clip_path, layer_transform);
            if let Some(existing_clip) = existing_clip {
                bbox = bbox.intersect(existing_clip.bbox);
            } else {
                bbox.x1 = bbox.x1.min(self.width);
                bbox.y1 = bbox.y1.min(self.height);
            }

            let strips = {
                let strip_storage = &mut self.strip_storage.borrow_mut();
                let strip_start = strip_storage.strips.len();
                self.strip_generator.generate_filled_path(
                    clip_path,
                    self.render_state.fill_rule,
                    layer_transform,
                    self.aliasing_threshold,
                    strip_storage,
                    existing_clip,
                );
                strip_start..strip_storage.strips.len()
            };
            LayerClip { strips, bbox }
        });

        let blend_mode = blend_mode.unwrap_or_default();
        let opacity = opacity.unwrap_or(1.0);
        let parent_root_id = self.active_root_id;
        let root_id = RootId(self.roots.len());
        self.roots.push(RecordedRoot::default());
        let layer_id = RecordedLayerId(self.layers.len());
        let depth = self.layer_stack.len() + 1;
        self.layers.push(RecordedLayer {
            root_id,
            depth,
            blend_mode,
            opacity,
            clip,
            filter: filter_data.clone(),
            filter_placement: FilterLayerPlacement::EMPTY,
            bbox: RectU16::INVERTED,
            output_bbox: RectU16::INVERTED,
        });
        self.layer_stack.push(LayerStackEntry {
            layer_id,
            parent_root_id,
            previous_root_transform: self.root_transform,
        });
        if let Some(filter_data) = &filter_data {
            let (shift_x, shift_y) = filter_data.source_shift();
            self.root_transform =
                Affine::translate((f64::from(shift_x), f64::from(shift_y))) * self.root_transform;
        }
        self.active_root_id = root_id;
    }

    /// Push a new clip layer.
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
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a new filter layer.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        let entry = self.layer_stack.pop().expect("layer stack underflowed");
        let root_id = self.layers[entry.layer_id.as_usize()].root_id;
        let content_bbox = self.root_bbox(self.root(root_id));
        let blend_mode = self.layers[entry.layer_id.as_usize()].blend_mode;
        let is_filter_layer = self.layers[entry.layer_id.as_usize()].filter.is_some();
        let is_empty_clear = content_bbox.is_empty() && blend_mode.compose == Compose::Clear;
        let is_non_default_blend = blend_mode != BlendMode::default();
        let is_destructive_blend = blend_mode.is_destructive();
        let mut bbox = content_bbox;
        let mut output_bbox = content_bbox;
        if let Some(filter_data) = &self.layers[entry.layer_id.as_usize()].filter {
            let placement = FilterLayerPlacement::new(content_bbox, filter_data);
            bbox = placement.pixmap_bbox;
            output_bbox = placement.composite_bbox;
            self.layers[entry.layer_id.as_usize()].filter_placement = placement;
        }
        if is_empty_clear {
            bbox = self.layers[entry.layer_id.as_usize()]
                .clip
                .as_ref()
                .map_or(RectU16::new(0, 0, self.width, self.height), |clip| {
                    clip.bbox
                });
            output_bbox = bbox;
        } else if is_non_default_blend {
            if is_destructive_blend {
                output_bbox = self.layers[entry.layer_id.as_usize()]
                    .clip
                    .as_ref()
                    .map_or(RectU16::new(0, 0, self.width, self.height), |clip| {
                        clip.bbox
                    });
                if bbox.is_empty() {
                    bbox = output_bbox;
                }
            } else if let Some(clip) = &self.layers[entry.layer_id.as_usize()].clip {
                output_bbox = output_bbox.intersect(clip.bbox);
            }
            if let Some(clip) = &self.layers[entry.layer_id.as_usize()].clip {
                if !is_filter_layer {
                    bbox.union(clip.bbox);
                }
            }
        } else if let Some(clip) = &self.layers[entry.layer_id.as_usize()].clip {
            output_bbox = output_bbox.intersect(clip.bbox);
            if !is_filter_layer {
                bbox.union(clip.bbox);
            }
        }
        self.layers[entry.layer_id.as_usize()].bbox = bbox;
        self.layers[entry.layer_id.as_usize()].output_bbox = output_bbox;
        if is_filter_layer {
            self.pop_filter_viewport();
        }
        self.root_transform = entry.previous_root_transform;
        self.active_root_id = entry.parent_root_id;
        self.active_commands()
            .push(RecordedCommand::Layer(entry.layer_id));
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        if blend_mode != BlendMode::default() {
            panic!(
                "vello_hybrid non-default draw blend modes are temporarily unsupported; use blend layers instead"
            );
        }
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
            let saved_filter = self.filter.take();
            self.push_filter_layer(filter);
            f(self);
            self.pop_layer();
            self.filter = saved_filter;
        } else {
            f(self);
        }
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.strip_generator.reset();
        self.strip_generator_stack.clear();
        {
            let mut ss = self.strip_storage.borrow_mut();
            ss.clear();
            ss.set_generation_mode(GenerationMode::Append);
        }
        self.encoded_paints.borrow_mut().clear();
        self.clip_context.reset();
        self.render_state.reset();
        self.root_transform = Affine::IDENTITY;
        self.roots.clear();
        self.roots.push(RecordedRoot::default());
        self.layers.clear();
        self.active_root_id = RootId(0);
        self.layer_stack.clear();
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
        core::mem::take(&mut self.render_state)
    }

    /// Save a copy of the current rendering state.
    pub fn save_current_state(&mut self) -> RenderState {
        self.render_state.clone()
    }

    /// Restore a previously saved rendering state.
    pub fn restore_state(&mut self, state: RenderState) {
        self.render_state = state;
        self.set_paint_visible();
    }
}
