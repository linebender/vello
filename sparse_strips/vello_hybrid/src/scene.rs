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
use vello_common::peniko::{BlendMode, Extend, Fill, ImageQuality, ImageSampler};
use vello_common::render_state::RenderState;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::util::{control_point_bbox_u16, is_axis_aligned};

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
    pub(crate) aliasing_threshold: Option<u8>,
    /// Storage for encoded non-solid paint data.
    pub(crate) encoded_paints: RefCell<Vec<EncodedPaint>>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Generator for converting paths to strips.
    pub(crate) strip_generator: StripGenerator,
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
            aliasing_threshold: None,
            encoded_paints: RefCell::new(vec![]),
            paint_visible: true,
            strip_generator: StripGenerator::new(width, height, settings.level),
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

    /// Encode the current paint into a `Paint` that can be used for rendering.
    fn encode_current_paint(&mut self) -> Paint {
        match self.render_state.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => g.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.render_state.transform * self.render_state.paint_transform,
                None,
            ),
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints.borrow_mut(),
                self.render_state.transform * self.render_state.paint_transform,
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

        self.assert_no_filter();
        let paint = self.encode_current_paint();
        self.fill_path_with(
            path,
            self.render_state.transform,
            self.render_state.fill_rule,
            paint,
            self.aliasing_threshold,
        );
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

        self.record_draw_command(FastStripCommand::Path(FastStripsPath { strips, paint }));
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

        self.assert_no_filter();
        let paint = self.encode_current_paint();
        self.stroke_path_with(
            path,
            self.render_state.transform,
            paint,
            self.aliasing_threshold,
        );
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

        self.record_draw_command(FastStripCommand::Path(FastStripsPath { strips, paint }));
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

        self.assert_no_filter();
        if self.try_fast_rect(rect) {
            return;
        }

        if is_axis_aligned(&self.render_state.transform) && self.aliasing_threshold.is_none() {
            let paint = self.encode_current_paint();
            let transformed_rect = self.render_state.transform.transform_rect_bbox(*rect);
            let strips = {
                let strip_storage = &mut self.strip_storage.borrow_mut();
                let strip_start = strip_storage.strips.len();
                self.strip_generator.generate_filled_rect_fast(
                    &transformed_rect,
                    strip_storage,
                    self.clip_context.get(),
                );
                strip_start..strip_storage.strips.len()
            };

            self.record_draw_command(FastStripCommand::Path(FastStripsPath { strips, paint }));
        } else {
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
    pub fn draw_texture_rects(
        &mut self,
        texture_id: TextureId,
        quality: ImageQuality,
        rects: impl IntoIterator<Item = SampleRect>,
    ) {
        self.assert_no_filter();

        let x_extend = Extend::Pad;
        let y_extend = Extend::Pad;

        for rect in rects {
            if rect.source_region.is_empty() {
                continue;
            }

            let w = f64::from(rect.source_region.width());
            let h = f64::from(rect.source_region.height());
            let transform = self.render_state.transform * rect.transform;
            let paint = self.encode_external_texture_paint(
                texture_id,
                rect.source_region,
                quality,
                x_extend,
                y_extend,
                transform,
            );
            let dst_rect = Rect::new(0., 0., w, h);

            if is_axis_aligned(&transform)
                && self.aliasing_threshold.is_none()
                && self.clip_context.get().is_none()
            {
                let transformed_rect = transform.transform_rect_bbox(dst_rect);
                let x0 = transformed_rect.x0.max(0.0).min(f64::from(self.width));
                let y0 = transformed_rect.y0.max(0.0).min(f64::from(self.height));
                let x1 = transformed_rect.x1.max(0.0).min(f64::from(self.width));
                let y1 = transformed_rect.y1.max(0.0).min(f64::from(self.height));

                // Skip mirrored or zero-sized rectangles.
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }

                self.push_fast_rect(Rect::new(x0, y0, x1, y1), paint);
                continue;
            }

            self.fill_path_with(
                &dst_rect.to_path(DEFAULT_TOLERANCE),
                transform,
                self.render_state.fill_rule,
                paint,
                self.aliasing_threshold,
            );
        }
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
        if !is_axis_aligned(&self.render_state.transform) {
            return None;
        }

        let transformed_rect = self.render_state.transform.transform_rect_bbox(*rect);

        let x0 = transformed_rect.x0.max(0.0).min(f64::from(self.width));
        let y0 = transformed_rect.y0.max(0.0).min(f64::from(self.height));
        let x1 = transformed_rect.x1.max(0.0).min(f64::from(self.width));
        let y1 = transformed_rect.y1.max(0.0).min(f64::from(self.height));

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
    pub fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        if !self.paint_visible {
            return;
        }

        self.assert_no_filter();
        let rect = rect.abs();
        let color = match self.render_state.paint {
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
        let transform = self.render_state.transform * self.render_state.paint_transform;
        let paint =
            blurred_rect.encode_into(&mut self.encoded_paints.borrow_mut(), transform, None);

        if let Some(bounds) = self.fast_rect_bounds(&inflated_rect) {
            self.push_fast_rect(bounds, paint);
            return;
        }

        if is_axis_aligned(&self.render_state.transform) && self.aliasing_threshold.is_none() {
            let transformed_rect = self
                .render_state
                .transform
                .transform_rect_bbox(inflated_rect);
            let strips = {
                let strip_storage = &mut self.strip_storage.borrow_mut();
                let strip_start = strip_storage.strips.len();
                self.strip_generator.generate_filled_rect_fast(
                    &transformed_rect,
                    strip_storage,
                    self.clip_context.get(),
                );
                strip_start..strip_storage.strips.len()
            };

            self.record_draw_command(FastStripCommand::Path(FastStripsPath { strips, paint }));
        } else {
            self.fill_path_with(
                &inflated_rect.to_path(DEFAULT_TOLERANCE),
                self.render_state.transform,
                Fill::NonZero,
                paint,
                self.aliasing_threshold,
            );
        }
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
        if filter.is_some() {
            panic!("vello_hybrid filter layers are temporarily unsupported");
        }

        let clip = clip_path.map(|clip_path| {
            let existing_clip = self.clip_context.get();
            let mut bbox = control_point_bbox_u16(clip_path, self.render_state.transform);
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
                    self.render_state.transform,
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
        });
        self.layer_stack.push(LayerStackEntry {
            layer_id,
            parent_root_id,
        });
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

    fn assert_no_filter(&self) {
        if self.filter.is_some() {
            panic!("vello_hybrid draw filters are temporarily unsupported");
        }
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.strip_generator.reset();
        {
            let mut ss = self.strip_storage.borrow_mut();
            ss.clear();
            ss.set_generation_mode(GenerationMode::Append);
        }
        self.encoded_paints.borrow_mut().clear();
        self.clip_context.reset();
        self.render_state.reset();
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
