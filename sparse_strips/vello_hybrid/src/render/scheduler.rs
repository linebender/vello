// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared layer scheduling for GPU render backends.

use crate::{
    RenderError, RenderSize,
    direct::RenderOrigin,
    scene::{
        FastPathRect, FastStripCommand, FastStripsPath, RecordedCommand, RecordedLayer,
        RecordedLayerId, Scene,
    },
};
use alloc::{vec, vec::Vec};
use vello_common::{
    TextureId,
    encode::{EncodedExternalTexture, EncodedPaint},
    geometry::RectU16,
    kurbo::Affine,
    paint::{IndexedPaint, Paint, Tint, TintMode},
    peniko::color::palette::css::WHITE,
    peniko::{self, Compose},
};

/// Pseudo layer index used when the scene root must be rendered offscreen first.
pub(crate) const ROOT_LAYER_IDX: usize = usize::MAX;

/// A backend-owned render target for a scheduled layer.
pub(crate) trait ScheduledLayerTarget {
    /// Synthetic texture ID used when the layer is sampled by its parent.
    fn texture_id(&self) -> TextureId;

    /// Width of the target in pixels.
    fn width(&self) -> u16;

    /// Height of the target in pixels.
    fn height(&self) -> u16;
}

/// Backend hooks used by [`LayerScheduler`].
pub(crate) trait LayerScheduleRenderer<T: ScheduledLayerTarget> {
    /// Allocate the render target for a layer.
    fn create_layer_target(&mut self, layer_idx: usize, texture_id: TextureId) -> T;

    /// Render a non-root layer.
    fn render_layer(
        &mut self,
        layer_idx: usize,
        layer: &RecordedLayer,
        target: T,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<T>],
    ) -> Result<T, RenderError>;

    /// Render the scene root after all descendant layers have been rendered.
    fn render_root(
        &mut self,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<T>],
    ) -> Result<(), RenderError>;
}

/// Scheduled work for a recorded root, preserving layer ordering boundaries.
#[derive(Debug)]
pub(crate) enum ScheduledLayerOp {
    /// Draw commands that can be rendered directly in one pass.
    Draw(Vec<FastStripCommand>),
    /// Composite a previously rendered child layer into the current target.
    CompositeLayer(RecordedLayerId),
}

/// Draw commands materialized from scheduled batches.
pub(crate) enum DrawCommands<'a> {
    /// The scheduler already has a contiguous draw batch.
    Borrowed(&'a [FastStripCommand]),
    /// Multiple draw batches had to be joined.
    Owned(Vec<FastStripCommand>),
}

impl<'a> DrawCommands<'a> {
    /// Return the commands as a slice.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[FastStripCommand] {
        match self {
            Self::Borrowed(commands) => commands,
            Self::Owned(commands) => commands,
        }
    }
}

pub(crate) fn flatten_draw_batches(batches: &[ScheduledLayerOp]) -> DrawCommands<'_> {
    debug_assert!(
        batches
            .iter()
            .all(|batch| matches!(batch, ScheduledLayerOp::Draw(_))),
        "layer composites require offscreen batch execution"
    );

    match batches {
        [] => DrawCommands::Borrowed(&[]),
        [ScheduledLayerOp::Draw(commands)] => DrawCommands::Borrowed(commands),
        _ => {
            let mut commands = Vec::new();
            for batch in batches {
                if let ScheduledLayerOp::Draw(batch_commands) = batch {
                    commands.extend(batch_commands.iter().cloned());
                }
            }
            DrawCommands::Owned(commands)
        }
    }
}

/// Whether a root needs an offscreen layer target before being copied to the final target.
#[inline]
pub(crate) fn root_needs_offscreen_layer(batches: &[ScheduledLayerOp]) -> bool {
    batches
        .iter()
        .any(|batch| matches!(batch, ScheduledLayerOp::CompositeLayer(_)))
}

/// Command that samples an offscreen-rendered scene root into the final target.
pub(crate) fn root_sample_command(
    target: &impl ScheduledLayerTarget,
    root_render_size: &RenderSize,
    encoded_paints: &mut Vec<EncodedPaint>,
) -> FastStripCommand {
    let paint_idx = encoded_paints.len();
    encoded_paints.push(layer_encoded_paint(
        target,
        1.0,
        RectU16::new(0, 0, target.width(), target.height()),
        Affine::IDENTITY,
    ));

    FastStripCommand::Rect(FastPathRect {
        x0: 0.0,
        y0: 0.0,
        x1: root_render_size.width as f32,
        y1: root_render_size.height as f32,
        paint: Paint::Indexed(IndexedPaint::new(paint_idx)),
    })
}

/// Origin and dimensions for the target used to render a recorded layer.
pub(crate) fn layer_target_origin_and_size(
    scene: &Scene,
    layer_idx: usize,
    root_render_size: &RenderSize,
) -> (RenderOrigin, RenderSize) {
    if layer_idx == ROOT_LAYER_IDX {
        return (RenderOrigin::default(), root_render_size.clone());
    }

    let bbox = scene.layers()[layer_idx].bbox;
    (
        RenderOrigin {
            x: bbox.x0,
            y: bbox.y0,
        },
        RenderSize {
            width: u32::from(bbox.width()).max(1),
            height: u32::from(bbox.height()).max(1),
        },
    )
}

/// Whether a child layer can be sampled directly into the parent render target.
#[inline]
pub(crate) fn layer_can_be_sampled_directly(layer: &RecordedLayer) -> bool {
    layer.blend_mode == peniko::BlendMode::default()
}

/// Scissor rectangle used when compositing an isolated layer with the blend shader.
pub(crate) fn layer_blend_scissor(
    layer: &RecordedLayer,
    parent_origin: RenderOrigin,
) -> Option<RectU16> {
    layer.clip.as_ref().map(|_| {
        layer
            .output_bbox
            .relative_to_origin((parent_origin.x, parent_origin.y))
    })
}

/// Region to cover when sampling a rendered layer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LayerSampleExtent {
    /// Cover the region affected by compositing the layer into its parent.
    Output,
    /// Cover only the region containing rendered source pixels.
    Content,
}

/// Schedules recorded roots/layers and materializes draw-only command streams.
pub(crate) struct LayerScheduler<'a> {
    scene: &'a Scene,
    scene_paint_count: usize,
}

impl<'a> LayerScheduler<'a> {
    /// Create a scheduler for `scene`.
    pub(crate) fn new(scene: &'a Scene, scene_paint_count: usize) -> Self {
        Self {
            scene,
            scene_paint_count,
        }
    }

    /// Execute the current naive schedule: deepest layers first, then the root.
    pub(crate) fn render<T, R>(
        &self,
        encoded_paints: &mut Vec<EncodedPaint>,
        renderer: &mut R,
    ) -> Result<Vec<Option<T>>, RenderError>
    where
        T: ScheduledLayerTarget,
        R: LayerScheduleRenderer<T>,
    {
        let mut layer_targets: Vec<Option<T>> = self.scene.layers().iter().map(|_| None).collect();
        let visible_layers = self.visible_layers();
        let max_depth = self
            .scene
            .layers()
            .iter()
            .map(|layer| layer.depth)
            .max()
            .unwrap_or(0);

        for depth in (1..=max_depth).rev() {
            for (layer_idx, layer) in self.scene.layers().iter().enumerate() {
                if layer.depth != depth || !visible_layers[layer_idx] {
                    continue;
                }

                let target = renderer.create_layer_target(layer_idx, layer_texture_id(layer_idx));
                let batches = self.materialize_root_batches(
                    self.scene.root(layer.root_id),
                    &layer_targets,
                    &visible_layers,
                );
                let target = renderer.render_layer(
                    layer_idx,
                    layer,
                    target,
                    &batches,
                    encoded_paints,
                    &layer_targets,
                )?;
                encoded_paints.truncate(self.scene_paint_count);
                layer_targets[layer_idx] = Some(target);
            }
        }

        let batches = self.materialize_root_batches(
            self.scene.root(self.scene.root_id()),
            &layer_targets,
            &visible_layers,
        );
        let result = renderer.render_root(&batches, encoded_paints, &layer_targets);
        encoded_paints.truncate(self.scene_paint_count);
        result.map(|()| layer_targets)
    }

    fn visible_layers(&self) -> Vec<bool> {
        let mut visible = vec![false; self.scene.layers().len()];
        let root_viewport = RectU16::new(0, 0, self.scene.width, self.scene.height);
        self.mark_visible_layers(self.scene.root_id(), root_viewport, &mut visible);
        visible
    }

    fn mark_visible_layers(
        &self,
        root_id: crate::scene::RootId,
        viewport: RectU16,
        visible: &mut [bool],
    ) {
        for command in &self.scene.root(root_id).commands {
            let RecordedCommand::Layer(layer_id) = command else {
                continue;
            };

            let layer_idx = layer_id.as_usize();
            let layer = &self.scene.layers()[layer_idx];
            if layer_is_empty(layer) || layer.output_bbox.intersect(viewport).is_empty() {
                continue;
            }

            visible[layer_idx] = true;
            self.mark_visible_layers(layer.root_id, layer.bbox, visible);
        }
    }

    fn materialize_root_batches<T: ScheduledLayerTarget>(
        &self,
        root: &crate::scene::RecordedRoot,
        layer_targets: &[Option<T>],
        visible_layers: &[bool],
    ) -> Vec<ScheduledLayerOp> {
        let mut batches = Vec::new();
        let mut commands = Vec::with_capacity(root.commands.len());
        for command in &root.commands {
            match command {
                RecordedCommand::Layer(layer_id) => {
                    let layer = &self.scene.layers()[layer_id.as_usize()];
                    if !visible_layers[layer_id.as_usize()] || layer_is_empty(layer) {
                        continue;
                    }
                    assert!(
                        layer_targets[layer_id.as_usize()].is_some(),
                        "child layer must be rendered before its parent"
                    );

                    if !commands.is_empty() {
                        batches.push(ScheduledLayerOp::Draw(core::mem::take(&mut commands)));
                    }
                    batches.push(ScheduledLayerOp::CompositeLayer(*layer_id));
                }
                RecordedCommand::Draw(command) => {
                    commands.push(command.clone());
                }
            }
        }
        if !commands.is_empty() {
            batches.push(ScheduledLayerOp::Draw(commands));
        }
        batches
    }
}

#[inline]
fn layer_texture_id(layer_idx: usize) -> TextureId {
    TextureId(u64::MAX - layer_idx as u64)
}

#[inline]
pub(crate) fn root_texture_id() -> TextureId {
    TextureId(u64::MAX / 2)
}

#[inline]
pub(crate) fn filter_scratch_texture_id(layer_idx: usize, scratch_idx: usize) -> TextureId {
    TextureId((u64::MAX / 5).saturating_sub((layer_idx * 2 + scratch_idx) as u64))
}

#[inline]
pub(crate) fn backdrop_texture_id(parent_idx: usize, layer_idx: usize) -> TextureId {
    TextureId(
        (u64::MAX / 4)
            .wrapping_sub((parent_idx as u64).wrapping_mul(4096))
            .wrapping_sub(layer_idx as u64),
    )
}

#[inline]
pub(crate) fn aligned_layer_source_texture_id(layer_idx: usize) -> TextureId {
    TextureId(u64::MAX / 3 - layer_idx as u64)
}

pub(crate) fn layer_is_empty(layer: &RecordedLayer) -> bool {
    if layer.output_bbox.is_empty() {
        return true;
    }

    if layer.blend_mode.compose == Compose::Clear {
        return false;
    }

    layer
        .clip
        .as_ref()
        .is_some_and(|clip| clip.bbox.is_empty() || clip.strips.is_empty())
}

pub(crate) fn layer_sample_command(
    _scene: &Scene,
    layer: &RecordedLayer,
    target: &impl ScheduledLayerTarget,
    opacity: f32,
    extent: LayerSampleExtent,
    encoded_paints: &mut Vec<EncodedPaint>,
) -> Option<FastStripCommand> {
    if layer_is_empty(layer) {
        return None;
    }

    let paint_idx = encoded_paints.len();
    let sample_bbox = match extent {
        LayerSampleExtent::Output => layer.output_bbox,
        LayerSampleExtent::Content if layer.filter.is_some() => layer.output_bbox,
        LayerSampleExtent::Content => layer.bbox,
    };
    let source_region = RectU16::new(0, 0, target.width(), target.height());
    let transform = if layer.filter.is_some() {
        let placement = layer.filter_placement;
        Affine::translate((
            f64::from(placement.src_x) - f64::from(placement.composite_bbox.x0),
            f64::from(placement.src_y) - f64::from(placement.composite_bbox.y0),
        ))
    } else {
        Affine::translate((-(f64::from(layer.bbox.x0)), -(f64::from(layer.bbox.y0))))
    };
    encoded_paints.push(layer_encoded_paint(
        target,
        opacity,
        source_region,
        transform,
    ));
    let paint = Paint::Indexed(IndexedPaint::new(paint_idx));
    Some(
        if let Some(clip) = &layer.clip
            && matches!(extent, LayerSampleExtent::Output)
        {
            FastStripCommand::Path(FastStripsPath {
                strips: clip.strips.clone(),
                bbox: clip.bbox,
                paint,
            })
        } else {
            FastStripCommand::Rect(FastPathRect {
                x0: f32::from(sample_bbox.x0),
                y0: f32::from(sample_bbox.y0),
                x1: f32::from(sample_bbox.x1),
                y1: f32::from(sample_bbox.y1),
                paint,
            })
        },
    )
}

pub(crate) fn layer_encoded_paint(
    target: &impl ScheduledLayerTarget,
    opacity: f32,
    source_region: RectU16,
    transform: Affine,
) -> EncodedPaint {
    EncodedPaint::ExternalTexture(EncodedExternalTexture {
        texture_id: target.texture_id(),
        source_region,
        sampler: peniko::ImageSampler {
            x_extend: peniko::Extend::Pad,
            y_extend: peniko::Extend::Pad,
            quality: peniko::ImageQuality::Low,
            alpha: 1.0,
        },
        may_have_transparency: true,
        transform,
        tint: Some(Tint {
            color: WHITE.with_alpha(opacity),
            mode: TintMode::Multiply,
        }),
    })
}
