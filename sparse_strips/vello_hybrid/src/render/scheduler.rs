// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared layer scheduling for GPU render backends.

use crate::{
    RenderError,
    scene::{
        FastPathRect, FastStripCommand, FastStripsPath, RecordedCommand, RecordedLayer,
        RecordedLayerId, Scene,
    },
};
use alloc::vec::Vec;
use vello_common::{
    TextureId,
    encode::{EncodedExternalTexture, EncodedPaint},
    geometry::RectU16,
    kurbo::Affine,
    paint::{IndexedPaint, Paint, Tint, TintMode},
    peniko::color::palette::css::WHITE,
    peniko::{self, Compose},
};

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
        target: &T,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<T>],
    ) -> Result<(), RenderError>;

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
        let max_depth = self
            .scene
            .layers()
            .iter()
            .map(|layer| layer.depth)
            .max()
            .unwrap_or(0);

        for depth in (1..=max_depth).rev() {
            for (layer_idx, layer) in self.scene.layers().iter().enumerate() {
                if layer.depth != depth {
                    continue;
                }

                let target = renderer.create_layer_target(layer_idx, layer_texture_id(layer_idx));
                let batches =
                    self.materialize_root_batches(self.scene.root(layer.root_id), &layer_targets);
                renderer.render_layer(
                    layer_idx,
                    layer,
                    &target,
                    &batches,
                    encoded_paints,
                    &layer_targets,
                )?;
                encoded_paints.truncate(self.scene_paint_count);
                layer_targets[layer_idx] = Some(target);
            }
        }

        let batches =
            self.materialize_root_batches(self.scene.root(self.scene.root_id()), &layer_targets);
        let result = renderer.render_root(&batches, encoded_paints, &layer_targets);
        encoded_paints.truncate(self.scene_paint_count);
        result.map(|()| layer_targets)
    }

    fn materialize_root_batches<T: ScheduledLayerTarget>(
        &self,
        root: &crate::scene::RecordedRoot,
        layer_targets: &[Option<T>],
    ) -> Vec<ScheduledLayerOp> {
        let mut batches = Vec::new();
        let mut commands = Vec::with_capacity(root.commands.len());
        for command in &root.commands {
            match command {
                RecordedCommand::Layer(layer_id) => {
                    let layer = &self.scene.layers()[layer_id.as_usize()];
                    assert!(
                        layer_targets[layer_id.as_usize()].is_some(),
                        "child layer must be rendered before its parent"
                    );
                    if layer_is_empty(layer) {
                        continue;
                    }

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
    let bbox = layer.bbox;
    let sample_bbox = match extent {
        LayerSampleExtent::Output => layer.output_bbox,
        LayerSampleExtent::Content => layer.bbox,
    };
    let source_region = RectU16::new(0, 0, target.width(), target.height());
    let transform = Affine::translate((-(f64::from(bbox.x0)), -(f64::from(bbox.y0))));
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
            && !contains_rect(clip.bbox, bbox)
        {
            FastStripCommand::Path(FastStripsPath {
                strips: clip.strips.clone(),
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

#[inline]
fn contains_rect(outer: RectU16, inner: RectU16) -> bool {
    outer.x0 <= inner.x0 && outer.y0 <= inner.y0 && outer.x1 >= inner.x1 && outer.y1 >= inner.y1
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
