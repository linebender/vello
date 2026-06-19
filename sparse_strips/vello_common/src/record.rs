// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// TODO: Update the doc comment

//! Recording rendering commands.
//!
//! Currently, the pipeline of Vello CPU can be split into roughly three parts:
//! 1) Recording rendering commands by generating strips + layer metadata.
//! 2) Bucketing the recorded commands per-strip row into render commands (coarse rasterization).
//! 3) Executing the render commands (fine rasterization).
//!
//! Why do we need 1) instead of just doing 2) as soon as the user sends their commands? Well,
//! this is actually what an earlier version of Vello CPU did. The main reason why we _don't_ do
//! this anymore are **filter layers**. Unlike normal layers with blends/opacity/masks/clips,
//! filter layers are special in two ways:
//!
//! - They always need to be rendered separately (assuming spatial filters are used), independently
//!   of the parent layer command stream.
//!   This is because spatial filters might require sampling neighboring pixels, so the whole filter
//!   layer needs to be rendered as a whole before you can apply the filter and then composite it into
//!   the parent layer. Therefore, commands for filter layers cannot simply be inlined into the
//!   parent layer stream (this is what was done in previous versions of Vello CPU, and resulted in
//!   incredibly complex and fragile code).
//!
//! - Filter layers might have a different dimension than the main viewport. If you apply a big
//!   blur filter, you might have to render shapes that would normally be culled away because they
//!   exceed the viewport.
//!
//! If we decided to skip step 1) and do 2) directly, we would need to store and reuse instances
//! of [`vello_cpu::coarse::bucketer::CommandBucketer`] for the root and each filter layer, which is especially cumbersome because
//! conceptually, a command bucketer is a 2D array. This makes it very difficult to resize and reuse
//! it while having the ability to retain and reuse inner allocations. By first recording all commands
//! and their strips into a single draw buffer, we can basically represent the commands of a single
//! layer with just one flat command buffer plus batch ranges into that shared draw buffer, making it
//! much easier to reuse them across frames. It also allows us to collect useful metadata (e.g. the
//! bounding box of a layer) and make appropriate decisions about how to render them (for example,
//! where to place a filter layer and what size to allocate for it).

use crate::filter::{FilterData, FilterLayerPlacement};
use crate::geometry::RectU16;
use crate::mask::Mask;
use crate::peniko::BlendMode;
use crate::strip::Strip;
use crate::util::{RectExt, VecPool};
use alloc::vec::Vec;
use core::ops::Range;

/// A drawable payload that can report its affected bounds.
pub trait Drawable {
    /// Return the drawable's coarse bounds in the current command stream's coordinate space.
    fn bbox(&self, strips: &[Strip], viewport_width: u16) -> RectU16;
}

/// A recorded command.
#[derive(Debug)]
pub enum RecordedCmd {
    /// A contiguous batch of draw commands in [`CommandRecorder::draws`].
    Draws(Range<u32>),
    /// Composite a recorded layer into the current layer.
    Layer(u32),
}

/// Metadata and command stream for one recorded layer.
#[derive(Debug)]
pub struct RecordedLayer {
    /// Layer compositing properties.
    pub props: LayerProps,
    /// The render commands of the layer.
    pub cmds: Vec<RecordedCmd>,
    /// The kind of recorded layer.
    pub kind: RecordedLayerKind,
    /// Nesting depth. Direct child layers of the root have depth 1.
    pub depth: usize,
    /// Bounds containing the pixels rendered into this layer's target.
    pub bbox: RectU16,
}

/// Compositing properties for a recorded layer.
#[derive(Debug)]
pub struct LayerProps {
    /// Blend mode used when compositing the layer into its parent.
    pub blend_mode: BlendMode,
    /// Opacity applied when compositing the layer into its parent.
    pub opacity: f32,
    /// Optional mask applied when compositing the layer.
    pub mask: Option<Mask>,
    /// Optional clip path applied when compositing the layer.
    pub clip_path: Option<LayerClip>,
}

/// Additional metadata that differs between regular and filter layers.
#[derive(Debug)]
pub enum RecordedLayerKind {
    /// A regular layer.
    Regular,
    /// A filter layer.
    Filter {
        /// Static data about the filter itself.
        filter_data: FilterData,
        /// Data about how to place the filter layer, which can only be determined once its
        /// contents have been recorded.
        placement: FilterLayerPlacement,
    },
}

impl RecordedLayer {
    fn regular(props: LayerProps, cmds: Vec<RecordedCmd>, depth: usize) -> Self {
        Self {
            props,
            cmds,
            kind: RecordedLayerKind::Regular,
            depth,
            // Will be initialized once we call `pop_layer`.
            bbox: RectU16::INVERTED,
        }
    }

    fn filter(
        props: LayerProps,
        filter_plan: FilterData,
        cmds: Vec<RecordedCmd>,
        depth: usize,
    ) -> Self {
        Self {
            props,
            cmds,
            kind: RecordedLayerKind::Filter {
                filter_data: filter_plan,
                // Will be initialized once we call `pop_layer`.
                placement: FilterLayerPlacement::EMPTY,
            },
            depth,
            // Will be initialized once we call `pop_layer`.
            bbox: RectU16::INVERTED,
        }
    }
}

/// Records draw commands and nested layers into reusable command streams.
#[derive(Debug)]
pub struct CommandRecorder<D> {
    /// The commands of the root layer.
    pub root_cmds: Vec<RecordedCmd>,
    /// Flat storage for all draw commands referenced by batched draws.
    pub draws: Vec<D>,
    /// Data about recorded layers, indexed by their ID.
    pub layers: Vec<RecordedLayer>,
    /// Recorded filter layers in creation order.
    pub filter_layers: Vec<u32>,
    /// Whether the root has a direct child layer with a non-default blend mode.
    pub root_is_blend_target: bool,
    /// A pool for reusable `Vec<RecordedCmd>` allocations.
    cmd_pool: VecPool<RecordedCmd>,
    /// The layer whose command stream is currently the base.
    ///
    /// This is `None` if there is no active layer and we are recording into the root layer instead.
    active_layer: Option<u32>,
    /// Stack of currently pushed layers.
    layer_stack: Vec<OpenLayer>,
}

impl<D> Default for CommandRecorder<D> {
    fn default() -> Self {
        Self {
            root_cmds: Vec::new(),
            draws: Vec::new(),
            layers: Vec::new(),
            filter_layers: Vec::new(),
            root_is_blend_target: false,
            cmd_pool: VecPool::default(),
            active_layer: None,
            layer_stack: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct OpenLayer {
    id: u32,
    /// The bounding box of the contents recorded into this layer.
    bbox: RectU16,
    parent_layer: Option<u32>,
}

impl<D> CommandRecorder<D> {
    /// Create an empty command recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any layers are currently open.
    pub fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Clear all recorded commands and layer state while retaining reusable allocations.
    #[inline]
    pub fn reset(&mut self) {
        self.root_cmds.clear();
        self.draws.clear();

        for layer in self.layers.drain(..) {
            // Make sure to reuse the allocations for those!
            self.cmd_pool.submit(layer.cmds);
        }
        self.filter_layers.clear();
        self.root_is_blend_target = false;
        self.active_layer = None;
        self.layer_stack.clear();
    }

    /// Push a new regular or filter layer.
    #[inline]
    pub fn push_layer(&mut self, props: LayerProps, filter_plan: Option<FilterData>) {
        if let Some(filter_plan) = filter_plan {
            self.push_filter_layer(props, filter_plan);
            return;
        }

        self.push_regular_layer(props);
    }

    fn push_regular_layer(&mut self, props: LayerProps) {
        let cmds = self.cmd_pool.take();
        let depth = self.layer_stack.len() + 1;
        self.push_recorded_layer(RecordedLayer::regular(props, cmds, depth));
    }

    fn push_filter_layer(&mut self, props: LayerProps, filter_plan: FilterData) {
        let cmds = self.cmd_pool.take();
        let depth = self.layer_stack.len() + 1;
        let id = self.push_recorded_layer(RecordedLayer::filter(props, filter_plan, cmds, depth));
        self.filter_layers.push(id);
    }

    fn push_recorded_layer(&mut self, layer: RecordedLayer) -> u32 {
        let parent_layer = self.active_layer;
        if parent_layer.is_none() && layer.props.blend_mode != BlendMode::default() {
            self.root_is_blend_target = true;
        }
        let id = self.push_layer_metadata(layer);
        self.push_render_cmd(RecordedCmd::Layer(id));
        self.active_layer = Some(id);
        self.layer_stack.push(OpenLayer {
            id,
            // Will be populated as we record commands.
            bbox: RectU16::INVERTED,
            parent_layer,
        });
        id
    }

    /// Pop the currently active layer.
    pub fn pop_layer(&mut self) -> PoppedLayer {
        let layer = self.layer_stack.pop().unwrap();
        let id = layer.id;
        let recorded_layer = &mut self.layers[id as usize];
        let (popped_layer, bbox) = match &mut recorded_layer.kind {
            RecordedLayerKind::Regular => {
                let bbox = regular_layer_bbox(layer.bbox, &recorded_layer.props);
                recorded_layer.bbox = bbox;
                (PoppedLayer::Regular, bbox)
            }
            RecordedLayerKind::Filter {
                filter_data: filter_plan,
                placement,
                ..
            } => {
                *placement = FilterLayerPlacement::new(layer.bbox, filter_plan);
                recorded_layer.bbox = placement.pixmap_bbox;
                let dest_bbox = placement.dest_bbox;

                (PoppedLayer::Filter, dest_bbox)
            }
        };

        self.active_layer = layer.parent_layer;
        self.record_bbox(|| bbox);
        popped_layer
    }

    #[inline]
    fn active_cmds_mut(&mut self) -> &mut Vec<RecordedCmd> {
        self.layer_cmds_mut(self.active_layer)
    }

    fn layer_cmds_mut(&mut self, root_layer: Option<u32>) -> &mut Vec<RecordedCmd> {
        if let Some(id) = root_layer {
            &mut self.layers[id as usize].cmds
        } else {
            &mut self.root_cmds
        }
    }

    #[inline]
    fn push_render_cmd(&mut self, cmd: RecordedCmd) -> usize {
        let cmds = self.active_cmds_mut();
        let idx = cmds.len();
        cmds.push(cmd);
        idx
    }

    fn push_layer_metadata(&mut self, layer: RecordedLayer) -> u32 {
        let id = self.layers.len() as u32;
        self.layers.push(layer);
        id
    }

    fn push_draw_batch(&mut self, draw_idx: u32) {
        match self.active_cmds_mut().last_mut() {
            Some(RecordedCmd::Draws(range)) if range.end == draw_idx => {
                range.end += 1;
            }
            _ => {
                self.push_render_cmd(RecordedCmd::Draws(draw_idx..draw_idx + 1));
            }
        };
    }

    fn record_bbox(&mut self, bbox: impl FnOnce() -> RectU16) {
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.bbox.union(bbox());
        }
    }
}

fn regular_layer_bbox(mut bbox: RectU16, props: &LayerProps) -> RectU16 {
    if let Some(clip_path) = &props.clip_path {
        bbox = bbox.intersect(clip_path.bbox);
    }

    if bbox.is_empty() {
        bbox
    } else {
        bbox.snap_to_tile_coordinates()
    }
}

impl<D: Drawable> CommandRecorder<D> {
    /// Push a draw command into the current command stream.
    #[inline]
    pub fn push_draw(&mut self, draw: D, strips: &[Strip], viewport_width: u16) {
        self.record_bbox(|| draw.bbox(strips, viewport_width));
        let draw_idx = self.draws.len() as u32;
        self.draws.push(draw);
        self.push_draw_batch(draw_idx);
    }
}

/// The kind of layer popped from the recorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoppedLayer {
    /// A regular layer was popped.
    Regular,
    /// A filter layer was popped.
    Filter,
}

/// A clip path associated with a recorded layer.
#[derive(Debug, Clone)]
pub struct LayerClip {
    /// Strip range for the clip path.
    pub strip_range: Range<usize>,
    /// Index of the strip storage that owns the strip range.
    pub thread_idx: u8,
    /// Coarse clip path bounds in viewport coordinates.
    pub bbox: RectU16,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::{Filter, FilterPrimitive};
    use crate::kurbo::Affine;
    use crate::peniko::Mix;

    #[derive(Debug)]
    struct TestDraw;

    impl Drawable for TestDraw {
        fn bbox(&self, _strips: &[Strip], _viewport_width: u16) -> RectU16 {
            RectU16::new(0, 0, 64, 4)
        }
    }

    enum ExpectedCmd {
        Layer(usize),
        Batch(Range<u32>),
    }

    fn layer_props() -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: None,
        }
    }

    fn blended_layer_props() -> LayerProps {
        LayerProps {
            blend_mode: Mix::Multiply.into(),
            ..layer_props()
        }
    }

    fn filter_data(filter_padding: RectU16, source_padding: RectU16) -> FilterData {
        FilterData {
            filter: Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 }),
            transform: Affine::IDENTITY,
            filter_padding,
            source_padding,
        }
    }

    fn assert_cmds(cmds: &[RecordedCmd], expected: &[ExpectedCmd]) {
        assert_eq!(cmds.len(), expected.len());

        for (cmd, expected) in cmds.iter().zip(expected) {
            match (cmd, expected) {
                (RecordedCmd::Layer(id), ExpectedCmd::Layer(expected_id)) => {
                    assert_eq!(*id as usize, *expected_id);
                }
                (RecordedCmd::Draws(range), ExpectedCmd::Batch(expected_range)) => {
                    assert_eq!(range, expected_range);
                }
                _ => panic!("unexpected command: {cmd:?}"),
            }
        }
    }

    fn layer_cmds(recorder: &CommandRecorder<TestDraw>, id: usize) -> &[RecordedCmd] {
        &recorder.layers[id].cmds
    }

    #[test]
    fn layer_behavior() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        recorder.push_layer(layer_props(), None);

        recorder.push_draw(TestDraw, &[], 0);

        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        recorder.push_layer(layer_props(), None);
        recorder.push_draw(TestDraw, &[], 0);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        assert_cmds(&recorder.root_cmds, &[ExpectedCmd::Layer(0)]);
        assert_cmds(
            layer_cmds(&recorder, 0),
            &[ExpectedCmd::Layer(1), ExpectedCmd::Layer(3)],
        );
        assert_cmds(layer_cmds(&recorder, 1), &[ExpectedCmd::Layer(2)]);
        assert_cmds(layer_cmds(&recorder, 2), &[ExpectedCmd::Batch(0..1)]);
        assert_cmds(layer_cmds(&recorder, 3), &[ExpectedCmd::Batch(1..2)]);
        assert_eq!(recorder.draws.len(), 2);
        assert_eq!(
            recorder.filter_layers.iter().copied().collect::<Vec<_>>(),
            [0, 1],
        );
    }

    #[test]
    fn draw_batches_are_split_by_layers() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        recorder.push_draw(TestDraw, &[], 0);
        recorder.push_draw(TestDraw, &[], 0);
        recorder.push_layer(layer_props(), None);
        recorder.push_draw(TestDraw, &[], 0);
        recorder.pop_layer();
        recorder.push_draw(TestDraw, &[], 0);

        assert_cmds(
            &recorder.root_cmds,
            &[
                ExpectedCmd::Batch(0..2),
                ExpectedCmd::Layer(0),
                ExpectedCmd::Batch(3..4),
            ],
        );
        assert_cmds(layer_cmds(&recorder, 0), &[ExpectedCmd::Batch(2..3)]);
    }

    #[test]
    fn blend_into_root() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        assert!(!recorder.root_is_blend_target);

        recorder.push_layer(layer_props(), None);
        assert!(!recorder.root_is_blend_target);
        recorder.pop_layer();

        recorder.push_layer(blended_layer_props(), None);
        assert!(recorder.root_is_blend_target);

        recorder.reset();
        assert!(!recorder.root_is_blend_target);
    }

    #[test]
    fn blend_into_non_root() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        recorder.push_layer(layer_props(), None);
        recorder.push_layer(blended_layer_props(), None);

        assert!(!recorder.root_is_blend_target);
    }
}
