// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recording rendering commands.
//!
//! (Note: Below description was written with Vello CPU in mind, but also applies to
//! Vello Hybrid, except for the fact that we don't do bucketing in Vello Hybrid.)
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
    fn bbox(&self, strips: &[Strip]) -> RectU16;
}

/// A node in the recorded render graph.
#[derive(Debug)]
pub struct CmdNode {
    /// A contiguous batch of draw commands in [`CommandRecorder::draws`].
    pub draws: Range<u32>,
    /// An optional layer composition, invoked after `draws`.
    pub layer: Option<u32>,
}

/// Metadata and command stream for one recorded layer.
#[derive(Debug)]
pub struct RecordedLayer {
    /// Layer compositing properties.
    pub props: LayerProps,
    /// The render commands of the layer.
    pub cmds: Vec<CmdNode>,
    /// The kind of recorded layer.
    pub kind: RecordedLayerKind,
    /// Nesting depth. Direct child layers of the root have depth 1.
    pub depth: usize,
    /// Bounding box of the layer.
    pub bbox: RectU16,
}

/// Compositing properties for a recorded layer.
#[derive(Debug)]
pub struct LayerProps {
    /// Blend mode used when compositing the layer into its parent.
    pub blend_mode: BlendMode,
    /// Opacity applied when compositing the layer into its parent.
    pub opacity: f32,
    /// Optional mask applied when compositing the layer into its parent.
    pub mask: Option<Mask>,
    /// Optional clip path applied when compositing the layer into its parent.
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
    fn regular(props: LayerProps, cmds: Vec<CmdNode>, depth: usize) -> Self {
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
        cmds: Vec<CmdNode>,
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
    pub root_cmds: Vec<CmdNode>,
    /// Flat storage for all draw commands referenced by batched draws.
    pub draws: Vec<D>,
    /// Data about recorded layers, indexed by their ID.
    pub layers: Vec<RecordedLayer>,
    /// IDs of recorded filter layers in creation order.
    pub filter_layers: Vec<u32>,
    /// Whether the root is the target of a non-default blending operation.
    pub root_is_blend_target: bool,
    /// Maximum layer depth across the whole layer graph.
    pub max_layer_depth: usize,
    /// Whether there exists at least one layer that uses a non-default blend mode.
    pub has_non_default_blend: bool,
    /// Whether there exists at least one layer that has a filter.
    pub has_filter_layer: bool,
    /// A pool for reusable `Vec<CmdNode>` allocations.
    cmd_pool: VecPool<CmdNode>,
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
            max_layer_depth: 0,
            has_non_default_blend: false,
            has_filter_layer: false,
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
    /// Create an new command recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any layers are currently open.
    pub fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Reset the command recorder.
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
        self.max_layer_depth = 0;
        self.has_non_default_blend = false;
        self.has_filter_layer = false;
        self.active_layer = None;
        self.layer_stack.clear();
    }

    /// Push a new layer.
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
        self.has_filter_layer = true;
    }

    fn push_recorded_layer(&mut self, layer: RecordedLayer) -> u32 {
        let parent_layer = self.active_layer;
        self.max_layer_depth = self.max_layer_depth.max(layer.depth);
        if layer.props.blend_mode != BlendMode::default() {
            self.has_non_default_blend = true;

            if parent_layer.is_none() {
                self.root_is_blend_target = true;
            }
        }
        let id = self.push_layer_metadata(layer);
        self.push_layer_node(id);
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
                let mut bbox = layer.bbox;

                if let Some(clip_path) = &recorded_layer.props.clip_path {
                    bbox = bbox
                        .intersect(clip_path.bbox)
                        // The clip path bounding box is derived directly from the clip path
                        // (and hence not necessarily tile-aligned), but we want to ensure that
                        // all bounding boxes _are_ tile-aligned since they should denote the
                        // bounding box of the strips, which are always tile-aligned.
                        .snap_to_tile_coordinates();
                }

                recorded_layer.bbox = bbox;
                (PoppedLayer::Regular, bbox)
            }
            RecordedLayerKind::Filter {
                filter_data: filter_plan,
                placement,
                ..
            } => {
                *placement = FilterLayerPlacement::new(layer.bbox, filter_plan);
                // Unlike normal layers, we don't want to intersect with the current clip bbox
                // because effects of spatial filters shouldn't be clipped away.
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
    fn active_cmds_mut(&mut self) -> &mut Vec<CmdNode> {
        self.layer_cmds_mut(self.active_layer)
    }

    fn layer_cmds_mut(&mut self, root_layer: Option<u32>) -> &mut Vec<CmdNode> {
        if let Some(id) = root_layer {
            &mut self.layers[id as usize].cmds
        } else {
            &mut self.root_cmds
        }
    }

    fn push_layer_node(&mut self, layer_id: u32) {
        let draw_idx = self.draws.len() as u32;

        match self.active_cmds_mut().last_mut() {
            Some(node) if node.layer.is_none() => {
                debug_assert_eq!(node.draws.end, draw_idx);

                node.layer = Some(layer_id);
            }
            _ => self.active_cmds_mut().push(CmdNode {
                draws: draw_idx..draw_idx,
                layer: Some(layer_id),
            }),
        }
    }

    fn push_layer_metadata(&mut self, layer: RecordedLayer) -> u32 {
        let id = self.layers.len() as u32;
        self.layers.push(layer);
        id
    }

    fn record_bbox(&mut self, bbox: impl FnOnce() -> RectU16) {
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.bbox.union(bbox());
        }
    }
}

impl<D: Drawable> CommandRecorder<D> {
    /// Push a draw command into the current command stream.
    #[inline]
    pub fn push_draw(&mut self, draw: D, strips: &[Strip]) {
        self.record_bbox(|| draw.bbox(strips));
        let draw_idx = self.draws.len() as u32;
        self.draws.push(draw);

        match self.active_cmds_mut().last_mut() {
            Some(node) if node.layer.is_none() && node.draws.end == draw_idx => {
                node.draws.end += 1;
            }
            _ => {
                self.active_cmds_mut().push(CmdNode {
                    draws: draw_idx..draw_idx + 1,
                    layer: None,
                });
            }
        };
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
        fn bbox(&self, _strips: &[Strip]) -> RectU16 {
            RectU16::new(0, 0, 64, 4)
        }
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

    fn assert_cmds(cmds: &[CmdNode], expected: &[(Range<u32>, Option<u32>)]) {
        assert_eq!(cmds.len(), expected.len());

        for (cmd, (draws, layer)) in cmds.iter().zip(expected) {
            assert_eq!(&cmd.draws, draws);
            assert_eq!(cmd.layer, *layer);
        }
    }

    fn layer_cmds(recorder: &CommandRecorder<TestDraw>, id: usize) -> &[CmdNode] {
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

        recorder.push_draw(TestDraw, &[]);

        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        recorder.push_layer(layer_props(), None);
        recorder.push_draw(TestDraw, &[]);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        assert_cmds(&recorder.root_cmds, &[(0..0, Some(0))]);
        assert_cmds(
            layer_cmds(&recorder, 0),
            &[(0..0, Some(1)), (1..1, Some(3))],
        );
        assert_cmds(layer_cmds(&recorder, 1), &[(0..0, Some(2))]);
        assert_cmds(layer_cmds(&recorder, 2), &[(0..1, None)]);
        assert_cmds(layer_cmds(&recorder, 3), &[(1..2, None)]);
        assert_eq!(recorder.draws.len(), 2);
        assert_eq!(recorder.filter_layers.to_vec(), [0, 1]);
        assert_eq!(recorder.max_layer_depth, 3);
        assert!(!recorder.has_non_default_blend);
    }

    #[test]
    fn draw_batches_are_split_by_layers() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        recorder.push_draw(TestDraw, &[]);
        recorder.push_draw(TestDraw, &[]);
        recorder.push_layer(layer_props(), None);
        recorder.push_draw(TestDraw, &[]);
        recorder.pop_layer();
        recorder.push_draw(TestDraw, &[]);

        assert_cmds(&recorder.root_cmds, &[(0..2, Some(0)), (3..4, None)]);
        assert_cmds(layer_cmds(&recorder, 0), &[(2..3, None)]);
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
        assert!(recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 1);
        assert!(!recorder.has_filter_layer);

        recorder.reset();
        assert!(!recorder.root_is_blend_target);
        assert!(!recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 0);
        assert!(!recorder.has_filter_layer);
    }

    #[test]
    fn blend_into_non_root() {
        let mut recorder = CommandRecorder::<TestDraw>::new();

        recorder.push_layer(layer_props(), None);
        recorder.push_layer(blended_layer_props(), None);

        assert!(!recorder.root_is_blend_target);
        assert!(recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 2);
    }

    #[test]
    fn filter_layers_are_tracked() {
        let mut recorder = CommandRecorder::<TestDraw>::new();
        assert!(!recorder.has_filter_layer);

        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        assert!(recorder.has_filter_layer);

        recorder.reset();
        assert!(!recorder.has_filter_layer);
    }
}
