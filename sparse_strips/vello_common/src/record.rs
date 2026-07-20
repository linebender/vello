// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recording rendering commands.
//!
//! Vello CPU and Vello Hybrid share a recording stage before their pipelines diverge. Their
//! pipelines can be split into roughly three parts:
//!
//! 1. Record rendering commands into a scene-graph-like structure.
//! 2. Turn the complete recording into renderer-specific work. Vello CPU buckets commands by
//!    strip row, while Vello Hybrid schedules draws and layer operations
//!    across intermediate textures and render passes.
//! 3. Execute that work using CPU fine rasterization or GPU render passes, respectively.
//!
//! The reasons for completing the recording before renderer-specific planning differ somewhat
//! between the two renderers:
//!
//! - Vello CPU can inline regular layers with blends, opacity, masks, or clips into their parent
//!   command stream. However, filter layers must instead be rendered separately because
//!   spatial filters might sample neighboring pixels. The complete layer must be rendered before
//!   the filter can be applied and its result composited into the parent.
//!
//! - Vello Hybrid needs to render *every* layer separately. Its scheduler therefore
//!   needs the complete layer hierarchy and bounds before it can allocate intermediate textures and
//!   order draws, filters, and composition operations.
//!
//! - In both renderers, filter layers might have different dimensions than the main viewport. A
//!   large blur, for example, might require rendering shapes that would normally be culled because
//!   they exceed the viewport.
//!
//! The recording stage allows us to serialize the whole scene into a graph that is enrichened
//! with metadata, allowing each renderer to "do their own thing" while providing a common
//! intermediate representation.

use crate::filter::{FilterData, FilterLayerPlacement};
use crate::geometry::{RectU16, SizeU16};
use crate::mask::Mask;
use crate::peniko::BlendMode;
use crate::strip::Strip;
use crate::util::RectExt;
use alloc::vec::Vec;
use core::ops::Range;
use smallvec::SmallVec;

/// A drawable object that can report its bounding box.
pub trait Drawable {
    /// Return the bounding box of the given object.
    fn bbox(&self, strips: &[Strip]) -> RectU16;
}

/// A node in the recorded render graph.
#[derive(Debug)]
pub struct Node {
    /// A contiguous (possibly empty) batch of draw commands indexing [`CommandRecorder::draws`].
    pub draws: Range<u32>,
    /// An optional layer composition, invoked after `draws`.
    pub layer: Option<u32>,
}

impl Node {
    /// Return the draw commands referenced by this node.
    pub fn draws_in<'a, D>(&self, draws: &'a [D]) -> &'a [D] {
        &draws[self.draws.start as usize..self.draws.end as usize]
    }
}

/// Metadata and child nodes for a recorded layer.
#[derive(Debug)]
pub struct RecordedLayer {
    /// Properties of the layer.
    pub props: LayerProps,
    /// The child nodes of the layer.
    pub nodes: SmallVec<[Node; 2]>,
    /// The kind of recorded layer.
    pub kind: RecordedLayerKind,
    /// Nesting depth of the layer.
    pub depth: usize,
    /// Bounding box of the layer.
    pub bbox: RectU16,
}

/// Properties for a recorded layer.
#[derive(Debug)]
pub struct LayerProps {
    /// Blend mode used when compositing the layer.
    pub blend_mode: BlendMode,
    /// Opacity applied when compositing the layer.
    pub opacity: f32,
    /// Optional mask applied when compositing the layer.
    pub mask: Option<Mask>,
    /// Optional clip path applied when compositing the layer.
    pub clip_path: Option<LayerClip>,
}

/// Clip path associated with a recorded layer.
#[derive(Debug, Clone)]
pub struct LayerClip {
    /// Range of strips representing the clip path.
    pub strip_range: Range<usize>,
    /// Index of the thread-local strip storage containing the strips.
    pub thread_idx: u8,
    /// Coarse bounds of the clip path.
    pub bbox: RectU16,
}

/// Additional metadata for regular and filter layers.
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
    fn regular(props: LayerProps, depth: usize) -> Self {
        Self {
            props,
            nodes: SmallVec::new(),
            kind: RecordedLayerKind::Regular,
            depth,
            // Will be initialized once we call `pop_layer`.
            bbox: RectU16::INVERTED,
        }
    }

    fn filter(props: LayerProps, filter_plan: FilterData, depth: usize) -> Self {
        Self {
            props,
            nodes: SmallVec::new(),
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

// TODO: Rename this: https://github.com/linebender/vello/pull/1746#discussion_r3611799919
/// Recorder for a scene description.
#[derive(Debug)]
pub struct CommandRecorder<D> {
    /// Tile-aligned dimensions of the root scene.
    pub scene_size: SizeU16,
    /// The nodes of the root layer.
    pub nodes: Vec<Node>,
    /// Flat storage for all draw commands that are part of the recording.
    pub draws: Vec<D>,
    /// Data about recorded layers, indexed by their ID.
    pub layers: Vec<RecordedLayer>,
    /// IDs of recorded filter layers in creation order.
    pub filter_layers: Vec<u32>,
    /// Whether the root is the target of a non-default blending operation.
    pub root_is_blend_target: bool,
    /// Maximum layer depth across the whole layer graph.
    pub max_layer_depth: usize,
    /// The largest dimensions of any recorded layer.
    pub largest_layer_size: Option<SizeU16>,
    /// The largest dimensions of any recorded filter layer.
    pub largest_filter_layer_size: Option<SizeU16>,
    /// Whether there exists at least one layer that uses a non-default blend mode.
    pub has_non_default_blend: bool,
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
            scene_size: SizeU16::ZERO,
            nodes: Vec::new(),
            draws: Vec::new(),
            layers: Vec::new(),
            filter_layers: Vec::new(),
            root_is_blend_target: false,
            max_layer_depth: 0,
            largest_layer_size: None,
            largest_filter_layer_size: None,
            has_non_default_blend: false,
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
    /// Create a new command recorder.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            scene_size: snapped_scene_size(width, height),
            ..Self::default()
        }
    }

    /// Whether any layers are currently open.
    pub fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Reset the command recorder.
    #[inline]
    pub fn reset(&mut self, width: u16, height: u16) {
        self.scene_size = snapped_scene_size(width, height);
        self.nodes.clear();
        self.draws.clear();

        self.layers.clear();
        self.filter_layers.clear();
        self.root_is_blend_target = false;
        self.max_layer_depth = 0;
        self.largest_layer_size = None;
        self.largest_filter_layer_size = None;
        self.has_non_default_blend = false;
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
        let depth = self.layer_stack.len() + 1;
        self.push_recorded_layer(RecordedLayer::regular(props, depth));
    }

    fn push_filter_layer(&mut self, props: LayerProps, filter_plan: FilterData) {
        let depth = self.layer_stack.len() + 1;
        let id = self.push_recorded_layer(RecordedLayer::filter(props, filter_plan, depth));

        self.filter_layers.push(id);
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

        let (popped_layer, bbox_in_parent) = {
            let recorded_layer = &mut self.layers[id as usize];
            match &mut recorded_layer.kind {
                RecordedLayerKind::Regular => {
                    recorded_layer.bbox = layer.bbox;

                    let layer_size = layer.bbox.into();
                    self.largest_layer_size = Some(
                        self.largest_layer_size
                            .map_or(layer_size, |current| current.max(layer_size)),
                    );

                    (PoppedLayer::Regular, layer.bbox)
                }
                RecordedLayerKind::Filter {
                    filter_data: filter_plan,
                    placement,
                } => {
                    *placement = FilterLayerPlacement::new(layer.bbox, filter_plan);
                    recorded_layer.bbox = placement.pixmap_bbox;

                    let filter_size = placement.pixmap_bbox.into();
                    self.largest_layer_size = Some(
                        self.largest_layer_size
                            .map_or(filter_size, |current| current.max(filter_size)),
                    );
                    self.largest_filter_layer_size = Some(
                        self.largest_filter_layer_size
                            .map_or(filter_size, |current| current.max(filter_size)),
                    );

                    (PoppedLayer::Filter, placement.dest_bbox)
                }
            }
        };

        // Update the parent bbox as well.
        self.active_layer = layer.parent_layer;
        self.record_bbox(|| bbox_in_parent);

        popped_layer
    }

    #[inline]
    fn active_node_mut(&mut self) -> Option<&mut Node> {
        if let Some(id) = self.active_layer {
            self.layers[id as usize].nodes.last_mut()
        } else {
            self.nodes.last_mut()
        }
    }

    fn push_node(&mut self, node: Node) {
        if let Some(id) = self.active_layer {
            self.layers[id as usize].nodes.push(node);
        } else {
            self.nodes.push(node);
        }
    }

    fn push_layer_node(&mut self, layer_id: u32) {
        let draw_idx = self.draws.len() as u32;

        match self.active_node_mut() {
            Some(node) if node.layer.is_none() => {
                node.layer = Some(layer_id);
            }
            _ => self.push_node(Node {
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

fn snapped_scene_size(width: u16, height: u16) -> SizeU16 {
    RectU16::new(0, 0, width, height)
        .snap_to_tile_coordinates()
        .into()
}

impl<D: Drawable> CommandRecorder<D> {
    /// Push a draw command.
    #[inline]
    pub fn push_draw(&mut self, draw: D, strips: &[Strip]) {
        self.record_bbox(|| draw.bbox(strips));
        let draw_idx = self.draws.len() as u32;
        self.draws.push(draw);

        match self.active_node_mut() {
            Some(node) if node.layer.is_none() && node.draws.end == draw_idx => {
                node.draws.end += 1;
            }
            _ => {
                self.push_node(Node {
                    draws: draw_idx..draw_idx + 1,
                    layer: None,
                });
            }
        };
    }
}

/// Kind of layer returned by [`CommandRecorder::pop_layer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoppedLayer {
    /// A regular layer.
    Regular,
    /// A filter layer.
    Filter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::{Filter, FilterPrimitive};
    use crate::kurbo::Affine;
    use crate::peniko::Mix;
    use crate::tile::Tile;

    const DEFAULT_SIZE: u16 = 10;

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

    fn assert_cmds(cmds: &[Node], expected: &[(Range<u32>, Option<u32>)]) {
        assert_eq!(cmds.len(), expected.len());

        for (cmd, (draws, layer)) in cmds.iter().zip(expected) {
            assert_eq!(&cmd.draws, draws);
            assert_eq!(cmd.layer, *layer);
        }
    }

    fn layer_cmds(recorder: &CommandRecorder<TestDraw>, id: usize) -> &[Node] {
        &recorder.layers[id].nodes
    }

    #[test]
    fn scene_size_is_tile_aligned() {
        let mut recorder = CommandRecorder::<TestDraw>::new(10, 10);
        assert_eq!(recorder.scene_size, SizeU16::new(12));

        recorder.reset(13, 7);
        assert_eq!(recorder.scene_size, SizeU16::from_wh(16, 8));

        recorder.reset(Tile::WIDTH * 5, Tile::HEIGHT * 3);
        assert_eq!(
            recorder.scene_size,
            SizeU16::from_wh(Tile::WIDTH * 5, Tile::HEIGHT * 3)
        );
    }

    #[test]
    fn filter_placement_padding_expands_bbox() {
        let placement = FilterLayerPlacement::new(
            RectU16::new(8, 8, 16, 20),
            &filter_data(RectU16::new(2, 4, 6, 8), RectU16::ZERO),
        );

        // Since we are tile-aligned, values are expanded to a multiple of tile-size.
        assert_eq!(placement.pixmap_bbox, RectU16::new(4, 4, 24, 28));
        assert_eq!(placement.dest_bbox, RectU16::new(4, 4, 24, 28));
        assert_eq!(placement.src_origin(), (0, 0));
    }

    #[test]
    fn filter_placement_with_source_shift() {
        let placement = FilterLayerPlacement::new(
            RectU16::new(8, 12, 20, 24),
            &filter_data(RectU16::new(6, 2, 4, 6), RectU16::new(10, 16, 0, 0)),
        );

        // Bbox expanded with padding is [8 - 6, 12 - 2, 20 + 4, 24 + 6]
        // = [2, 10, 24, 30], snappding this gives us [0, 8, 24, 32].
        assert_eq!(placement.pixmap_bbox, RectU16::new(0, 8, 24, 32));
        // Account for source origin using saturing sub of 10 horizontally and
        // 16 vertically.
        assert_eq!(placement.dest_bbox, RectU16::new(0, 0, 14, 16));
        // Source origin is now 10 - 0 = 10 and 16 - 8 = 8.
        assert_eq!(placement.src_origin(), (10, 8));
    }

    #[test]
    fn layer_behavior() {
        let mut recorder = CommandRecorder::<TestDraw>::new(DEFAULT_SIZE, DEFAULT_SIZE);

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

        assert_cmds(&recorder.nodes, &[(0..0, Some(0))]);
        assert_cmds(
            layer_cmds(&recorder, 0),
            &[(0..0, Some(1)), (1..1, Some(3))],
        );
        assert_cmds(layer_cmds(&recorder, 1), &[(0..0, Some(2))]);
        assert_cmds(layer_cmds(&recorder, 2), &[(0..1, None)]);
        assert_cmds(layer_cmds(&recorder, 3), &[(1..2, None)]);
        assert_eq!(recorder.draws.len(), 2);
        assert_eq!(recorder.filter_layers, [0, 1]);
        assert_eq!(
            recorder
                .layers
                .iter()
                .map(|layer| layer.depth)
                .collect::<Vec<_>>(),
            [1, 2, 3, 2]
        );
        assert_eq!(recorder.max_layer_depth, 3);
        assert!(!recorder.has_non_default_blend);
        assert_eq!(recorder.largest_layer_size, Some(SizeU16::from_wh(64, 4)));
        assert_eq!(
            recorder.largest_filter_layer_size,
            Some(SizeU16::from_wh(64, 4))
        );
    }

    #[test]
    fn draw_batches_are_split_by_layers() {
        let mut recorder = CommandRecorder::<TestDraw>::new(DEFAULT_SIZE, DEFAULT_SIZE);

        recorder.push_draw(TestDraw, &[]);
        recorder.push_draw(TestDraw, &[]);
        recorder.push_layer(layer_props(), None);
        recorder.push_draw(TestDraw, &[]);
        recorder.pop_layer();
        recorder.push_draw(TestDraw, &[]);

        assert_cmds(&recorder.nodes, &[(0..2, Some(0)), (3..4, None)]);
        assert_cmds(layer_cmds(&recorder, 0), &[(2..3, None)]);
    }

    #[test]
    fn node_resolves_draw_range() {
        let draws = [0, 1, 2, 3];
        let node = Node {
            draws: 1..3,
            layer: None,
        };

        assert_eq!(node.draws_in(&draws), &[1, 2]);
    }

    #[test]
    fn blend_metadata_distinguishes_root_and_nested_targets() {
        let mut recorder = CommandRecorder::<TestDraw>::new(DEFAULT_SIZE, DEFAULT_SIZE);

        recorder.push_layer(layer_props(), None);
        recorder.push_layer(blended_layer_props(), None);

        assert!(!recorder.root_is_blend_target);
        assert!(recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 2);

        recorder.pop_layer();
        recorder.pop_layer();
        recorder.push_layer(blended_layer_props(), None);

        assert!(recorder.root_is_blend_target);
    }

    #[test]
    fn reset_clears_all_metadata() {
        let mut recorder = CommandRecorder::<TestDraw>::new(DEFAULT_SIZE, DEFAULT_SIZE);

        recorder.push_layer(blended_layer_props(), None);
        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        recorder.push_draw(TestDraw, &[]);
        recorder.pop_layer();
        recorder.pop_layer();

        assert!(recorder.root_is_blend_target);
        assert!(recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 2);
        assert!(recorder.largest_layer_size.is_some());
        assert!(recorder.largest_filter_layer_size.is_some());

        recorder.reset(13, 7);

        assert_eq!(recorder.scene_size, SizeU16::from_wh(16, 8));
        assert!(!recorder.root_is_blend_target);
        assert!(!recorder.has_non_default_blend);
        assert_eq!(recorder.max_layer_depth, 0);
        assert!(recorder.largest_layer_size.is_none());
        assert!(recorder.largest_filter_layer_size.is_none());
        assert!(recorder.filter_layers.is_empty());
    }
}
