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
use crate::geometry::RectU16;
use crate::mask::Mask;
use crate::paint::Paint;
use crate::peniko::BlendMode;
use crate::strip::Strip;
use crate::util::{VecPool, strip_bbox};
use alloc::vec::Vec;
use core::ops::Range;

/// A recorded command.
#[derive(Debug)]
pub enum RecordedCmd {
    /// A path fill command.
    Fill {
        /// Index of the thread-local strip storage containing the strips.
        thread_idx: u8,
        /// Range of strips representing the fill.
        strip_range: Range<usize>,
        /// Paint applied to the fill.
        paint: Paint,
        /// Blend mode applied to the fill.
        blend_mode: BlendMode,
        /// Optional mask applied to the fill.
        mask: Option<Mask>,
    },
    /// Push a new (non-filter) layer to the layer stack.
    PushLayer {
        /// Identifier of the layer.
        id: LayerId,
    },
    /// Composite the filter layer with the given ID into the current layer.
    FilterLayer {
        /// Identifier of the filter layer.
        id: LayerId,
    },
    /// Pop the last (non-filter) layer from the layer stack.
    PopLayer,
}

/// Identifier of a recorded layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerId(u32);

impl LayerId {
    /// Create a layer identifier from an index.
    pub fn new(id: usize) -> Self {
        Self(id as u32)
    }

    /// Return the layer index.
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

/// Metadata for a recorded layer.
#[derive(Debug)]
pub struct RecordedLayer {
    /// Properties of the layer.
    pub props: LayerProps,
    /// Kind-specific layer data.
    pub kind: RecordedLayerKind,
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

/// Additional data for each kind of recorded layer.
#[derive(Debug)]
pub enum RecordedLayerKind {
    /// A regular layer whose commands will be inlined into the parent root layer.
    Regular,
    /// A filter layer storing its commands separately.
    Filter {
        /// The render commands of the layer.
        cmds: Vec<RecordedCmd>,
        /// Static data about the filter itself.
        filter_data: FilterData,
        /// Data about how to place the filter layer, which can only be determined once its
        /// contents have been recorded.
        placement: FilterLayerPlacement,
    },
}

impl RecordedLayer {
    /// Create a regular recorded layer.
    pub fn regular(props: LayerProps) -> Self {
        Self {
            props,
            kind: RecordedLayerKind::Regular,
        }
    }

    fn filter(props: LayerProps, filter_plan: FilterData, cmds: Vec<RecordedCmd>) -> Self {
        Self {
            props,
            kind: RecordedLayerKind::Filter {
                cmds,
                filter_data: filter_plan,
                // Will be initialized once we call `pop_layer`.
                placement: FilterLayerPlacement::EMPTY,
            },
        }
    }
}

/// Recorder for fills and nested layers.
#[derive(Debug, Default)]
pub struct CommandRecorder {
    /// The commands of the root layer.
    pub root_cmds: Vec<RecordedCmd>,
    /// Data about recorded layers, indexed by their ID.
    pub layers: Vec<RecordedLayer>,
    /// A pool for reusable `Vec<RecordedCmd>` allocations.
    cmd_pool: VecPool<RecordedCmd>,
    /// The filter layer whose command stream is currently the base.
    ///
    /// This is `None` if there is no active filter layer and we are recording into the
    /// root layer instead.
    active_filter_layer: Option<LayerId>,
    /// Stack of currently pushed layers.
    layer_stack: Vec<OpenLayer>,
}

#[derive(Debug)]
struct OpenLayer {
    id: LayerId,
    /// The bounding box of the contents recorded into this layer.
    bbox: RectU16,
    enclosing_filter_layer: Option<LayerId>,
}

impl CommandRecorder {
    /// Create a new command recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return whether any layers are currently open.
    pub fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Reset the recorder.
    #[inline]
    pub fn reset(&mut self) {
        self.root_cmds.clear();

        for layer in self.layers.drain(..) {
            // Make sure to reuse the allocations for those!
            if let RecordedLayerKind::Filter { cmds, .. } = layer.kind {
                self.cmd_pool.submit(cmds);
            }
        }
        self.active_filter_layer = None;
        self.layer_stack.clear();
    }

    /// Record a path fill.
    #[inline]
    pub fn push_fill(
        &mut self,
        strip_range: Range<usize>,
        strips: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        thread_idx: u8,
    ) {
        self.active_cmds_mut().push(RecordedCmd::Fill {
            thread_idx,
            strip_range,
            paint,
            blend_mode,
            mask,
        });

        self.record_bbox(|| strip_bbox(strips));
    }

    /// Push a layer.
    pub fn push_layer(&mut self, props: LayerProps, filter_plan: Option<FilterData>) {
        if let Some(filter_plan) = filter_plan {
            self.push_filter_layer(props, filter_plan);
            return;
        }

        self.push_regular_layer(props);
    }

    fn push_regular_layer(&mut self, props: LayerProps) {
        let id = self.push_layer_metadata(RecordedLayer::regular(props));
        self.push_render_cmd(RecordedCmd::PushLayer { id });
        self.layer_stack.push(OpenLayer {
            id,
            // Will be populated as we record commands.
            bbox: RectU16::INVERTED,
            enclosing_filter_layer: self.active_filter_layer,
        });
    }

    fn push_filter_layer(&mut self, props: LayerProps, filter_plan: FilterData) {
        let enclosing_filter_layer = self.active_filter_layer;
        let cmds = self.cmd_pool.take();
        let id = self.push_layer_metadata(RecordedLayer::filter(props, filter_plan, cmds));
        self.push_render_cmd(RecordedCmd::FilterLayer { id });
        self.active_filter_layer = Some(id);
        self.layer_stack.push(OpenLayer {
            id,
            // Will be populated as we record commands.
            bbox: RectU16::INVERTED,
            enclosing_filter_layer,
        });
    }

    /// Pop the currently active layer.
    pub fn pop_layer(&mut self) -> PoppedLayer {
        let layer = self.layer_stack.pop().unwrap();
        let id = layer.id;
        match &mut self.layers[id.get()].kind {
            RecordedLayerKind::Regular => {
                self.record_bbox(|| layer.bbox);
                self.active_cmds_mut().push(RecordedCmd::PopLayer);

                PoppedLayer::Regular
            }
            RecordedLayerKind::Filter {
                filter_data: filter_plan,
                placement,
                ..
            } => {
                *placement = FilterLayerPlacement::new(layer.bbox, filter_plan);
                let dest_bbox = placement.dest_bbox;

                self.active_filter_layer = layer.enclosing_filter_layer;
                self.record_bbox(|| dest_bbox);
                PoppedLayer::Filter
            }
        }
    }

    #[inline]
    fn active_cmds_mut(&mut self) -> &mut Vec<RecordedCmd> {
        self.layer_cmds_mut(self.active_filter_layer)
    }

    fn layer_cmds_mut(&mut self, root_layer: Option<LayerId>) -> &mut Vec<RecordedCmd> {
        if let Some(id) = root_layer {
            match &mut self.layers[id.get()].kind {
                RecordedLayerKind::Filter { cmds, .. } => cmds,
                RecordedLayerKind::Regular => {
                    unreachable!("regular layers cannot be active root layers")
                }
            }
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

    fn push_layer_metadata(&mut self, layer: RecordedLayer) -> LayerId {
        let id = LayerId::new(self.layers.len());
        self.layers.push(layer);
        id
    }

    fn record_bbox(&mut self, bbox: impl FnOnce() -> RectU16) {
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.bbox.union(bbox());
        }
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
    use crate::color::palette::css::BLACK;
    use crate::filter_effects::{Filter, FilterPrimitive};
    use crate::kurbo::Affine;
    use crate::paint::PremulColor;

    enum ExpectedCmd {
        PushLayer(usize),
        FilterLayer(usize),
        Fill,
        PopLayer,
    }

    fn row_end(x: u16, y: u16, alpha_idx: u32, fill_gap: bool) -> Strip {
        Strip::new(x, y, alpha_idx, fill_gap)
    }

    fn layer_props() -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: None,
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
                (RecordedCmd::PushLayer { id }, ExpectedCmd::PushLayer(expected_id)) => {
                    assert_eq!(id.get(), *expected_id);
                }
                (RecordedCmd::FilterLayer { id }, ExpectedCmd::FilterLayer(expected_id)) => {
                    assert_eq!(id.get(), *expected_id);
                }
                (RecordedCmd::Fill { .. }, ExpectedCmd::Fill) => {}
                (RecordedCmd::PopLayer, ExpectedCmd::PopLayer) => {}
                _ => panic!("unexpected command: {cmd:?}"),
            }
        }
    }

    fn filter_cmds(recorder: &CommandRecorder, id: usize) -> &[RecordedCmd] {
        match &recorder.layers[id].kind {
            RecordedLayerKind::Filter { cmds, .. } => cmds,
            RecordedLayerKind::Regular => panic!("expected filter layer"),
        }
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
        let mut recorder = CommandRecorder::new();

        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        recorder.push_layer(
            layer_props(),
            Some(filter_data(RectU16::ZERO, RectU16::ZERO)),
        );
        recorder.push_layer(layer_props(), None);

        recorder.push_fill(
            0..3,
            &[
                Strip::new(0, 0, 0, false),
                row_end(64, 0, 16, true),
                Strip::sentinel(0, 16),
            ],
            Paint::Solid(PremulColor::from_alpha_color(BLACK)),
            BlendMode::default(),
            None,
            0,
        );

        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        recorder.push_layer(layer_props(), None);
        recorder.push_fill(
            0..3,
            &[
                Strip::new(0, 0, 16, false),
                row_end(64, 0, 32, true),
                Strip::sentinel(0, 32),
            ],
            Paint::Solid(PremulColor::from_alpha_color(BLACK)),
            BlendMode::default(),
            None,
            0,
        );
        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        assert_cmds(&recorder.root_cmds, &[ExpectedCmd::FilterLayer(0)]);
        assert_cmds(
            filter_cmds(&recorder, 0),
            &[
                ExpectedCmd::FilterLayer(1),
                ExpectedCmd::PushLayer(3),
                ExpectedCmd::Fill,
                ExpectedCmd::PopLayer,
            ],
        );
        assert_cmds(
            filter_cmds(&recorder, 1),
            &[
                ExpectedCmd::PushLayer(2),
                ExpectedCmd::Fill,
                ExpectedCmd::PopLayer,
            ],
        );
    }
}
