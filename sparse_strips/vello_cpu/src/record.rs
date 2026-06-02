// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::coarse::{LayerClip, RenderCmd};
use crate::kurbo::{Affine, Rect};
use crate::peniko::BlendMode;
use alloc::vec::Vec;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::tile::Tile;
use vello_common::util::RectExt;

#[derive(Debug)]
struct RecordedLayer {
    /// The command list that contains this layer's opening command.
    ///
    /// `None` means the root command list; `Some(id)` means the command list
    /// for the filter layer with that id.
    cmd_filter_layer_id: Option<usize>,
    push_cmd_idx: usize,
    content_bbox: RectU16,
    kind: RecordedLayerKind,
}

#[derive(Debug)]
enum RecordedLayerKind {
    Regular,
    /// Filter layers get a stable id while recording. The id is the index into
    /// [`CommandRecorder::filter_layers`].
    Filter {
        id: usize,
    },
}

#[derive(Debug)]
pub(crate) struct RecordedFilterLayer {
    /// Commands recorded while this filter layer is active.
    pub(crate) cmds: Vec<RenderCmd>,
    pub(crate) filter: Filter,
    pub(crate) transform: Affine,
    /// Filter-specific output expansion, in layer coordinates.
    expansion: Rect,
    /// Position of the unexpanded source content inside this layer's pixmap.
    pub(crate) source_origin: (u16, u16),
    /// Bounds of the commands recorded into this layer, before filter expansion.
    pub(crate) content_bbox: RectU16,
    /// Bounds of the pixmap rendered for this layer, after filter expansion and
    /// tile snapping.
    pub(crate) bbox: RectU16,
}

/// A small pool for reusing `Vec<RenderCmd>` allocations across multiple
/// frames for filter layers.
#[derive(Debug, Default)]
struct CommandPool {
    cmds: Vec<Vec<RenderCmd>>,
}

impl CommandPool {
    /// Takes a command vector from the pool, or creates a fresh empty vector.
    fn take(&mut self) -> Vec<RenderCmd> {
        self.cmds.pop().unwrap_or_default()
    }

    /// Returns a command vector to the pool after clearing its contents.
    fn submit(&mut self, mut cmds: Vec<RenderCmd>) {
        cmds.clear();
        self.cmds.push(cmds);
    }
}

/// Records root commands and filter-layer command lists while tracking layer
/// content bounds.
#[derive(Debug, Default)]
pub(crate) struct CommandRecorder {
    cmds: Vec<RenderCmd>,
    filter_layers: Vec<RecordedFilterLayer>,
    cmd_pool: CommandPool,
    /// Stack of active filter layer ids. The top selects which command list new
    /// commands are recorded into.
    active_filter_layer_stack: Vec<usize>,
    /// Stack of open regular and filter layers, used to patch content bounds
    /// back into the command that opened the layer when it is popped.
    layer_stack: Vec<RecordedLayer>,
}

impl CommandRecorder {
    /// Creates an empty recorder.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns commands recorded into the root command list.
    pub(crate) fn root_cmds(&self) -> &[RenderCmd] {
        &self.cmds
    }

    /// Returns recorded filter layers indexed by filter layer id.
    pub(crate) fn filter_layers(&self) -> &[RecordedFilterLayer] {
        &self.filter_layers
    }

    /// Returns whether there are currently unpopped layers.
    pub(crate) fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Clears recorded commands and returns filter-layer command vectors to the
    /// pool.
    pub(crate) fn reset(&mut self) {
        self.cmds.clear();
        for layer in self.filter_layers.drain(..) {
            self.cmd_pool.submit(layer.cmds);
        }
        self.active_filter_layer_stack.clear();
        self.layer_stack.clear();
    }

    /// Records a fill into the active command list and folds its bounds into
    /// the current open layer, if any.
    pub(crate) fn record_fill(
        &mut self,
        strip_range: core::ops::Range<usize>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        thread_idx: u8,
        content_bbox: RectU16,
    ) {
        self.active_cmds_mut().push(RenderCmd::Fill {
            thread_idx,
            strip_range,
            paint,
            blend_mode,
            mask,
        });
        self.include_content_bbox(content_bbox);
    }

    /// Records the start of a regular blend/opacity/mask/clip layer.
    pub(crate) fn push_layer(
        &mut self,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
    ) {
        let cmd_filter_layer_id = self.active_filter_layer_id();
        let push_cmd_idx = self.push_render_cmd(RenderCmd::PushLayer {
            blend_mode,
            opacity,
            mask,
            clip,
            content_bbox: RectU16::INVERTED,
        });
        self.layer_stack.push(RecordedLayer {
            cmd_filter_layer_id,
            push_cmd_idx,
            content_bbox: RectU16::INVERTED,
            kind: RecordedLayerKind::Regular,
        });
    }

    /// Records a filter layer composite command in the parent layer, then makes
    /// the new filter layer's command list active.
    pub(crate) fn push_filter_layer(
        &mut self,
        filter: Filter,
        transform: Affine,
        expansion: Rect,
        source_origin: (u16, u16),
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
    ) {
        let parent_filter_layer_id = self.active_filter_layer_id();
        let filter_layer_id = self.filter_layers.len();
        let push_cmd_idx = self.push_render_cmd(RenderCmd::CompositeFilterLayer {
            id: filter_layer_id,
            bbox: RectU16::INVERTED,
            src_x: 0,
            src_y: 0,
            blend_mode,
            opacity,
            mask,
            clip,
        });
        let cmds = self.cmd_pool.take();
        self.filter_layers.push(RecordedFilterLayer {
            cmds,
            filter,
            transform,
            expansion,
            source_origin,
            content_bbox: RectU16::INVERTED,
            bbox: RectU16::INVERTED,
        });
        self.active_filter_layer_stack.push(filter_layer_id);
        self.layer_stack.push(RecordedLayer {
            cmd_filter_layer_id: parent_filter_layer_id,
            push_cmd_idx,
            content_bbox: RectU16::INVERTED,
            kind: RecordedLayerKind::Filter {
                id: filter_layer_id,
            },
        });
    }

    /// Pops the current layer, patches its recorded content bounds, and returns
    /// the layer kind that was popped.
    pub(crate) fn pop_layer(&mut self) -> PoppedLayer {
        let layer = self.layer_stack.pop().expect("layer stack underflow");
        let content_bbox = layer.content_bbox;
        let popped = match layer.kind {
            RecordedLayerKind::Regular => {
                self.set_push_layer_content_bbox(
                    layer.cmd_filter_layer_id,
                    layer.push_cmd_idx,
                    content_bbox,
                );
                self.include_content_bbox(content_bbox);
                self.active_cmds_mut().push(RenderCmd::PopLayer);
                PoppedLayer::Regular
            }
            RecordedLayerKind::Filter {
                id: filter_layer_id,
            } => {
                let popped = self
                    .active_filter_layer_stack
                    .pop()
                    .expect("filter stack underflow");
                assert_eq!(popped, filter_layer_id, "filter layer stack mismatch");
                self.set_filter_layer_bbox(
                    filter_layer_id,
                    layer.cmd_filter_layer_id,
                    layer.push_cmd_idx,
                    content_bbox,
                );
                PoppedLayer::Filter
            }
        };
        popped
    }

    fn active_filter_layer_id(&self) -> Option<usize> {
        self.active_filter_layer_stack.last().copied()
    }

    fn active_cmds_mut(&mut self) -> &mut Vec<RenderCmd> {
        if let Some(id) = self.active_filter_layer_id() {
            &mut self.filter_layers[id].cmds
        } else {
            &mut self.cmds
        }
    }

    fn filter_layer_cmds_mut(&mut self, filter_layer_id: Option<usize>) -> &mut Vec<RenderCmd> {
        if let Some(id) = filter_layer_id {
            &mut self.filter_layers[id].cmds
        } else {
            &mut self.cmds
        }
    }

    fn push_render_cmd(&mut self, cmd: RenderCmd) -> usize {
        let cmds = self.active_cmds_mut();
        let idx = cmds.len();
        cmds.push(cmd);
        idx
    }

    fn set_push_layer_content_bbox(
        &mut self,
        filter_layer_id: Option<usize>,
        push_cmd_idx: usize,
        content_bbox: RectU16,
    ) {
        match &mut self.filter_layer_cmds_mut(filter_layer_id)[push_cmd_idx] {
            RenderCmd::PushLayer {
                content_bbox: bbox, ..
            } => *bbox = content_bbox,
            _ => unreachable!("layer stack referenced a non-layer command"),
        }
    }

    fn set_filter_layer_bbox(
        &mut self,
        filter_layer_id: usize,
        parent_filter_layer_id: Option<usize>,
        composite_cmd_idx: usize,
        bbox: RectU16,
    ) {
        self.filter_layers[filter_layer_id].content_bbox = bbox;
        let expansion = self.filter_layers[filter_layer_id].expansion;
        let source_origin = self.filter_layers[filter_layer_id].source_origin;
        let render_bbox = snap_bbox_to_tile(expand_bbox(bbox, expansion));
        let (output_bbox, src_x, src_y) = shift_bbox_to_parent(render_bbox, source_origin);
        self.filter_layers[filter_layer_id].bbox = render_bbox;
        match &mut self.filter_layer_cmds_mut(parent_filter_layer_id)[composite_cmd_idx] {
            RenderCmd::CompositeFilterLayer {
                bbox,
                src_x: cmd_src_x,
                src_y: cmd_src_y,
                ..
            } => {
                *bbox = output_bbox;
                *cmd_src_x = src_x;
                *cmd_src_y = src_y;
            }
            _ => unreachable!("filter layer stack referenced a non-filter command"),
        }
        self.include_content_bbox(output_bbox);
    }

    fn include_content_bbox(&mut self, bbox: RectU16) {
        if bbox.is_empty() {
            return;
        }
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.content_bbox.union(bbox);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoppedLayer {
    Regular,
    Filter,
}

pub(crate) fn expansion_left_top(expansion: Rect) -> (u16, u16) {
    let (left, top, _, _) = expansion_padding(expansion);
    (left, top)
}

pub(crate) fn expansion_padding(expansion: Rect) -> (u16, u16, u16, u16) {
    let expansion = expansion.snap_to_tile_coordinates();
    let left = (-expansion.x0).max(0.0) as u16;
    let top = (-expansion.y0).max(0.0) as u16;
    let right = expansion.x1.max(0.0) as u16;
    let bottom = expansion.y1.max(0.0) as u16;
    (left, top, right, bottom)
}

fn expand_bbox(bbox: RectU16, expansion: Rect) -> RectU16 {
    if bbox.is_empty() {
        return bbox;
    }

    let (left, top, right, bottom) = expansion_padding(expansion);
    RectU16::new(
        bbox.x0.saturating_sub(left),
        bbox.y0.saturating_sub(top),
        bbox.x1.saturating_add(right),
        bbox.y1.saturating_add(bottom),
    )
}

fn snap_bbox_to_tile(bbox: RectU16) -> RectU16 {
    if bbox.is_empty() {
        return bbox;
    }

    RectU16::new(
        (bbox.x0 / Tile::WIDTH) * Tile::WIDTH,
        (bbox.y0 / Tile::HEIGHT) * Tile::HEIGHT,
        bbox.x1
            .checked_next_multiple_of(Tile::WIDTH)
            .unwrap_or(u16::MAX),
        bbox.y1
            .checked_next_multiple_of(Tile::HEIGHT)
            .unwrap_or(u16::MAX),
    )
}

fn shift_bbox_to_parent(bbox: RectU16, origin: (u16, u16)) -> (RectU16, u16, u16) {
    let (left, top) = origin;
    let src_x = left.saturating_sub(bbox.x0);
    let src_y = top.saturating_sub(bbox.y0);
    (
        RectU16::new(
            bbox.x0.saturating_sub(left),
            bbox.y0.saturating_sub(top),
            bbox.x1.saturating_sub(left),
            bbox.y1.saturating_sub(top),
        ),
        src_x,
        src_y,
    )
}
