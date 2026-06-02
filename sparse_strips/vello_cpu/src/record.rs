// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

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
//! - They always need to be rendered separately, independently from the parent layer
//! command stream. This is because spatial filters might require sampling neighboring pixels,
//! so the whole filter layer needs to be rendered as a whole before you can apply the filter and
//! then composite it into the parent layer. Therefore, commands for filter layers cannot simply
//! be inlined into the parent layer stream (this is what was done in previous versions of Vello CPU,
//! and resulted in incredibly complex and fragile code).
//!
//! - Filter layers might have a different dimension than the main viewport. If you apply a big
//! blur filter, you might have to render shapes that would normally be culled away because they
//! exceed the viewport.
//!
//! If we decided to skip step 1) and do 2) directly, we would need to store and reuse instances
//! of [`CommandBucketer`] for the root and each filter layer, which is especially cumbersome because
//! conceptually, a command bucketer is a 2D array. This makes it very difficult to resize and reuse
//! it while having the ability to retain and reuse inner allocations. By first recording all commands
//! and their strips into a single [`Vec<RenderCmd>`], we can basically represent the commands of
//! a single layer with just one flat buffer, making it much easier to reuse them across frames. It
//! also allows us to collect useful metadata (e.g. the bounding box of a layer) and make appropriate
//! decisions about how to render them (for example, where to place a filter layer and what size to
//! allocate for it).

use crate::coarse::{LayerClip, RenderCmd};
use crate::kurbo::{Affine, Rect};
use crate::peniko::BlendMode;
use alloc::vec::Vec;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::tile::Tile;
use vello_common::util::RectExt;

/// Metadata about a layer.
///
/// This is mainly needed because we want the layer metadata to be in the `PushLayer` command
/// instead of the `PopLayer` one, so it can be conveniently accessed when doing coarse rasterization.
/// However, we only actually know the bounding box of a layer once we've finished rendering it,
/// so we can only update this information once we actually pop the layer.
#[derive(Debug)]
struct LayerMetadata {
    /// The filter layer id of the layer this layer is composited into.
    cmd_filter_layer_id: Option<usize>,
    /// The index of the `PushLayer` render command of this layer.
    push_cmd_idx: usize,
    /// A conservative bounding box of the layer.
    bbox: RectU16,
    kind: LayerKind,
}

#[derive(Debug)]
enum LayerKind {
    Regular,
    Filter { id: usize },
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
    fn take(&mut self) -> Vec<RenderCmd> {
        self.cmds.pop().unwrap_or_default()
    }

    fn submit(&mut self, mut cmds: Vec<RenderCmd>) {
        cmds.clear();
        self.cmds.push(cmds);
    }
}


#[derive(Debug, Default)]
pub(crate) struct CommandRecorder {
    /// The commands of the root layer.
    pub(crate) root_cmds: Vec<RenderCmd>,
    /// Recorded filter layers, indexed by their ID.
    pub(crate) filter_layers: Vec<RecordedFilterLayer>,
    cmd_pool: CommandPool,
    /// Stack of currently active filter layers.
    active_filter_layer_stack: Vec<usize>,
    /// Stack of currently active layers, with their metadata.
    layer_stack: Vec<LayerMetadata>,
}

impl CommandRecorder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    pub(crate) fn reset(&mut self) {
        self.root_cmds.clear();
        for layer in self.filter_layers.drain(..) {
            self.cmd_pool.submit(layer.cmds);
        }
        self.active_filter_layer_stack.clear();
        self.layer_stack.clear();
    }

    pub(crate) fn record_fill(
        &mut self,
        strip_range: core::ops::Range<usize>,
        strips: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        thread_idx: u8,
    ) {
        self.active_cmds_mut().push(RenderCmd::Fill {
            thread_idx,
            strip_range,
            paint,
            blend_mode,
            mask,
        });

        if self.layer_stack.is_empty() {
            return;
        }

        let content_bbox = strip_bbox(strips);
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
        self.layer_stack.push(LayerMetadata {
            cmd_filter_layer_id,
            push_cmd_idx,
            bbox: RectU16::INVERTED,
            kind: LayerKind::Regular,
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
        self.layer_stack.push(LayerMetadata {
            cmd_filter_layer_id: parent_filter_layer_id,
            push_cmd_idx,
            bbox: RectU16::INVERTED,
            kind: LayerKind::Filter {
                id: filter_layer_id,
            },
        });
    }

    /// Pops the current layer, patches its recorded content bounds, and returns
    /// the layer kind that was popped.
    pub(crate) fn pop_layer(&mut self) -> PoppedLayer {
        let layer = self.layer_stack.pop().expect("layer stack underflow");
        let content_bbox = layer.bbox;
        let popped = match layer.kind {
            LayerKind::Regular => {
                self.set_push_layer_content_bbox(
                    layer.cmd_filter_layer_id,
                    layer.push_cmd_idx,
                    content_bbox,
                );
                self.include_content_bbox(content_bbox);
                self.active_cmds_mut().push(RenderCmd::PopLayer);
                PoppedLayer::Regular
            }
            LayerKind::Filter {
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
            &mut self.root_cmds
        }
    }

    fn filter_layer_cmds_mut(&mut self, filter_layer_id: Option<usize>) -> &mut Vec<RenderCmd> {
        if let Some(id) = filter_layer_id {
            &mut self.filter_layers[id].cmds
        } else {
            &mut self.root_cmds
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
            layer.bbox.union(bbox);
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

fn strip_bbox(strips: &[Strip]) -> RectU16 {
    let mut bbox = RectU16::INVERTED;

    // Need at least one strip (and the sentinel one).
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
        let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
        let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
        // TODO: We likely have a couple of other places that do the same
        // calculation, maybe extract into a method.
        let strip_width = next_col.saturating_sub(col) as u16;
        let strip_x1 = strip.x.saturating_add(strip_width);

        if strip_width > 0 {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() && strip_x1 < next_strip.x {
            bbox.union(RectU16::new(strip_x1, row_y, next_strip.x, row_y1));
        }
    }

    bbox
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentinel(y: u16, alpha_idx: u32) -> Strip {
        Strip::new(u16::MAX, y, alpha_idx, false)
    }

    #[test]
    fn empty_strip_bbox() {
        let strips = [sentinel(0, 0), sentinel(0, 0)];

        assert_eq!(strip_bbox(&strips), RectU16::INVERTED);
    }

    #[test]
    fn single_strip_bbox() {
        let strips = [
            Strip::new(8, 4, 0, false),
            sentinel(4, u32::from(Tile::HEIGHT) * 4),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(8, 4, 12, 8));
    }

    #[test]
    fn strip_with_fill_bbox() {
        let strips = [
            Strip::new(4, 0, 0, false),
            Strip::new(20, 0, u32::from(Tile::HEIGHT) * 4, true),
            sentinel(0, u32::from(Tile::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(4, 0, 24, 4));
    }

    #[test]
    fn strips_with_multiple_rows_bbox() {
        let strips = [
            Strip::new(12, 0, 0, false),
            sentinel(0, u32::from(Tile::HEIGHT) * 4),
            Strip::new(4, 8, u32::from(Tile::HEIGHT) * 4, false),
            sentinel(8, u32::from(Tile::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(4, 0, 16, 12));
    }
}
