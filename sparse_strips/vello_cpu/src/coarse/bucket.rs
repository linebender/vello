// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{
    BlendAttrs, BlendFill, FillAttrs, Fill, FilterLayerAttrs, FilterLayer, RenderCmd,
};
use super::depth::DepthState;
use crate::peniko::BlendMode;
use crate::record::{RecordedCmd, RecordedFilterLayer};
use crate::util::{Span, bbox_relative_to, snap_bbox_to_tile};
use alloc::vec;
use alloc::vec::Vec;
use std::ops::Range;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::strip::Strip;
use vello_common::tile::Tile;
use vello_common::util::{Clear, RetainVec};

#[derive(Debug, Default)]
pub(crate) struct RowCommands {
    /// Normal commands rendered in back-to-front with depth buffer read.
    pub(crate) cmds: Vec<RenderCmd>,
    /// Opaque fill commands rendered front-to-back with depth buffer read and write.
    ///
    /// `alpha_idx` is always `None` for these commands.
    pub(crate) depth_writes: Vec<Fill>,
    /// A coarse span of all pixels that might be touched in that row.
    coarse_span: Option<Span>,
    depth: DepthState,
    pub(super) layer_depth: usize,
}

impl RowCommands {
    pub(super) fn new() -> Self {
        Self {
            cmds: Vec::new(),
            depth_writes: Vec::new(),
            coarse_span: None,
            depth: DepthState::default(),
            layer_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.cmds.clear();
        self.depth_writes.clear();
        self.coarse_span = None;
        self.depth.reset();
        self.layer_depth = 0;
    }

    pub(super) fn push_cmd(&mut self, cmd: RenderCmd) {
        if let Some(span) = cmd.span() {
            self.include_span(span);
        }
        self.cmds.push(cmd);
    }

    pub(super) fn push_layer(&mut self) {
        self.cmds.push(RenderCmd::PushLayer);
        self.layer_depth += 1;
    }

    pub(super) fn pop_layer(
        &mut self,
        span: Span,
        mask_idx: Option<u32>,
        opacity: f32,
        blend_attrs_idx: u32,
    ) {
        if let Some(mask_idx) = mask_idx {
            self.cmds.push(RenderCmd::Mask(mask_idx));
        }
        if opacity != 1.0 {
            self.cmds.push(RenderCmd::Opacity(opacity));
        }
        self.cmds.push(RenderCmd::BlendFill(BlendFill::new(
            span,
            None,
            blend_attrs_idx,
        )));
        self.pop_buf();
    }

    pub(super) fn push_layer_props(&mut self, mask_idx: Option<u32>, opacity: f32) {
        if let Some(mask_idx) = mask_idx {
            self.cmds.push(RenderCmd::Mask(mask_idx));
        }
        if opacity != 1.0 {
            self.cmds.push(RenderCmd::Opacity(opacity));
        }
    }

    pub(super) fn push_blend_fill(
        &mut self,
        span: Span,
        alpha_idx: Option<u32>,
        blend_attrs_idx: u32,
    ) {
        self.push_cmd(RenderCmd::BlendFill(BlendFill::new(
            span,
            alpha_idx,
            blend_attrs_idx,
        )));
    }

    pub(super) fn pop_buf(&mut self) {
        self.cmds.push(RenderCmd::PopBuf);
        self.layer_depth -= 1;
    }

    pub(super) fn push_depth_write(&mut self, cmd: Fill, draw_id: u32) {
        self.include_span(cmd.span);
        self.depth.include_span(cmd.span, draw_id);
        self.depth_writes.push(cmd);
    }

    pub(crate) fn coarse_span(&self) -> Option<Span> {
        self.coarse_span
    }

    pub(crate) fn can_skip_depth(&self, span: Span, draw_id: u32) -> bool {
        self.depth.can_skip(span, draw_id)
    }

    fn include_span(&mut self, span: Span) {
        if let Some(bounds) = &mut self.coarse_span {
            bounds.extend(span);
        } else {
            self.coarse_span = Some(span);
        }
    }
}

impl Clear for RowCommands {
    fn clear(&mut self) {
        Self::clear(self);
    }
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    pub(super) clip_bboxes: Vec<RectU16>,
    pub(super) rows: RetainVec<RowCommands>,
    pub(super) attrs: Vec<FillAttrs>,
    pub(super) blend_attrs: Vec<BlendAttrs>,
    pub(super) filter_attrs: Vec<FilterLayerAttrs>,
    pub(super) masks: Vec<Mask>,
    pub(super) active_layers: Vec<ActiveLayer>,
    pub(super) next_draw_id: u32,
}

impl CommandBucketer {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let full_clip_bbox = Self::full_clip_bbox(width, height);
        let num_rows = usize::from(full_clip_bbox.height() / Tile::HEIGHT);
        Self {
            clip_bboxes: vec![full_clip_bbox],
            rows: RetainVec::with_len(num_rows, RowCommands::new),
            attrs: Vec::new(),
            blend_attrs: Vec::new(),
            filter_attrs: Vec::new(),
            masks: Vec::new(),
            active_layers: Vec::new(),
            next_draw_id: 1,
        }
    }

    fn full_clip_bbox(width: u16, height: u16) -> RectU16 {
        snap_bbox_to_tile(RectU16::new(0, 0, width, height))
    }

    fn bbox_span(bbox: RectU16) -> Span {
        Span::new(bbox.x0, bbox.x1.saturating_sub(bbox.x0))
    }

    pub(crate) fn rows(&self) -> &[RowCommands] {
        self.rows.as_slice()
    }

    pub(crate) fn width(&self) -> u16 {
        // TODO: Should be + 1?
        self.clip_bboxes[0].width()
    }

    pub(crate) fn attrs(&self) -> &[FillAttrs] {
        &self.attrs
    }

    pub(crate) fn blend_attrs(&self) -> &[BlendAttrs] {
        &self.blend_attrs
    }

    pub(crate) fn filter_attrs(&self) -> &[FilterLayerAttrs] {
        &self.filter_attrs
    }

    pub(crate) fn masks(&self) -> &[Mask] {
        &self.masks
    }

    pub(crate) fn reset(&mut self, width: u16, height: u16) {
        let full_clip_bbox = Self::full_clip_bbox(width, height);
        let num_rows = usize::from(full_clip_bbox.height() / Tile::HEIGHT);
        self.rows.clear();
        self.rows.resize_with(num_rows, RowCommands::new);
        self.attrs.clear();
        self.blend_attrs.clear();
        self.filter_attrs.clear();
        self.masks.clear();
        self.active_layers.clear();
        self.next_draw_id = 1;
        self.clip_bboxes.truncate(1);
        self.clip_bboxes[0] = full_clip_bbox;
    }

    pub(crate) fn bucket_commands(
        &mut self,
        cmds: &[RecordedCmd],
        filter_layers: &[RecordedFilterLayer],
        strips: &[Strip],
        encoded_paints: &[EncodedPaint],
        // When rendering filter layers, we always anchor them so that the top-left of the bounding
        // box lands at (0, 0), even if the bounding box's top-left is for example at (200, 200).
        // Therefore, we need to keep track of this offset so that for example paints know that
        // they should actually be sampled at (200, 200) instead of (0, 0).
        pixmap_origin: (u16, u16),
    ) {
        assert_eq!(pixmap_origin.0 % Tile::WIDTH, 0);
        assert_eq!(pixmap_origin.1 % Tile::HEIGHT, 0);

        for cmd in cmds {
            match cmd {
                RecordedCmd::Fill {
                    thread_idx,
                    strip_range,
                    paint,
                    blend_mode,
                    mask,
                } => {
                    let draw_id = self.next_draw_id();
                    let attrs = FillAttrs {
                        paint: paint.clone(),
                        blend_mode: *blend_mode,
                        mask: mask.clone(),
                        draw_id,
                        thread_idx: *thread_idx,
                        pixmap_origin,
                    };
                    self.generate_fill(&strips[strip_range.clone()], &attrs, encoded_paints);
                }
                RecordedCmd::PushLayer {
                    blend_mode,
                    opacity,
                    mask,
                    clip,
                    ..
                } => self.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone()),
                RecordedCmd::CompositeFilterLayer {
                    id,
                    blend_mode,
                    opacity,
                    mask,
                    clip,
                } => {
                    let placement = filter_layers[*id].placement;
                    let bbox = bbox_relative_to(placement.composite_bbox, pixmap_origin);
                    let needs_layer = *blend_mode != BlendMode::default()
                        || *opacity != 1.0
                        || mask.is_some()
                        || clip.is_some();
                    if needs_layer {
                        self.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone());
                    }
                    self.generate_filter_layer(*id, bbox, placement.src_origin());
                    if needs_layer {
                        self.pop_layer(strips, pixmap_origin);
                    }
                }
                RecordedCmd::PopLayer => self.pop_layer(strips, pixmap_origin),
            }
        }
    }

    fn next_draw_id(&mut self) -> u32 {
        let draw_id = self.next_draw_id;
        self.next_draw_id = self.next_draw_id + 1;
        draw_id
    }

    pub(crate) fn push_layer(
        &mut self,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
    ) {
        let parent_bbox = *self.clip_bboxes.last().unwrap();
        let bbox = clip
            .as_ref()
            .map(|clip| clip.bbox.intersect(parent_bbox))
            .unwrap_or(parent_bbox);
        if clip.is_some() {
            self.clip_bboxes.push(bbox);
        }

        self.active_layers.push(ActiveLayer {
            mask,
            blend_mode,
            opacity,
            clip,
            span: Self::bbox_span(bbox),
            occupied_rows: Vec::new(),
        });
        if blend_mode.is_destructive() {
            self.ensure_layer_rows(bbox);
        }
    }

    pub(crate) fn pop_layer(&mut self, strips: &[Strip], pixmap_origin: (u16, u16)) {
        let mut layer = self.active_layers.pop().unwrap();
        let mask_idx = layer.mask.as_ref().map(|mask| {
            let idx = self.masks.len() as u32;
            self.masks.push(mask.clone());
            idx
        });
        let opacity = layer.opacity;
        let blend_mode = layer.blend_mode;
        let full_width = self.width();
        if let Some(clip) = layer.clip {
            let blend_attrs_idx = self.blend_attrs.len() as u32;
            self.blend_attrs.push(BlendAttrs {
                blend_mode,
                thread_idx: clip.thread_idx,
            });
            self.clip_bboxes.pop();
            let clip_strips = &strips[clip.strip_range];

            let mut occupied_rows = vec![false; self.rows.len()];
            for &row_idx in &layer.occupied_rows {
                occupied_rows[row_idx] = true;
                let row = &mut self.rows[row_idx];
                debug_assert_eq!(
                    row.layer_depth,
                    self.active_layers.len() + 1,
                    "occupied row must still be inside the popped layer"
                );
                row.push_layer_props(mask_idx, opacity);
            }

            self.generate(
                clip_strips,
                pixmap_origin,
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_fill(fill, None, blend_attrs_idx);
                    }
                },
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_fill(
                            fill.span,
                            Some(fill.alpha_idx),
                            blend_attrs_idx,
                        );
                    }
                },
            );

            for row_idx in layer.occupied_rows.drain(..) {
                self.rows[row_idx].pop_buf();
            }
        } else {
            let blend_attrs_idx = self.blend_attrs.len() as u32;
            self.blend_attrs.push(BlendAttrs {
                blend_mode,
                thread_idx: 0,
            });
            let blend_span = if blend_mode.is_destructive() {
                layer.span
            } else {
                Span::new(0, full_width)
            };
            for row_idx in layer.occupied_rows.drain(..) {
                let row = &mut self.rows[row_idx];
                debug_assert_eq!(
                    row.layer_depth,
                    self.active_layers.len() + 1,
                    "occupied row must still be inside the popped layer"
                );
                row.pop_layer(blend_span, mask_idx, opacity, blend_attrs_idx);
            }
        }
    }

    fn ensure_layer_rows(&mut self, bbox: RectU16) {
        if bbox.is_empty() {
            return;
        }

        let row_start = usize::from(bbox.y0 / Tile::HEIGHT);
        let row_end = usize::from(bbox.y1.div_ceil(Tile::HEIGHT)).min(self.rows.len());
        for row_idx in row_start..row_end {
            self.ensure_row_layers(row_idx);
        }
    }

    #[inline(always)]
    pub(super) fn ensure_row_layers(&mut self, row_idx: usize) {
        let layer_depth = self.rows[row_idx].layer_depth;
        if layer_depth == self.active_layers.len() {
            return;
        }

        for layer_idx in layer_depth..self.active_layers.len() {
            self.rows[row_idx].push_layer();
            self.active_layers[layer_idx].occupied_rows.push(row_idx);
        }
    }

    pub(crate) fn generate_filter_layer(
        &mut self,
        filter_layer_id: usize,
        bbox: RectU16,
        src_origin: (u16, u16),
    ) {
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let src_bbox = bbox;
        let bbox = bbox.intersect(clip_bbox);
        if bbox.is_empty() {
            return;
        }

        let draw_id = self.next_draw_id();
        let span = Self::bbox_span(bbox);
        let filter_attrs_idx = self.filter_attrs.len() as u32;
        self.filter_attrs.push(FilterLayerAttrs {
            id: filter_layer_id,
            draw_id,
            dst_bbox: bbox,
            src_origin: (
                src_origin.0 + span.pixel_x().saturating_sub(src_bbox.x0),
                src_origin.1 + (bbox.y0 - src_bbox.y0),
            ),
        });
        let row_start = usize::from(bbox.y0 / Tile::HEIGHT);
        let row_end = usize::from(bbox.y1.div_ceil(Tile::HEIGHT)).min(self.rows.len());
        for row_idx in row_start..row_end {
            self.ensure_row_layers(row_idx);
            let row_y = row_idx as u16 * Tile::HEIGHT;
            let row_y1 = row_y.saturating_add(Tile::HEIGHT);
            if row_y1 <= bbox.y0 || row_y >= bbox.y1 {
                continue;
            }
            self.rows[row_idx].push_cmd(RenderCmd::FilterLayer(FilterLayer {
                span,
                attrs_idx: filter_attrs_idx,
            }));
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ActiveLayer {
    pub(crate) mask: Option<Mask>,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) clip: Option<LayerClip>,
    pub(crate) span: Span,
    pub(crate) occupied_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerClip {
    pub(crate) strip_range: Range<usize>,
    pub(crate) thread_idx: u8,
    pub(crate) bbox: RectU16,
}
