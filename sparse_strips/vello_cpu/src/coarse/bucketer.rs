// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{Cmd, FillAttrs, FilterLayerCmd};
use super::layer::{ActiveLayer, LayerClip};
use super::row::RowCommands;
use crate::peniko::BlendMode;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::strip::Strip;
use vello_common::tile::Tile;

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    pub(super) clip_bboxes: Vec<RectU16>,
    pub(super) rows: Vec<RowCommands>,
    pub(super) attrs: Vec<FillAttrs>,
    pub(super) active_layers: Vec<ActiveLayer>,
    pub(super) next_path_id: u32,
}

impl CommandBucketer {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let full_clip_bbox = Self::full_clip_bbox(width, height);
        let num_rows = usize::from(full_clip_bbox.height() / Tile::HEIGHT);
        Self {
            clip_bboxes: vec![full_clip_bbox],
            rows: (0..num_rows).map(|_| RowCommands::new()).collect(),
            attrs: Vec::new(),
            active_layers: Vec::new(),
            next_path_id: 1,
        }
    }

    fn full_clip_bbox(width: u16, height: u16) -> RectU16 {
        RectU16::new(
            0,
            0,
            Self::ceil_to_tile_width(width),
            Self::ceil_to_tile_height(height),
        )
    }

    pub(super) fn ceil_to_tile_width(width: u16) -> u16 {
        width
            .checked_next_multiple_of(Tile::WIDTH)
            .unwrap_or(u16::MAX)
    }

    fn ceil_to_tile_height(height: u16) -> u16 {
        height
            .checked_next_multiple_of(Tile::HEIGHT)
            .unwrap_or(u16::MAX)
    }

    pub(crate) fn rows(&self) -> &[RowCommands] {
        &self.rows
    }

    pub(crate) fn width(&self) -> u16 {
        // TODO: Should be + 1?
        self.clip_bboxes[0].width()
    }

    pub(crate) fn attrs(&self) -> &[FillAttrs] {
        &self.attrs
    }

    pub(crate) fn reset(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
        self.attrs.clear();
        self.active_layers.clear();
        self.next_path_id = 1;
        self.clip_bboxes.truncate(1);
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
            bbox,
            occupied_rows: Vec::new(),
        });
        if blend_mode.is_destructive() {
            self.ensure_layer_rows(bbox);
        }
    }

    pub(crate) fn pop_layer(&mut self, strips: &[Strip]) {
        let mut layer = self.active_layers.pop().unwrap();
        let mask = layer.mask.clone();
        let opacity = layer.opacity;
        let blend_mode = layer.blend_mode;
        let full_width = self.width();
        if let Some(clip) = layer.clip {
            self.clip_bboxes.pop();
            let clip_strips = &strips[clip.strip_range];

            let mut occupied_rows = vec![false; self.rows.len()];
            for &row_idx in &layer.occupied_rows {
                occupied_rows[row_idx] = true;
                let row = &mut self.rows[row_idx];
                debug_assert_eq!(row.layer_depth, self.active_layers.len() + 1);
                row.push_layer_props(mask.as_ref(), opacity);
            }

            self.generate(
                clip_strips,
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_fill(fill, blend_mode, full_width);
                    }
                },
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_alpha_fill(
                            fill,
                            blend_mode,
                            clip.thread_idx,
                            full_width,
                        );
                    }
                },
            );

            for row_idx in layer.occupied_rows.drain(..) {
                self.rows[row_idx].pop_buf();
            }
        } else {
            let (blend_x, blend_width) = if blend_mode.is_destructive() {
                (layer.bbox.x0, layer.bbox.width())
            } else {
                (0, full_width)
            };
            for row_idx in layer.occupied_rows.drain(..) {
                let row = &mut self.rows[row_idx];
                debug_assert_eq!(row.layer_depth, self.active_layers.len() + 1);
                row.pop_layer(blend_x, blend_width, mask.as_ref(), opacity, blend_mode);
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
        layer_id: usize,
        bbox: RectU16,
        src_origin: (u16, u16),
    ) {
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let src_bbox = bbox;
        let bbox = bbox.intersect(clip_bbox);
        if bbox.is_empty() {
            return;
        }

        let path_id = self.next_path_id;
        self.next_path_id = self
            .next_path_id
            .checked_add(1)
            .expect("row-bucket path ID overflow");
        let full_width = self.width();
        let row_start = usize::from(bbox.y0 / Tile::HEIGHT);
        let row_end = usize::from(bbox.y1.div_ceil(Tile::HEIGHT)).min(self.rows.len());
        for row_idx in row_start..row_end {
            self.ensure_row_layers(row_idx);
            let row_y = row_idx as u16 * Tile::HEIGHT;
            let row_y1 = row_y.saturating_add(Tile::HEIGHT);
            if row_y1 <= bbox.y0 || row_y >= bbox.y1 {
                continue;
            }
            let draw_y = row_y.max(bbox.y0);
            let draw_y1 = row_y1.min(bbox.y1);

            self.rows[row_idx].push_cmd(
                Cmd::FilterLayer(FilterLayerCmd {
                    x: bbox.x0,
                    width: bbox.width(),
                    layer_id,
                    path_id,
                    src_x: src_origin.0 + (bbox.x0 - src_bbox.x0),
                    src_y: src_origin.1 + (draw_y - src_bbox.y0),
                    dst_y_offset: (draw_y - row_y) as u8,
                    height: (draw_y1 - draw_y) as u8,
                }),
                full_width,
            );
        }
    }
}
