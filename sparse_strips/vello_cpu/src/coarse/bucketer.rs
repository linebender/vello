// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{
    BlendAttrs, FillAttrs, FilterLayerAttrs, FilterLayerCmd, FineCmd, RenderCmd, Span,
};
use super::layer::{ActiveLayer, LayerClip};
use super::row::RowCommands;
use crate::peniko::BlendMode;
use crate::record::RecordedFilterLayer;
use crate::util::bbox_relative_to;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::strip::Strip;
use vello_common::tile::Tile;
use vello_common::util::RetainVec;

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    pub(super) clip_bboxes: Vec<RectU16>,
    pub(super) rows: RetainVec<RowCommands>,
    pub(super) attrs: Vec<FillAttrs>,
    pub(super) blend_attrs: Vec<BlendAttrs>,
    pub(super) filter_attrs: Vec<FilterLayerAttrs>,
    pub(super) masks: Vec<Mask>,
    pub(super) active_layers: Vec<ActiveLayer>,
    pub(super) next_path_id: u32,
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

    fn bbox_span(bbox: RectU16) -> Span {
        let tile_x = bbox.x0 / Tile::WIDTH;
        let tile_x1 = bbox.x1.div_ceil(Tile::WIDTH);
        Span::new(tile_x, tile_x1.saturating_sub(tile_x))
    }

    fn ceil_to_tile_height(height: u16) -> u16 {
        height
            .checked_next_multiple_of(Tile::HEIGHT)
            .unwrap_or(u16::MAX)
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
        self.next_path_id = 1;
        self.clip_bboxes.truncate(1);
        self.clip_bboxes[0] = full_clip_bbox;
    }

    pub(crate) fn bucket_commands(
        &mut self,
        cmds: &[RenderCmd],
        filter_layers: &[RecordedFilterLayer],
        strips: &[Strip],
        encoded_paints: &[EncodedPaint],
        origin: (u16, u16),
    ) {
        let translated_strips = if origin == (0, 0) {
            Vec::new()
        } else {
            strips
                .iter()
                .map(|strip| translate_strip(*strip, origin))
                .collect::<Vec<_>>()
        };
        let strips = if origin == (0, 0) {
            strips
        } else {
            translated_strips.as_slice()
        };

        for cmd in cmds {
            match cmd {
                RenderCmd::Fill {
                    thread_idx,
                    strip_range,
                    paint,
                    blend_mode,
                    mask,
                } => {
                    self.generate_fill(
                        &strips[strip_range.clone()],
                        paint.clone(),
                        *blend_mode,
                        mask.clone(),
                        *thread_idx,
                        origin,
                        encoded_paints,
                    );
                }
                RenderCmd::PushLayer {
                    blend_mode,
                    opacity,
                    mask,
                    clip,
                    ..
                } => self.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone()),
                RenderCmd::CompositeFilterLayer {
                    id,
                    blend_mode,
                    opacity,
                    mask,
                    clip,
                } => {
                    let placement = filter_layers[*id].placement;
                    let bbox = bbox_relative_to(placement.composite_bbox, origin);
                    let needs_layer = *blend_mode != BlendMode::default()
                        || *opacity != 1.0
                        || mask.is_some()
                        || clip.is_some();
                    if needs_layer {
                        self.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone());
                    }
                    self.generate_filter_layer(*id, bbox, placement.src_origin());
                    if needs_layer {
                        self.pop_layer(strips);
                    }
                }
                RenderCmd::PopLayer => self.pop_layer(strips),
            }
        }
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

    pub(crate) fn pop_layer(&mut self, strips: &[Strip]) {
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
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_fill(
                            fill,
                            None,
                            blend_attrs_idx,
                            full_width,
                        );
                    }
                },
                |bucketer, row_idx, fill| {
                    if occupied_rows[row_idx] {
                        bucketer.rows[row_idx].push_blend_fill(
                            fill.span,
                            Some(fill.alpha_idx),
                            blend_attrs_idx,
                            full_width,
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
                Span::new(0, full_width / Tile::WIDTH)
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

        let path_id = self.next_path_id;
        self.next_path_id = self
            .next_path_id
            .checked_add(1)
            .expect("row-bucket path ID overflow");
        let full_width = self.width();
        let span = Self::bbox_span(bbox);
        let filter_attrs_idx = self.filter_attrs.len() as u32;
        self.filter_attrs.push(FilterLayerAttrs {
            id: filter_layer_id,
            path_id,
            src_x: src_origin.0 + span.pixel_x().saturating_sub(src_bbox.x0),
            src_y: src_origin.1 + (bbox.y0 - src_bbox.y0),
            y0: bbox.y0,
            y1: bbox.y1,
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
            self.rows[row_idx].push_cmd(
                FineCmd::FilterLayer(FilterLayerCmd {
                    span,
                    attrs_idx: filter_attrs_idx,
                }),
                full_width,
            );
        }
    }
}

fn translate_strip(strip: Strip, origin: (u16, u16)) -> Strip {
    let x = if strip.is_sentinel() {
        strip.x
    } else {
        strip.x.saturating_sub(origin.0)
    };
    Strip::new(
        x,
        strip.y.saturating_sub(origin.1),
        strip.alpha_idx(),
        strip.fill_gap(),
    )
}
