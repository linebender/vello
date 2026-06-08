// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{LayerFill, LayerFillAttrs, PaintFill, PaintFillAttrs, RenderCmd};
use super::depth::{DepthSegment, DepthState};
use crate::coarse::depth;
use crate::filter::context::FilterContext;
use crate::kurbo::{Affine, Vec2};
use crate::peniko::{BlendMode, Extend, ImageQuality, ImageSampler};
use crate::record::{LayerProps, RecordedCmd, RecordedLayer, RecordedLayerKind};
use crate::util::{Span, VecPool, bbox_relative_to, snap_bbox_to_tile_coordinates};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::ops::Range;
use vello_common::encode::{EncodedImage, EncodedPaint};
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageSource, IndexedPaint, Paint};
use vello_common::pixmap::Pixmap;
use vello_common::strip::Strip;
use vello_common::tile::Tile;
use vello_common::util::{Clear, RetainVec};

#[derive(Debug, Default)]
pub(crate) struct RowCmds {
    /// Normal commands rendered in back-to-front with depth buffer read.
    pub(crate) cmds: Vec<RenderCmd>,
    /// Opaque fill commands rendered front-to-back with depth buffer read and write.
    ///
    /// `alpha_idx` is always `None` for these commands.
    pub(crate) depth_writes: Vec<PaintFill>,
    /// A coarse span of all pixels that might be touched in that row.
    coarse_span: Option<Span>,
    /// State of the depth buffer for this row.
    depth: DepthState,
    /// Current layer depth.
    pub(super) layer_depth: usize,
}

impl RowCmds {
    pub(super) fn new() -> Self {
        Self::default()
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

    pub(super) fn push_buf(&mut self) {
        self.cmds.push(RenderCmd::PushBuf);

        self.layer_depth += 1;
    }

    pub(super) fn pop_buf(&mut self) {
        self.cmds.push(RenderCmd::PopBuf);

        self.layer_depth -= 1;
    }

    pub(super) fn push_depth_write(&mut self, cmd: PaintFill, draw_id: u32) {
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

impl Clear for RowCmds {
    fn clear(&mut self) {
        Self::clear(self);
    }
}

/// A bucketer that groups commands into strip-row-sized buckets.
#[derive(Debug)]
pub(crate) struct CommandBucketer {
    /// The viewport of the root layer we are currently bucketing.
    viewport: RectU16,
    /// The currently active stack of clip bboxes (from layer clips), tracked for two reasons:
    /// - So we can clamp fill commands to the bbox and avoid unnecessary rendering work.
    /// - In case we have a filter layer with a clip, it all happens in three steps:
    ///   1) We first push a new intermediate layer with the clip.
    ///   2) We composite the whole filter layer **cropped to the active clip bbox**. Note that
    ///      the filter layer iself is _not_ affected by the clip bbox, as filter layers should
    ///      always be fully rendered before applying clipping. However, by limiting the composition
    ///      to the coarse clip bbox we might be able to save a lot of work if only a small part
    ///      of the filter layer is visible.
    ///   3) Pop the intermediate layer, which will take care of doing the fine-grained clipping
    ///      (e.g. applying anti-aliasing from the clip path).
    pub(super) clip_bboxes: Vec<RectU16>,
    /// The actual render commands for each strip row.
    ///
    /// Since this is essentially a 2D-array, we use [`RetainVec`] to preserve inner allocations
    /// upon resetting.
    pub(super) rows: RetainVec<RowCmds>,
    pub(super) paint_fill_attrs: Vec<PaintFillAttrs>,
    pub(super) layer_fill_attrs: Vec<LayerFillAttrs>,
    pub(super) filter_paints: Vec<EncodedPaint>,
    /// Keeping track of currently active layers to enable lazy layer pushing.
    pub(super) active_layers: Vec<ActiveLayer>,
    occupied_rows_pool: VecPool<usize>,
    /// Scratch space used when replaying clip strips while popping clipped layers.
    occupied_rows_bool_scratch: Vec<bool>,
    /// A counter to assign monotonically increasing IDs to draws to enable depth buffer rendering.
    pub(super) next_draw_id: u32,
}

impl CommandBucketer {
    pub(crate) fn from_wh(width: u16, height: u16) -> Self {
        Self::new(RectU16::new(0, 0, width, height))
    }

    pub(crate) fn new(mut viewport: RectU16) -> Self {
        // It's _very_ important that we snap to tile coordinates. Fine rasterization assumes
        // that the width is a multiple of the tile width, so if that's not the case bad things
        // will happen!
        viewport = snap_bbox_to_tile_coordinates(viewport);
        let clip_bbox =
            snap_bbox_to_tile_coordinates(RectU16::new(0, 0, viewport.width(), viewport.height()));
        // Note: `viewport_clip_bbox` is already snapped to tile coordinates, so no need to `div_ceil`
        // here.
        let num_rows = usize::from(clip_bbox.height() / Tile::HEIGHT);

        Self {
            viewport,
            clip_bboxes: vec![clip_bbox],
            rows: RetainVec::with_len(num_rows, RowCmds::new),
            paint_fill_attrs: Vec::new(),
            layer_fill_attrs: Vec::new(),
            filter_paints: Vec::new(),
            active_layers: Vec::new(),
            occupied_rows_pool: VecPool::default(),
            occupied_rows_bool_scratch: vec![false; num_rows],
            // It is important to start at 1, because depth buffer uses 0 for "no entries yet".
            next_draw_id: 1,
        }
    }

    fn bbox_span(bbox: RectU16) -> Span {
        Span::new(bbox.x0, bbox.x1.saturating_sub(bbox.x0))
    }

    pub(crate) fn rows(&self) -> &[RowCmds] {
        self.rows.as_slice()
    }

    pub(crate) fn width(&self) -> u16 {
        self.clip_bboxes[0].width()
    }

    pub(crate) fn attrs(&self) -> &[PaintFillAttrs] {
        &self.paint_fill_attrs
    }

    pub(crate) fn layer_attrs(&self) -> &[LayerFillAttrs] {
        &self.layer_fill_attrs
    }

    pub(crate) fn filter_paints(&self) -> &[EncodedPaint] {
        &self.filter_paints
    }

    pub(crate) fn reset(&mut self, mut viewport: RectU16) {
        viewport = snap_bbox_to_tile_coordinates(viewport);
        let clip_bbox =
            snap_bbox_to_tile_coordinates(RectU16::new(0, 0, viewport.width(), viewport.height()));

        let num_rows = usize::from(viewport.height() / Tile::HEIGHT);
        self.rows.clear();
        self.rows.resize_with(num_rows, RowCmds::new);
        self.paint_fill_attrs.clear();
        self.layer_fill_attrs.clear();
        self.filter_paints.clear();
        for layer in self.active_layers.drain(..) {
            self.occupied_rows_pool.submit(layer.occupied_rows);
        }
        self.occupied_rows_bool_scratch.clear();
        self.occupied_rows_bool_scratch.resize(num_rows, false);
        self.next_draw_id = 1;
        self.viewport = viewport;
        self.clip_bboxes.truncate(1);
        self.clip_bboxes[0] = clip_bbox;
    }

    fn next_draw_id(&mut self) -> u32 {
        let draw_id = self.next_draw_id;
        self.next_draw_id = self.next_draw_id + 1;

        draw_id
    }

    #[inline(always)]
    pub(super) fn ensure_row_layers(&mut self, row_idx: usize) {
        let layer_depth = self.rows[row_idx].layer_depth;
        if layer_depth == self.active_layers.len() {
            return;
        }

        for layer_idx in layer_depth..self.active_layers.len() {
            self.rows[row_idx].push_buf();
            self.active_layers[layer_idx].occupied_rows.push(row_idx);
        }
    }

    pub(crate) fn bucket_commands(
        &mut self,
        cmds: &[RecordedCmd],
        layers: &[RecordedLayer],
        strips: &[Strip],
        encoded_paints: &[EncodedPaint],
        filter_ctx: &FilterContext,
    ) {
        // When rendering filter layers, we always anchor them so that the top-left of the bounding
        // box lands at (0, 0), even if the bounding box's top-left is for example at (200, 200).
        // This is to ensure that the pixmap is as small as possible. Therefore, we need to keep
        // track of this offset so that for example paints know that they should actually be
        // sampled at (200, 200) instead of (0, 0). Otherwise, if we for example had a filter layer
        // that draws something small in the bottom right but nowhere else, we would still need to
        // allocate a full view-port sized pixmap!

        let origin = (self.viewport.x0, self.viewport.y0);
        assert_eq!(origin.0 % Tile::WIDTH, 0);
        assert_eq!(origin.1 % Tile::HEIGHT, 0);

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
                    let attrs = PaintFillAttrs {
                        paint: paint.clone(),
                        blend_mode: *blend_mode,
                        mask: mask.clone(),
                        draw_id,
                        thread_idx: *thread_idx,
                        origin,
                    };
                    self.generate_fill(&strips[strip_range.clone()], &attrs, encoded_paints);
                }
                RecordedCmd::PushLayer { id } => {
                    let props = &layers[id.get()].props;
                    self.push_layer(props)
                }
                RecordedCmd::CompositeFilterLayer { id } => {
                    let props = &layers[id.get()].props;
                    let RecordedLayerKind::Filter { placement, .. } = &layers[id.get()].kind else {
                        unreachable!()
                    };

                    let needs_layer = props.blend_mode != BlendMode::default()
                        || props.opacity != 1.0
                        || props.mask.is_some()
                        || props.clip.is_some();

                    if needs_layer {
                        self.push_layer(props);
                    }
                    // Note: At this point, we've already rasterized all dependent
                    // filter layers, so this should never fail.
                    if let Some(pixmap) = filter_ctx.filter_layer(id.get()) {
                        self.generate_filter_layer_fill(
                            pixmap,
                            placement.composite_bbox,
                            placement.src_origin(),
                            encoded_paints.len(),
                            origin,
                        );
                    }
                    if needs_layer {
                        self.pop_layer(strips, origin);
                    }
                }
                RecordedCmd::PopLayer => self.pop_layer(strips, origin),
            }
        }
    }

    pub(crate) fn push_layer(&mut self, props: &LayerProps) {
        let parent_bbox = *self.clip_bboxes.last().unwrap();
        let bbox = props
            .clip
            .as_ref()
            .map(|clip| clip.bbox.intersect(parent_bbox))
            .unwrap_or(parent_bbox);

        let bbox = snap_bbox_to_tile_coordinates(bbox);
        if props.clip.is_some() {
            self.clip_bboxes.push(bbox);
        }

        self.active_layers.push(ActiveLayer {
            // TODO: Masks are currently probably broken if they are used inside of a filter layer,
            // since they aren't shifted and also only work if the mask has the same dimensions as
            // our allocated filter layer.
            mask: props.mask.clone(),
            blend_mode: props.blend_mode,
            opacity: props.opacity,
            clip: props.clip.clone(),
            span: Self::bbox_span(bbox),
            occupied_rows: self.occupied_rows_pool.take(),
        });

        // If the blend mode is destructive, we need to eagerly push to all rows in the clip bbox,
        // since even areas where we didn't draw anything need to be blended with the destructive
        // blend mode.
        if props.blend_mode.is_destructive() {
            let row_start = usize::from(bbox.y0 / Tile::HEIGHT);
            let row_end = usize::from(bbox.y1.div_ceil(Tile::HEIGHT)).min(self.rows.len());
            for row_idx in row_start..row_end {
                self.ensure_row_layers(row_idx);
            }
        }
    }

    pub(crate) fn pop_layer(&mut self, strips: &[Strip], pixmap_origin: (u16, u16)) {
        let mut layer = self.active_layers.pop().unwrap();
        let opacity = layer.opacity;
        let blend_mode = layer.blend_mode;
        let full_width = self.width();

        // Two cases that need to be distinguished.
        //
        // If no clip was associated, we simply iterate over all rows that
        // were lazily associated with some rendered contents and emit a `LayerFill`
        // instructions across the whole width of the bounding box of the layer.
        //
        // If there _was_ a clip, things get trickier because the `LayerFill` commands
        // need to be generated on a more fine-grained basis, and might in certain cases
        // also require anti-aliasing.

        if let Some(clip) = layer.clip {
            let attrs_idx = self.layer_fill_attrs.len() as u32;

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                thread_idx: clip.thread_idx,
            });
            self.clip_bboxes.pop();
            let clip_strips = &strips[clip.strip_range];

            // We need to make sure that we only emit `LayerFill` commands for rows that actually
            // have been pushed into. If we don't have a destructive blend mode and only parts of
            // the layer were painted, it can happen that the clip path itself covers a row
            // that actually doesn't have an associated `PushBuf` command.
            // For such rows, we don't want to emit any layer fill commands, because there is
            // nothing to fill into in the first place!

            // `occupied_rows` stores the indices of all rows that have been marked, but they
            // are not sorted or anything. Therefore, we convert it into an indexable array
            // such that in the `generate` closure, we can easily check whether the row is included
            // or now.
            for &row_idx in &layer.occupied_rows {
                self.occupied_rows_bool_scratch[row_idx] = true;
            }

            // Only generate layer fill commands if it actually lies within a row that has
            // been touched by the contents of the layer.
            self.generate(
                clip_strips,
                pixmap_origin,
                |bucketer, fill| {
                    if bucketer.occupied_rows_bool_scratch[fill.row_idx] {
                        bucketer.rows[fill.row_idx].push_cmd(RenderCmd::LayerFill(LayerFill::new(
                            fill.span, None, attrs_idx,
                        )));
                    }
                },
                |bucketer, fill| {
                    if bucketer.occupied_rows_bool_scratch[fill.row_idx] {
                        bucketer.rows[fill.row_idx].push_cmd(RenderCmd::LayerFill(LayerFill::new(
                            fill.span,
                            Some(fill.alpha_idx),
                            attrs_idx,
                        )));
                    }
                },
            );

            for row_idx in layer.occupied_rows.drain(..) {
                self.rows[row_idx].pop_buf();
                // Make sure to reset it so that by the end, the vector is all `false` again.
                self.occupied_rows_bool_scratch[row_idx] = false;
            }

            self.occupied_rows_pool.submit(layer.occupied_rows);
        } else {
            let attrs_idx = self.layer_fill_attrs.len() as u32;

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                thread_idx: 0,
            });

            let span = if blend_mode.is_destructive() {
                layer.span
            } else {
                Span::new(0, full_width)
            };

            for row_idx in layer.occupied_rows.drain(..) {
                let row = &mut self.rows[row_idx];

                row.push_cmd(RenderCmd::LayerFill(LayerFill::new(span, None, attrs_idx)));
                row.pop_buf();
            }

            self.occupied_rows_pool.submit(layer.occupied_rows);
        }
    }

    pub(crate) fn generate_filter_layer_fill(
        &mut self,
        pixmap: Arc<Pixmap>,
        dest_bbox: RectU16,
        src_origin: (u16, u16),
        static_paint_count: usize,
        origin: (u16, u16),
    ) {
        // TODO: Make clearer what this does.
        let src_offset = (
            i32::from(src_origin.0) - i32::from(dest_bbox.x0),
            i32::from(src_origin.1) - i32::from(dest_bbox.y0),
        );
        // Similarly to how indexed paints are handled, shift by the pixmap origin.
        let dest_bbox = bbox_relative_to(dest_bbox, origin);

        let source_dest_bbox = {
            let local_x0 = (-i32::from(origin.0) - src_offset.0).max(0);
            let local_y0 = (-i32::from(origin.1) - src_offset.1).max(0);
            let local_x1 = (local_x0 + i32::from(pixmap.width())).min(u16::MAX as i32);
            let local_y1 = (local_y0 + i32::from(pixmap.height())).min(u16::MAX as i32);

            let shifted = RectU16::new(
                local_x0 as u16,
                local_y0 as u16,
                local_x1 as u16,
                local_y1 as u16,
            );

            shifted.intersect(dest_bbox)
        };
        
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        // As noted in [`CommandBucketer::clip_bboxes`], we only need to composite the parts
        // of the filter layer that actually lie within clip bounding box.
        let clipped_dest_bbox = source_dest_bbox.intersect(clip_bbox);
        if clipped_dest_bbox.is_empty() {
            return;
        }

        let draw_id = self.next_draw_id();
        let span = Self::bbox_span(clipped_dest_bbox);
        let paint_idx = static_paint_count + self.filter_paints.len();
        self.filter_paints.push(EncodedPaint::Image(EncodedImage {
            source: ImageSource::Pixmap(pixmap),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
            may_have_transparency: true,
            transform: Affine::translate((f64::from(src_offset.0), f64::from(src_offset.1))),
            x_advance: Vec2::new(1.0, 0.0),
            y_advance: Vec2::new(0.0, 1.0),
            tint: None,
        }));
        let attrs_idx = self.paint_fill_attrs.len() as u32;
        self.paint_fill_attrs.push(PaintFillAttrs {
            paint: Paint::Indexed(IndexedPaint::new(paint_idx)),
            blend_mode: BlendMode::default(),
            mask: None,
            draw_id,
            thread_idx: 0,
            origin,
        });
        let row_start = usize::from(clipped_dest_bbox.y0 / Tile::HEIGHT);
        let row_end = usize::from(clipped_dest_bbox.y1.div_ceil(Tile::HEIGHT));
        for row_idx in row_start..row_end {
            self.push_fill(GeneratedFill { row_idx, span }, attrs_idx, None);
        }
    }

    pub(crate) fn generate_fill(
        &mut self,
        strip_buf: &[Strip],
        attrs: &PaintFillAttrs,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        debug_assert_ne!(attrs.draw_id, 0, "fill draw IDs should start at 1");

        let pixmap_origin = attrs.origin;
        let attrs_idx = self.paint_fill_attrs.len() as u32;
        self.paint_fill_attrs.push(attrs.clone());

        let draw_id =
            // While in certain cases it _might_ be okay to use depth culling while inside of
            // a layer, it can get very finicky with blend modes etc., so we just outright
            // reject those for now.
            (self.active_layers.is_empty()
                && attrs.blend_mode == BlendMode::default()
                && attrs.mask.is_none()
                && !attrs.paint.may_have_transparency(encoded_paints))
                .then_some(attrs.draw_id);

        self.generate(
            strip_buf,
            pixmap_origin,
            |bucketer, fill| {
                // No need to call `ensure_row_layers` here because we are guaranteed to not
                // be inside of a layer.
                bucketer.push_fill(fill, attrs_idx, draw_id);
            },
            |bucketer, fill| {
                let row_idx = fill.row_idx;
                bucketer.ensure_row_layers(row_idx);
                bucketer.rows[row_idx].push_cmd(RenderCmd::PaintFill(PaintFill::new(
                    fill.span,
                    Some(fill.alpha_idx),
                    attrs_idx,
                )));
            },
        );
    }

    pub(crate) fn generate<F, A>(
        &mut self,
        strip_buf: &[Strip],
        pixmap_origin: (u16, u16),
        mut fill_cmd: F,
        mut alpha_fill_cmd: A,
    ) where
        F: FnMut(&mut Self, GeneratedFill),
        A: FnMut(&mut Self, GeneratedAlphaFill),
    {
        if strip_buf.is_empty() {
            return;
        }

        let clip_bbox = *self.clip_bboxes.last().unwrap();
        // Note: Those will always be aligned to tile coordinates.
        let clip_x0 = clip_bbox.x0;
        let clip_x1 = clip_bbox.x1.min(self.width());

        debug_assert!(clip_x0.is_multiple_of(Tile::WIDTH));
        debug_assert!(clip_x1.is_multiple_of(Tile::WIDTH));

        let strip_x = |strip: &Strip| {
            if strip.is_sentinel() {
                strip.x
            } else {
                strip.x.saturating_sub(pixmap_origin.0)
            }
        };

        let strip_row = |strip: &Strip| strip.y.saturating_sub(pixmap_origin.1) / Tile::HEIGHT;

        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];
            let strip_y = strip_row(strip);
            let row_y = strip_y.saturating_mul(Tile::HEIGHT);

            if row_y < clip_bbox.y0 {
                continue;
            }
            if row_y >= clip_bbox.y1 {
                break;
            }

            let row_idx = strip_y as usize;

            let next_strip = &strip_buf[i + 1];
            let strip_width = strip.width_to(next_strip);
            let x0 = strip_x(strip);
            let x1 = x0.saturating_add(strip_width);

            if strip_width > 0 && x0 < clip_x1 && x1 > clip_x0 {
                alpha_fill_cmd(
                    self,
                    GeneratedAlphaFill {
                        row_idx,
                        span: Span::new(x0, strip_width),
                        alpha_idx: strip.alpha_idx(),
                    },
                );
            }

            if next_strip.fill_gap() && strip_y == strip_row(next_strip) {
                let fill_x0 = x1.max(clip_x0);
                let fill_x1 = strip_x(next_strip).min(clip_x1);
                if fill_x0 < fill_x1 {
                    fill_cmd(
                        self,
                        GeneratedFill {
                            row_idx,
                            span: Span::new(fill_x0, fill_x1 - fill_x0),
                        },
                    );
                }
            }
        }
    }

    /// Note: If depth-culling should be disabled, pass `None` to `draw_id`.
    fn push_fill(&mut self, fill: GeneratedFill, attrs_idx: u32, draw_id: Option<u32>) {
        self.ensure_row_layers(fill.row_idx);
        let row = &mut self.rows[fill.row_idx];
        let draw_id = draw_id.filter(|_| row.layer_depth == 0);

        let Some(draw_id) = draw_id else {
            row.push_cmd(RenderCmd::PaintFill(PaintFill::new(
                fill.span, None, attrs_idx,
            )));

            return;
        };

        depth::split_opaque_span(fill.span, |span, segment| match segment {
            DepthSegment::Regular => {
                row.push_cmd(RenderCmd::PaintFill(PaintFill::new(span, None, attrs_idx)));
            }
            DepthSegment::Opaque => {
                row.push_depth_write(PaintFill::new(span, None, attrs_idx), draw_id);
            }
        });
    }
}

/// Metadata about the currently active layer.
#[derive(Debug, Clone)]
pub(crate) struct ActiveLayer {
    pub(crate) mask: Option<Mask>,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) clip: Option<LayerClip>,
    pub(crate) span: Span,
    /// Which rows have been drawn into and thus contain lazily-allocated `PushBuf` instructions.
    pub(crate) occupied_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerClip {
    pub(crate) strip_range: Range<usize>,
    pub(crate) thread_idx: u8,
    pub(crate) bbox: RectU16,
}

/// A generic fill to allow using `generate_fill` to create either paint fills or blend fills.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedFill {
    pub(crate) row_idx: usize,
    pub(crate) span: Span,
}

/// A generic alpha fill to allow using `generate_fill` to create either paint fills or blend fills.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedAlphaFill {
    pub(crate) row_idx: usize,
    pub(crate) span: Span,
    pub(crate) alpha_idx: u32,
}

#[cfg(test)]
mod tests {
    use crate::coarse::CommandBucketer;
    use crate::coarse::cmd::{PaintFillAttrs, RenderCmd};
    use crate::coarse::depth::DEPTH_BUCKET_WIDTH;
    use crate::record::LayerProps;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::BlendMode;
    use vello_common::strip::Strip;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    fn fill_attrs(paint: Paint) -> PaintFillAttrs {
        PaintFillAttrs {
            paint,
            blend_mode: BlendMode::default(),
            mask: None,
            draw_id: 1,
            thread_idx: 0,
            origin: (0, 0),
        }
    }

    fn layer_props() -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip: None,
        }
    }

    #[test]
    fn opaque_fill_inside_layer_does_not_use_depth_write() {
        let mut bucketer = CommandBucketer::from_wh(DEPTH_BUCKET_WIDTH, 4);
        let strips = [
            Strip::new(0, 0, 0, false),
            Strip::new(DEPTH_BUCKET_WIDTH, 0, 0, true),
        ];

        bucketer.push_layer(&layer_props());
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_writes.len(), 0);
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], RenderCmd::PushBuf));
        assert!(
            matches!(row.cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }

    #[test]
    fn opaque_fill_uses_depth_write_when_possible() {
        let end = DEPTH_BUCKET_WIDTH * 2 + 4;
        let mut bucketer = CommandBucketer::from_wh(end, 4);
        let strips = [Strip::new(4, 0, 0, false), Strip::new(end, 0, 0, true)];

        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_writes.len(), 1);
        assert_eq!(row.depth_writes[0].span.pixel_x(), DEPTH_BUCKET_WIDTH);
        assert_eq!(row.depth_writes[0].span.pixel_width(), DEPTH_BUCKET_WIDTH);
        assert_eq!(row.cmds.len(), 2);
        assert!(
            matches!(row.cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 4 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH - 4)
        );
        assert!(
            matches!(row.cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == DEPTH_BUCKET_WIDTH * 2 && cmd.span.pixel_width() == 4)
        );
    }

    #[test]
    fn non_opaque_fill_uses_regular_commands() {
        let mut bucketer = CommandBucketer::from_wh(DEPTH_BUCKET_WIDTH, 4);
        let strips = [
            Strip::new(0, 0, 0, false),
            Strip::new(DEPTH_BUCKET_WIDTH, 0, 0, true),
        ];

        bucketer.generate_fill(
            &strips,
            &fill_attrs(Paint::Solid(color(BLUE.with_alpha(0.5)))),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_writes.len(), 0);
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }
}
