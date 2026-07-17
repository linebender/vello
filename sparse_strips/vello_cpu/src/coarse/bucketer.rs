// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::cmd::{DepthFill, LayerFill, LayerFillAttrs, PaintFill, PaintFillAttrs, RenderCmd};
use super::depth::{DepthSegment, DepthState};
use crate::coarse::depth;
use crate::filter::context::FilterContext;
use crate::kurbo::{Affine, Vec2};
use crate::peniko::{BlendMode, Extend, ImageQuality, ImageSampler};
use crate::util::Span;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::encode::{EncodedImage, EncodedPaint};
use vello_common::filter::FilterLayerPlacement;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageSource, IndexedPaint, Paint};
use vello_common::pixmap::Pixmap;
use vello_common::record::{LayerClip, LayerProps, RecordedCmd, RecordedLayer, RecordedLayerKind};
use vello_common::strip::{Strip, visit_strip_fill_segments};
use vello_common::tile::Tile;
use vello_common::util::{Clear, RectExt, RetainVec, VecPool};

/// State for a single row of strips.
#[derive(Debug, Default)]
pub(crate) struct RowState {
    /// Normal render commands rendered in back-to-front with depth buffer read.
    pub(crate) render_cmds: Vec<RenderCmd>,
    /// Opaque fill commands rendered front-to-back with depth buffer read and write.
    pub(crate) depth_cmds: Vec<DepthFill>,
    /// State of the depth buffer for this row.
    depth: DepthState,
    /// Current layer depth.
    pub(super) layer_depth: usize,
    layer_stack: Vec<RowLayerState>,
}

/// Each recorded layer by itself already stores a bounding box.
/// However, we want to take this one step further and track a horizontal bounding
/// box of each row band individually. This allows us to save a lot of work when clearing buffers
/// during fine rasterization.
///
/// For example, let's say that we are rendering a triangle with the points
/// (0, 0), (500, 0) and (250, 250). The bounding box would be (0, 0) and (500, 500). However,
/// the triangle gets much more narrow as we advance through the rows. If we just chose
/// the coarse bounding box, we would always end up clearing a whole width of 500 pixels during
/// fine rasterization. By tracking the bounding box per row, we can reduce the work to the
/// absolute minimum necessary for each row.
#[derive(Debug)]
struct RowLayerState {
    /// The index in the command stream where the `PushBuf` command of the corresponding
    /// layer lives.
    push_cmd_idx: usize,
    /// The horizontal span of the layer.
    span: Option<Span>,
}

impl RowState {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.render_cmds.clear();
        self.depth_cmds.clear();
        self.depth.reset();
        self.layer_depth = 0;
        self.layer_stack.clear();
    }

    #[inline]
    pub(super) fn push_cmd(&mut self, cmd: RenderCmd) {
        match cmd {
            RenderCmd::PaintFill(cmd) => self.include_current_span(cmd.span),
            RenderCmd::LayerFill(cmd) => self.include_current_span(cmd.span),
            RenderCmd::PushBuf(_) | RenderCmd::PopBuf => {}
        }

        self.render_cmds.push(cmd);
    }

    #[inline]
    pub(super) fn push_buf(&mut self) {
        let push_cmd_idx = self.render_cmds.len();
        self.render_cmds.push(RenderCmd::PushBuf(None));
        self.layer_stack.push(RowLayerState {
            push_cmd_idx,
            span: None,
        });
        self.layer_depth += 1;
    }

    #[inline]
    pub(super) fn pop_buf(&mut self) {
        let layer = self.layer_stack.pop().unwrap();
        match &mut self.render_cmds[layer.push_cmd_idx] {
            RenderCmd::PushBuf(push_span) => *push_span = layer.span,
            _ => unreachable!("layer stack must point to a PushBuf command"),
        }

        self.render_cmds.push(RenderCmd::PopBuf);

        self.layer_depth -= 1;
    }

    #[inline]
    pub(super) fn push_depth_fill(&mut self, cmd: DepthFill, draw_id: u32) {
        let span = cmd.span();
        self.depth.include_span(span, draw_id);
        self.depth_cmds.push(cmd);
    }

    #[inline]
    pub(crate) fn can_skip_depth(&self, span: Span, draw_id: u32) -> bool {
        self.depth.can_skip(span, draw_id)
    }

    #[inline]
    fn include_current_span(&mut self, span: Span) {
        if let Some(layer) = self.layer_stack.last_mut() {
            match &mut layer.span {
                Some(layer_span) => layer_span.extend(span),
                None => layer.span = Some(span),
            }
        }
    }
}

impl Clear for RowState {
    fn clear(&mut self) {
        Self::clear(self);
    }
}

/// A bucketer that groups commands into strip-row-sized buckets.
#[derive(Debug)]
pub(crate) struct CommandBucketer {
    /// The viewport of the root/filter layer we are currently bucketing.
    viewport: RectU16,
    /// The currently active stack of clip bboxes (from layer clips), always anchored at the
    /// (0, 0) origin, regardless of the viewport, tracked for two reasons:
    /// - So we can clamp fill commands to the bbox and avoid unnecessary rendering work.
    /// - In case we have a filter layer with a clip, it all happens in three steps:
    ///   1) We first push a new intermediate layer with the clip.
    ///   2) We composite the whole filter layer **cropped to the active clip bbox**. Note that
    ///      the filter layer itself is _not_ affected by the clip bbox, as filter layers should
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
    pub(super) rows: RetainVec<RowState>,
    pub(crate) paint_fill_attrs: Vec<PaintFillAttrs>,
    pub(crate) layer_fill_attrs: Vec<LayerFillAttrs>,
    pub(crate) filter_paints: Vec<EncodedPaint>,
    /// Keeping track of currently active layers to enable lazy layer pushing.
    pub(super) active_layers: Vec<ActiveLayer>,
    /// A vector pool for keeping track of occupied rows in a layer.
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
        // could happen!
        viewport = viewport.snap_to_tile_coordinates();
        let clip_bbox = RectU16::new(0, 0, viewport.width(), viewport.height());
        // Note: `clip_bbox` is already snapped to tile coordinates because `viewport` is, so no
        // need to `div_ceil` here.
        let num_rows = usize::from(clip_bbox.height() / Tile::HEIGHT);

        Self {
            viewport,
            clip_bboxes: vec![clip_bbox],
            rows: RetainVec::with_len(num_rows, RowState::new),
            paint_fill_attrs: Vec::new(),
            layer_fill_attrs: Vec::new(),
            filter_paints: Vec::new(),
            active_layers: Vec::new(),
            occupied_rows_pool: VecPool::default(),
            occupied_rows_bool_scratch: vec![false; num_rows],
            // It is important to start at 1, because the depth buffer uses 0 for "no entries yet".
            next_draw_id: 1,
        }
    }

    fn bbox_span(bbox: RectU16) -> Span {
        Span::new(bbox.x0, bbox.x1.saturating_sub(bbox.x0))
    }

    pub(crate) fn rows(&self) -> &[RowState] {
        self.rows.as_slice()
    }

    pub(crate) fn width(&self) -> u16 {
        self.clip_bboxes[0].width()
    }

    fn viewport_origin(&self) -> (u16, u16) {
        (self.viewport.x0, self.viewport.y0)
    }

    pub(crate) fn reset(&mut self, mut viewport: RectU16) {
        // See comments in `CommandBucketer::reset`.
        viewport = viewport.snap_to_tile_coordinates();
        let clip_bbox = RectU16::new(0, 0, viewport.width(), viewport.height());
        let num_rows = usize::from(viewport.height() / Tile::HEIGHT);

        self.rows.clear();
        self.rows.resize_with(num_rows, RowState::new);
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
        self.next_draw_id += 1;

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
        // This is to ensure that the pixmap is as small as possible. Otherwise, if we for example
        // had a filter layer that draws something small in the bottom right but nowhere else, we
        // would still need to allocate a full viewport-sized pixmap!
        // Therefore, we need to keep track of this offset so that for example paints know that
        // they should actually be sampled at (200, 200) instead of (0, 0).

        debug_assert_eq!(
            self.viewport.x0 % Tile::WIDTH,
            0,
            "viewport origin must be tile-width aligned",
        );
        debug_assert_eq!(
            self.viewport.y0 % Tile::HEIGHT,
            0,
            "viewport origin must be tile-height aligned",
        );

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
                        origin: self.viewport_origin(),
                    };
                    self.generate_fill(&strips[strip_range.clone()], &attrs, encoded_paints);
                }
                RecordedCmd::PushLayer { id } => {
                    let props = &layers[id.get()].props;
                    self.push_layer(props);
                }
                RecordedCmd::FilterLayer { id } => {
                    let props = &layers[id.get()].props;
                    let RecordedLayerKind::Filter { placement, .. } = &layers[id.get()].kind else {
                        unreachable!()
                    };

                    let needs_layer = props.blend_mode != BlendMode::default()
                        || props.opacity != 1.0
                        || props.mask.is_some()
                        || props.clip_path.is_some();

                    if needs_layer {
                        self.push_layer(props);
                    }

                    // Note: At this point, we've already rasterized all dependent
                    // filter layers, so this should never fail.
                    if let Some(pixmap) = filter_ctx.filter_layer(id.get()) {
                        self.generate_filter_layer_fill(pixmap, *placement, encoded_paints.len());
                    }

                    if needs_layer {
                        self.pop_layer(strips);
                    }
                }
                RecordedCmd::PopLayer => self.pop_layer(strips),
            }
        }
    }

    pub(crate) fn push_layer(&mut self, props: &LayerProps) {
        let parent_bbox = *self.clip_bboxes.last().unwrap();
        let bbox = props
            .clip_path
            .as_ref()
            .map(|clip| {
                // Make sure to translate the clip path from viewport-space to local space, since
                // `clip_bboxes` uses this coordinate system.
                let clip_bbox = clip.bbox.relative_to_origin(self.viewport_origin());
                clip_bbox.intersect(parent_bbox)
            })
            .unwrap_or(parent_bbox);

        // Once again, this need to be snapped because fine rasterization assumes tile-alignment.
        // The bbox is used to clip fill commands, but if the bbox itself is not aligned we might
        // end up making the fill commands unaligned as well.
        let bbox = bbox.snap_to_tile_coordinates();
        if props.clip_path.is_some() {
            self.clip_bboxes.push(bbox);
        }

        self.active_layers.push(ActiveLayer {
            // TODO: Masks are currently probably broken if they are used inside of a filter layer,
            // since they aren't shifted and also only work if the mask has the same dimensions as
            // our allocated filter layer.
            mask: props.mask.clone(),
            blend_mode: props.blend_mode,
            opacity: props.opacity,
            clip: props.clip_path.clone(),
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

    pub(crate) fn pop_layer(&mut self, strips: &[Strip]) {
        let mut layer = self.active_layers.pop().unwrap();
        let opacity = layer.opacity;
        let blend_mode = layer.blend_mode;

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
            let draw_id = self.next_draw_id();

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                draw_id,
                thread_idx: clip.thread_idx,
            });
            self.clip_bboxes.pop();

            // Note: The clip strips themselves are still in viewport space, not in local space.
            // They will be converted to local space when calling `generate`.
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
            let draw_id = self.next_draw_id();

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                draw_id,
                thread_idx: 0,
            });

            for row_idx in layer.occupied_rows.drain(..) {
                let row = &mut self.rows[row_idx];

                // TODO: Instead of always pushing the full layer bbox across all rows, it
                // would be nice to instead only emit the per-row bounding box.
                row.push_cmd(RenderCmd::LayerFill(LayerFill::new(
                    layer.span, None, attrs_idx,
                )));
                row.pop_buf();
            }

            self.occupied_rows_pool.submit(layer.occupied_rows);
        }
    }

    pub(crate) fn generate_filter_layer_fill(
        &mut self,
        pixmap: Arc<Pixmap>,
        placement: FilterLayerPlacement,
        static_paint_count: usize,
    ) {
        let origin = self.viewport_origin();
        let dest_bbox = placement.dest_bbox;
        let src_sample_shift = placement.src_origin();

        // Here, we want to determine the absolute transform that needs to be applied to the filter
        // image for it to be placed correctly. On the one hand, we position it such that it starts
        // at the top-left of `dest_bbox`, which is the intended location the filter layer should
        // be composited into. On the other hand, we optionally apply a correction to ensure we
        // are sampling from the right location of the filter pixmap (see
        // `FilterLayerPlacement::new` for a more detailed description of how/what this offset
        // represents).
        let src_offset = (
            i32::from(src_sample_shift.0) - i32::from(dest_bbox.x0),
            i32::from(src_sample_shift.1) - i32::from(dest_bbox.y0),
        );

        // Now that we have determined the image transform, we next need to determine which
        // strip rows in the bucketer should actually be emitted. Keep in mind that regardless
        // of what `viewport_origin` is, when actually rendering we always shift the origin to
        // (0, 0). (Note: We did _not_ do this for computing `src_offset` because the shift for the
        // paint itself will be applied later when resolving the indexed paint)
        let dest_bbox = dest_bbox.relative_to_origin(origin);
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        // As noted in [`CommandBucketer::clip_bboxes`], we only need to composite the parts
        // of the filter layer that actually lie within clip bounding box.
        // `clip_bbox` already is in viewport-local coordinates, so we don't need to shift it
        // like we did for `dest_bbox`.
        let clipped_dest_bbox = dest_bbox.intersect(clip_bbox);
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
            |bucketer, fill| {
                // `push_fill` already calls `ensure_row_layers` so no need to call it twice.
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

        debug_assert_eq!(
            clip_x0 % Tile::WIDTH,
            0,
            "clip start must be tile-width aligned",
        );
        debug_assert_eq!(
            clip_x1 % Tile::WIDTH,
            0,
            "clip end must be tile-width aligned",
        );
        debug_assert_eq!(
            clip_bbox.y0 % Tile::HEIGHT,
            0,
            "clip start must be tile-height aligned",
        );
        debug_assert_eq!(
            clip_bbox.y1 % Tile::HEIGHT,
            0,
            "clip end must be tile-height aligned",
        );

        let origin = self.viewport_origin();
        debug_assert_eq!(
            origin.0 % Tile::WIDTH,
            0,
            "viewport x origin must be tile-width aligned",
        );
        debug_assert_eq!(
            origin.1 % Tile::HEIGHT,
            0,
            "viewport y origin must be tile-height aligned",
        );

        // Note: the viewport of a filter layer is based on the bounds of its rendered contents.
        // Therefore, those are always guaranteed to be within the viewport rect. However, this does not
        // apply to any clip paths associated with the filter layer. Including those in the filter
        // layer bbox could unnecessarily balloon the size of the pixmap we allocate if it is
        // very large, since those don't actually contribute any visible output. However, this does
        // mean that this method might be called with strips that do not lie within the viewport.
        // Therefore, we need to make sure to clip those appropriately.

        let viewport_y1 = self.viewport.y1;
        let origin_tile_x = origin.0 / Tile::WIDTH;
        let origin_tile_y = origin.1 / Tile::HEIGHT;
        let clip_scene_y0 = viewport_y1.min(origin.1.saturating_add(clip_bbox.y0));
        let clip_scene_y1 = viewport_y1.min(origin.1.saturating_add(clip_bbox.y1));
        // Convert to scene coordinates.
        let clip_scene_x0 = origin.0.saturating_add(clip_x0);
        let clip_scene_x1 = origin.0.saturating_add(clip_x1);

        // Clip bounding box in tile units.
        let tile_bounds = RectU16::new(
            clip_scene_x0 / Tile::WIDTH,
            clip_scene_y0 / Tile::HEIGHT,
            clip_scene_x1 / Tile::WIDTH,
            clip_scene_y1 / Tile::HEIGHT,
        );

        visit_strip_fill_segments(
            strip_buf,
            tile_bounds,
            self,
            |bucketer, segment| {
                let row_idx = usize::from(segment.tile_y - origin_tile_y);
                let x0 = (segment.tile_x0 - origin_tile_x) * Tile::WIDTH;
                let x1 = (segment.tile_x1 - origin_tile_x) * Tile::WIDTH;
                alpha_fill_cmd(
                    bucketer,
                    GeneratedAlphaFill {
                        row_idx,
                        span: Span::new(x0, x1 - x0),
                        alpha_idx: segment.alpha_idx,
                    },
                );
            },
            |bucketer, segment| {
                let row_idx = usize::from(segment.tile_y - origin_tile_y);
                let x0 = (segment.tile_x0 - origin_tile_x) * Tile::WIDTH;
                let x1 = (segment.tile_x1 - origin_tile_x) * Tile::WIDTH;
                fill_cmd(
                    bucketer,
                    GeneratedFill {
                        row_idx,
                        span: Span::new(x0, x1 - x0),
                    },
                );
            },
        );
    }

    /// Note: If depth-culling should be disabled, pass `None` to `draw_id`.
    fn push_fill(&mut self, fill: GeneratedFill, attrs_idx: u32, draw_id: Option<u32>) {
        self.ensure_row_layers(fill.row_idx);
        let row = &mut self.rows[fill.row_idx];
        let draw_id = draw_id.filter(|_| row.layer_depth == 0);

        let Some(draw_id) = draw_id else {
            // If depth-culling is disabled, we can just push it as a single contiguous command.
            row.push_cmd(RenderCmd::PaintFill(PaintFill::new(
                fill.span, None, attrs_idx,
            )));

            return;
        };

        depth::split_opaque_span(fill.span, |segment| match segment {
            DepthSegment::Regular(span) => {
                row.push_cmd(RenderCmd::PaintFill(PaintFill::new(span, None, attrs_idx)));
            }
            DepthSegment::Opaque(bucket_range) => {
                row.push_depth_fill(DepthFill::new(bucket_range, attrs_idx), draw_id);
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
    use super::LayerClip;
    use crate::coarse::CommandBucketer;
    use crate::coarse::cmd::{PaintFillAttrs, RenderCmd};
    use crate::coarse::depth::{BucketRange, DEPTH_BUCKET_WIDTH};
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::geometry::RectU16;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::BlendMode;
    use vello_common::record::LayerProps;
    use vello_common::strip::Strip;
    use vello_common::tile::Tile;

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
            clip_path: None,
        }
    }

    fn clipped_layer_props(bbox: RectU16) -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: Some(LayerClip {
                strip_range: 0..0,
                thread_idx: 0,
                bbox,
            }),
        }
    }

    fn clipped_layer_props_with_strips(
        bbox: RectU16,
        strip_range: core::ops::Range<usize>,
    ) -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: Some(LayerClip {
                strip_range,
                thread_idx: 0,
                bbox,
            }),
        }
    }

    fn count_layer_fills(cmds: &[RenderCmd]) -> usize {
        cmds.iter()
            .filter(|cmd| matches!(cmd, RenderCmd::LayerFill(_)))
            .count()
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
        assert_eq!(row.depth_cmds.len(), 0);
        assert_eq!(row.render_cmds.len(), 2);
        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        assert!(
            matches!(row.render_cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }

    #[test]
    fn alpha_fill_is_clipped_to_active_layer_bbox() {
        let mut bucketer = CommandBucketer::from_wh(8, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(12, 0, 48, false)];

        bucketer.push_layer(&clipped_layer_props(RectU16::new(4, 0, 8, 4)));
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.render_cmds.len(), 2);
        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        assert!(matches!(
            row.render_cmds[1],
            RenderCmd::PaintFill(cmd)
                if cmd.span.pixel_x() == 4
                    && cmd.span.pixel_width() == 4
                    && cmd.alpha_idx() == Some(u32::from(4 * Tile::HEIGHT))
        ));
    }

    #[test]
    fn opaque_fill_uses_depth_write_when_possible() {
        let end = DEPTH_BUCKET_WIDTH * 2 + 4;
        let mut bucketer = CommandBucketer::from_wh(end, 4);
        let strips = [Strip::new(4, 0, 0, false), Strip::new(end, 0, 0, true)];

        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_cmds.len(), 1);
        assert_eq!(row.depth_cmds[0].bucket_range(), BucketRange::new(1, 2));
        assert_eq!(row.render_cmds.len(), 2);
        assert!(
            matches!(row.render_cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 4 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH - 4)
        );
        assert!(
            matches!(row.render_cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == DEPTH_BUCKET_WIDTH * 2 && cmd.span.pixel_width() == 4)
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
        assert_eq!(row.depth_cmds.len(), 0);
        assert_eq!(row.render_cmds.len(), 1);
        assert!(
            matches!(row.render_cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }

    #[test]
    fn clips_fills_correctly_inside_nonzero_origin_viewport() {
        // Viewport spans scene (32, 32) to (96, 96). Local space is 64x64, the origin at (32, 32).
        let mut bucketer = CommandBucketer::new(RectU16::new(32, 32, 96, 96));
        // Clip bbox in scene coordinates: (40, 32)..(72, 96) => local (8, 0)..(40, 64).
        bucketer.push_layer(&clipped_layer_props(RectU16::new(40, 32, 72, 96)));

        // A 32px-wide alpha strip at scene (40, 32) => local (8, 0), fully inside the clip.
        let strips = [
            Strip::new(40, 32, 0, false),
            Strip::new(72, 32, 32 * u32::from(Tile::HEIGHT), false),
        ];
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];

        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        // The fill inside should not have been clipped.
        assert!(matches!(
            row.render_cmds[1],
            RenderCmd::PaintFill(cmd)
                if cmd.span.pixel_x() == 8
                    && cmd.span.pixel_width() == 32
                    && cmd.alpha_idx() == Some(0)
        ));
    }

    #[test]
    fn culls_clip_strips_above_viewport_origin() {
        // Viewport spans scene (0, 32) to (64, 96). Local space is 64x64, the origin at (0, 32).
        let mut bucketer = CommandBucketer::new(RectU16::new(0, 32, 64, 96));

        let alpha = u32::from(Tile::HEIGHT);
        let strips = [
            // Content: 16px alpha strip at scene (0, 32) => local row 0.
            Strip::new(0, 32, 0, false),
            Strip::new(16, 32, 16 * alpha, false),
            // Clip strip above the viewport origin at scene y = 0, covering nothing visible.
            Strip::new(0, 0, 16 * alpha, false),
            Strip::new(16, 0, 32 * alpha, false),
            // Clip strip at scene y = 32 => local row 0: the real coverage.
            Strip::new(0, 32, 32 * alpha, false),
            Strip::new(16, 32, 48 * alpha, false),
        ];

        // Clip bbox in scene coordinates: (0, 0)..(16, 40) => local (0, 0)..(16, 8).
        bucketer.push_layer(&clipped_layer_props_with_strips(
            RectU16::new(0, 0, 16, 40),
            2..6,
        ));
        bucketer.generate_fill(&strips[0..2], &fill_attrs(Paint::Solid(color(RED))), &[]);
        bucketer.pop_layer(&strips);

        let row = &bucketer.rows()[0];
        assert_eq!(
            count_layer_fills(&row.render_cmds),
            1,
            "row 0 must be composited exactly once, got {:?}",
            row.render_cmds
        );
    }
}
