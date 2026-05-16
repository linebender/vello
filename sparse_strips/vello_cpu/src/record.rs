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
//! of [`crate::coarse::bucketer::CommandBucketer`] for the root and each filter layer, which is especially cumbersome because
//! conceptually, a command bucketer is a 2D array. This makes it very difficult to resize and reuse
//! it while having the ability to retain and reuse inner allocations. By first recording all commands
//! and their strips into a single [`Vec<RecordedCmd>`], we can basically represent the commands of
//! a single layer with just one flat buffer, making it much easier to reuse them across frames. It
//! also allows us to collect useful metadata (e.g. the bounding box of a layer) and make appropriate
//! decisions about how to render them (for example, where to place a filter layer and what size to
//! allocate for it).

use crate::coarse::bucketer::LayerClip;
use crate::kurbo::{Affine, Rect};
use crate::peniko::BlendMode;
use crate::util::VecPool;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::tile::Tile;
use vello_common::util::RectExt;

/// A recorded command.
#[derive(Debug)]
pub(crate) enum RecordedCmd {
    /// A path fill command.
    Fill {
        thread_idx: u8,
        strip_range: Range<usize>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    /// Push a new (non-filter) layer to the layer stack.
    PushLayer { id: LayerId },
    /// Composite the filter layer with the given ID into the current layer.
    FilterLayer { id: LayerId },
    /// Pop the last (non-filter) layer from the layer stack.
    PopLayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerId(u32);

impl LayerId {
    pub(crate) fn new(id: usize) -> Self {
        Self(id as u32)
    }

    pub(crate) fn get(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub(crate) struct RecordedLayer {
    pub(crate) props: LayerProps,
    pub(crate) kind: RecordedLayerKind,
}

#[derive(Debug)]
pub(crate) struct LayerProps {
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) mask: Option<Mask>,
    pub(crate) clip_path: Option<LayerClip>,
}

#[derive(Debug)]
pub(crate) enum RecordedLayerKind {
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
    pub(crate) fn regular(props: LayerProps) -> Self {
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

/// Metadata about a filter layer and how it should be composited back into the parent layer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterLayerPlacement {
    /// The conceptual bounding box of the pixmap that needs to be allocated to render
    /// a layer correctly, including the area affected by the filter.
    ///
    /// For example, if the filter layer contains a rect spanning (200, 200) to (300, 300)
    /// with a blur that has a radius exceeding the rectangle 40 pixels on each side, the pixmap
    /// bbox will be (160, 160) to (340, 340).
    ///
    /// See the comments in [`FilterLayerPlacement::new`] for more information.
    pub(crate) pixmap_bbox: RectU16,
    /// Rectangle in the parent layer's coordinate space the filtered pixmap is composited into.
    ///
    /// See the comments in [`FilterLayerPlacement::new`] for more information.
    pub(crate) dest_bbox: RectU16,
    /// Source x offset used when sampling from the filter pixmap.
    ///
    /// See the comments in [`FilterLayerPlacement::new`] for more information.
    pub(crate) src_x: u16,
    /// Source y offset used when sampling from the filter pixmap.
    ///
    /// See the comments in [`FilterLayerPlacement::new`] for more information.
    pub(crate) src_y: u16,
}

impl FilterLayerPlacement {
    const EMPTY: Self = Self {
        pixmap_bbox: RectU16::INVERTED,
        dest_bbox: RectU16::INVERTED,
        src_x: 0,
        src_y: 0,
    };

    fn new(bbox: RectU16, filter_plan: &FilterData) -> Self {
        // Some more detailed explanations of what's going on here since this
        // part is a bit confusing.

        // `bbox` is the tight bounding box across all strips in the filter
        // layer. We now need to expand it by the filter padding to know how
        // large of a pixmap we actually need to allocate. Also, as mentioned
        // in [`FilterLayerPlan::new`], we need to ensure the pixmap itself is
        // also a multiple of the tile width / tile height.
        let pixmap_bbox = bbox
            .expand(filter_plan.filter_padding)
            .snap_to_tile_coordinates();

        // Remember that in `RenderContext`, we eagerly shift everything drawn by `source_shift`
        // to conservatively ensure that everything that might be needed for the filter is in the
        // viewport area. Therefore, when compositing the filter layer back, we need to undo that
        // shift.
        let (shift_x, shift_y) = filter_plan.source_shift();
        // For example, if `shift_x` is 20 and `pixmap_bbox.x0` is 4,
        // shifting the pixmap back would place its left edge at -16. Since we
        // start compositing at x=0, we need to skip the first 16 pixels
        // inside the cropped pixmap (`src_x = 20 - 4`). If `pixmap_bbox.x0`
        // is already >= `shift_x`, nothing is clipped and `src_x` is 0.
        let src_x = shift_x.saturating_sub(pixmap_bbox.x0);
        let src_y = shift_y.saturating_sub(pixmap_bbox.y0);
        let dest_bbox = pixmap_bbox.relative_to_origin((shift_x, shift_y));

        Self {
            pixmap_bbox,
            dest_bbox,
            src_x,
            src_y,
        }
    }

    pub(crate) fn src_origin(self) -> (u16, u16) {
        (self.src_x, self.src_y)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FilterData {
    pub(crate) filter: Filter,
    /// The transform that was in place when the filter layer was invoked.
    pub(crate) transform: Affine,
    /// Padding that needs to be added for the area where the filter is applied.
    ///
    /// See [`Filter::filter_expansion`].
    pub(crate) filter_padding: RectU16,
    /// Padding that needs to be added to the source region for correct filter application.
    ///
    /// See [`Filter::source_expansion`].
    pub(crate) source_padding: RectU16,
}

impl FilterData {
    pub(crate) fn new(filter: Filter, transform: Affine) -> Self {
        fn expansion_padding(expansion: Rect) -> RectU16 {
            // TODO: We technically shouldn't need to snap here. `source_padding` is only
            // used to shift the contents when rendering into the render context, and the
            // final pixmap bbox (which is derived from `filter_expansion` will be snapped
            // separately. However, not snapping here causes larger mismatches with Vello Hybrid
            // since the size of the final pixmap determines in which way we decimate for the
            // gaussian blur filter. Therefore, we keep this for compatibility.
            let expansion = expansion.snap_to_tile_coordinates();

            RectU16::new(
                (-expansion.x0) as u16,
                (-expansion.y0) as u16,
                expansion.x1 as u16,
                expansion.y1 as u16,
            )
        }

        let source_padding = expansion_padding(filter.source_expansion(&transform));
        let filter_padding = expansion_padding(filter.filter_expansion(&transform));

        Self {
            filter,
            transform,
            filter_padding,
            source_padding,
        }
    }

    /// By how much to shift all rendered contents to ensure that all rendered contents
    /// are visible in the viewport [0, 0, width, height].
    pub(crate) fn source_shift(&self) -> (u16, u16) {
        (self.source_padding.x0, self.source_padding.y0)
    }
}

#[derive(Debug, Default)]
pub(crate) struct CommandRecorder {
    /// The commands of the root layer.
    pub(crate) root_cmds: Vec<RecordedCmd>,
    /// Data about recorded layers, indexed by their ID.
    pub(crate) layers: Vec<RecordedLayer>,
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
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
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

    #[inline]
    pub(crate) fn push_fill(
        &mut self,
        strip_range: Range<usize>,
        strips: &[Strip],
        viewport_width: u16,
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

        self.record_bbox(|| strip_bbox(strips, viewport_width));
    }

    pub(crate) fn push_layer(&mut self, props: LayerProps, filter_plan: Option<FilterData>) {
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

    pub(crate) fn pop_layer(&mut self) -> PoppedLayer {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoppedLayer {
    Regular,
    Filter,
}

fn strip_bbox(strips: &[Strip], viewport_width: u16) -> RectU16 {
    let mut bbox = RectU16::INVERTED;

    // Need at least one strip (and the sentinel one).
    if strips.len() < 2 {
        return bbox;
    }

    // TODO: It _feels_ like this "iterating over strips code" should be possible to
    // deduplicate with other locations (i.e. the code used to generate fill commands).

    for pair in strips.windows(2) {
        let strip = pair[0];
        let next_strip = pair[1];
        if strip.is_sentinel() {
            continue;
        }

        let strip_y = strip.strip_y();
        let row_y = strip_y.saturating_mul(Tile::<vello_common::tile::SmallSize>::HEIGHT);
        let row_y1 = row_y.saturating_add(Tile::<vello_common::tile::SmallSize>::HEIGHT);
        let strip_width = strip.width_to(&next_strip);
        let strip_x1 = strip.x.saturating_add(strip_width);

        if strip_width > 0 {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() {
            // TODO: We should probably not emit sentinel strips with fill_gap = true
            // in the first place... Then we don't have to pass `viewport_width` to this
            // method.
            let fill_x1 = if next_strip.is_sentinel() {
                viewport_width
            } else {
                next_strip.x
            };
            if strip_x1 < fill_x1 {
                bbox.union(RectU16::new(strip_x1, row_y, fill_x1, row_y1));
            }
        }
    }

    bbox
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::palette::css::BLACK;
    use vello_common::filter_effects::FilterPrimitive;
    use vello_common::paint::PremulColor;

    enum ExpectedCmd {
        PushLayer(usize),
        FilterLayer(usize),
        Fill,
        PopLayer,
    }

    fn sentinel(y: u16, alpha_idx: u32) -> Strip {
        Strip::new(u16::MAX, y, alpha_idx, false)
    }

    fn fill_gap_sentinel(y: u16, alpha_idx: u32) -> Strip {
        Strip::new(u16::MAX, y, alpha_idx, true)
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
            0..2,
            &[
                Strip::new(0, 0, 0, false),
                Strip::new(u16::MAX, 0, 16, true),
            ],
            64,
            Paint::Solid(PremulColor::from_alpha_color(BLACK)),
            BlendMode::default(),
            None,
            0,
        );

        assert_eq!(recorder.pop_layer(), PoppedLayer::Regular);
        assert_eq!(recorder.pop_layer(), PoppedLayer::Filter);

        recorder.push_layer(layer_props(), None);
        recorder.push_fill(
            0..2,
            &[
                Strip::new(0, 0, 16, false),
                Strip::new(u16::MAX, 0, 32, true),
            ],
            64,
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
    #[test]
    fn empty_strip_bbox() {
        let strips = [sentinel(0, 0), sentinel(0, 0)];

        assert_eq!(strip_bbox(&strips, 32), RectU16::INVERTED);
    }

    #[test]
    fn single_strip_bbox() {
        let strips = [
            Strip::new(8, 4, 0, false),
            sentinel(4, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 4),
        ];

        assert_eq!(strip_bbox(&strips, 32), RectU16::new(8, 4, 12, 8));
    }

    #[test]
    fn strip_with_fill_bbox() {
        let strips = [
            Strip::new(4, 0, 0, false),
            Strip::new(20, 0, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 4, true),
            sentinel(0, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips, 32), RectU16::new(4, 0, 24, 4));
    }

    #[test]
    fn strip_with_sentinel_fill_gap_bbox_is_clamped_to_viewport() {
        let strips = [
            Strip::new(4, 0, 0, false),
            fill_gap_sentinel(0, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 4),
        ];

        assert_eq!(strip_bbox(&strips, 32), RectU16::new(4, 0, 32, 4));
    }

    #[test]
    fn strips_with_multiple_rows_bbox() {
        let strips = [
            Strip::new(12, 0, 0, false),
            sentinel(0, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 4),
            Strip::new(4, 8, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 4, false),
            sentinel(8, u32::from(Tile::<vello_common::tile::SmallSize>::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips, 32), RectU16::new(4, 0, 16, 12));
    }
}
