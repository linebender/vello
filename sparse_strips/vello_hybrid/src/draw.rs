// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draw construction for strip render passes.

use crate::GpuStrip;
use crate::paint::{COLOR_SOURCE_LAYER, COLOR_SOURCE_SHIFT, PaintResolver};
use crate::rect::{RectPart, split_rect};
use crate::scene::{RecordedDraw, RecordedPath};
use crate::target::{DrawTarget, LayerTextureRegion};
use crate::util::{Ranges, VecExt, pack_opacity, pack_u16_pair};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::paint::Paint;
use vello_common::record::LayerClip;
use vello_common::strip::{StripAlphaFillSegment, StripFillSegment, visit_strip_fill_segments};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;
use vello_common::util::Clear;

/// Strip ranges and texture-binding state for one scheduled draw pass.
#[derive(Debug, Default, Clone)]
pub(crate) struct Draw {
    /// Ranges selecting this draw's strips from [`DrawBuffers::strips`].
    pub(crate) strip_ranges: Ranges,
    /// Runs that require an externally supplied texture binding.
    pub(crate) external_texture_runs: Vec<ExternalTextureRun>,
    /// Whether any strip in this draw samples a child layer.
    pub(crate) has_child_layer: bool,
}

impl Draw {
    #[inline]
    fn push(
        &mut self,
        strips: &mut Vec<GpuStrip>,
        gpu_strip: GpuStrip,
        external_texture_id: Option<TextureId>,
    ) {
        if let Some(texture_id) = external_texture_id {
            let needs_new_run = self
                .external_texture_runs
                .last()
                .is_none_or(|run| run.texture_id != texture_id);

            if needs_new_run {
                let strips_start = if self.external_texture_runs.is_empty() {
                    0
                } else {
                    self.strip_ranges.len()
                };

                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        strips.push_ranged(&mut self.strip_ranges, gpu_strip);
    }
}

impl Clear for Draw {
    fn clear(&mut self) {
        self.strip_ranges.clear();
        self.external_texture_runs.clear();
        self.has_child_layer = false;
    }
}

/// Appends recorded draws to a scheduled [`Draw`] and its shared buffers.
#[derive(Debug)]
pub(crate) struct DrawBuilder<'a, T: DrawTarget> {
    /// Draw whose ranges and binding state are being built.
    draw: &'a mut Draw,
    /// Shared buffer receiving alpha-blended strips.
    strips: &'a mut Vec<GpuStrip>,
    /// Shared buffer receiving root-level opaque strips.
    opaque: &'a mut Vec<GpuStrip>,
    /// Target and depth state used to encode strips.
    state: &'a mut DrawState<T>,
}

impl<'a, T: DrawTarget> DrawBuilder<'a, T> {
    pub(crate) fn new(
        draw: &'a mut Draw,
        draw_buffers: &'a mut DrawBuffers,
        state: &'a mut DrawState<T>,
    ) -> Self {
        Self {
            draw,
            strips: &mut draw_buffers.strips,
            opaque: &mut draw_buffers.opaque_strips,
            state,
        }
    }

    pub(crate) fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        paint_resolver: PaintResolver<'_>,
    ) {
        match draw {
            RecordedDraw::Path(path) => self.push_path(path, strip_storage, paint_resolver),
            RecordedDraw::Rect(rect) => {
                self.push_rect(&rect.rect, &rect.paint, paint_resolver);
            }
        }
    }

    fn push_opaque(&mut self, strip: GpuStrip) -> bool {
        if !self.state.target.enable_depth() {
            return false;
        }

        self.opaque.push(strip);
        true
    }

    fn push_path(
        &mut self,
        path: &RecordedPath,
        strip_storage: &StripStorage,
        paint_resolver: PaintResolver<'_>,
    ) {
        let strips = &strip_storage.strips[path.strips.clone()];

        let paint = paint_resolver.pack(&path.paint);
        // Note: This will also advance the depth index for layer root draws even though
        // those _currently_ never use the depth buffer, but it's better to keep the
        // condition simple.
        let depth_index = self.state.depth_counter.next(paint.opaque);
        let tile_bounds = self.state.target_bbox.to_tile_bounds();
        let geometry_shift = self.state.target.geometry_shift();

        // Note: This method will also take care of culling any strips to the active clip bbox.
        visit_strip_fill_segments(
            strips,
            tile_bounds,
            self,
            |builder, segment| {
                let shifted = segment.shift(geometry_shift);
                let strip = GpuStrip::from_fill(
                    shifted,
                    Some(segment.col_idx()),
                    paint.payload_at(segment.x0(), segment.y()),
                    paint.paint,
                    depth_index,
                );

                builder
                    .draw
                    .push(builder.strips, strip, paint.external_texture_id);
            },
            |builder, segment| {
                let shifted = segment.shift(geometry_shift);

                let strip = GpuStrip::from_fill(
                    shifted,
                    None,
                    paint.payload_at(segment.x0(), segment.y()),
                    paint.paint,
                    depth_index,
                );

                if !paint.opaque || !builder.push_opaque(strip) {
                    builder
                        .draw
                        .push(builder.strips, strip, paint.external_texture_id);
                }
            },
        );
    }

    fn push_rect(&mut self, rect: &Rect, paint: &Paint, paint_resolver: PaintResolver<'_>) {
        // Recordings might contain geometry that exceeds the actual layer
        // bounding box. This can happen when a clip path is associated with the layer.
        // Recordings will not cull those for us, so we need to do this manually here.
        // For normal paths, the `visit_strip_fill_segments` method takes care of doing this.
        // For rectangles, we can do a simple intersection.
        let clipped_rect = rect.intersect(self.state.target_bbox.as_rect());
        if clipped_rect.is_zero_area() {
            return;
        }

        let paint = paint_resolver.pack(paint);
        let depth_index = self.state.depth_counter.next(paint.opaque);

        let split = split_rect(&clipped_rect);

        for part in [
            Some(split.main),
            split.top,
            split.bottom,
            split.left,
            split.right,
        ]
        .into_iter()
        .flatten()
        {
            let shifted = part.shift(self.state.target.geometry_shift());

            let strip = GpuStrip::from_rect(
                shifted,
                paint.payload_at(part.rect.x0, part.rect.y0),
                paint.paint,
                depth_index,
            );

            if !(paint.opaque && part.frac == 0 && self.push_opaque(strip)) {
                self.draw
                    .push(self.strips, strip, paint.external_texture_id);
            }
        }
    }

    pub(crate) fn push_layer_fill(
        &mut self,
        sample: LayerTextureRegion,
        opacity: f32,
        clip_path: Option<&LayerClip>,
        strip_storage: &StripStorage,
    ) {
        let sample_bbox = sample.layer_bbox.intersect(self.state.target_bbox);
        if sample_bbox.is_empty() {
            return;
        }

        let paint = (COLOR_SOURCE_LAYER << COLOR_SOURCE_SHIFT) | u32::from(pack_opacity(opacity));
        if let Some(clip_path) = clip_path {
            let strips = &strip_storage.strips[clip_path.strip_range.clone()];
            let depth_index = self.state.depth_counter.next(false);
            let tile_bounds = sample_bbox.to_tile_bounds();

            visit_strip_fill_segments(
                strips,
                tile_bounds,
                self,
                |builder, segment| {
                    builder.push_layer_fill_segment(
                        sample,
                        *segment,
                        Some(segment.col_idx()),
                        paint,
                        depth_index,
                    );
                },
                |builder, segment| {
                    builder.push_layer_fill_segment(sample, segment, None, paint, depth_index);
                },
            );
        } else {
            let depth_index = self.state.depth_counter.next(false);

            let rect_part = RectPart {
                rect: sample_bbox.shift(self.state.target.geometry_shift()),
                frac: 0,
            };

            self.draw.has_child_layer = true;
            self.draw.push(
                self.strips,
                GpuStrip::from_rect(
                    rect_part,
                    sample.payload_at(sample_bbox.x0, sample_bbox.y0),
                    paint,
                    depth_index,
                ),
                None,
            );
        }
    }

    fn push_layer_fill_segment(
        &mut self,
        sample: LayerTextureRegion,
        segment: StripFillSegment,
        col_idx: Option<u32>,
        paint: u32,
        depth_index: u32,
    ) {
        self.draw.has_child_layer = true;
        let payload = sample.payload_at(segment.x0(), segment.y());
        let shifted = segment.shift(self.state.target.geometry_shift());
        let strip = GpuStrip::from_fill(shifted, col_idx, payload, paint, depth_index);
        self.draw.push(self.strips, strip, None);
    }
}

/// Reusable strip storage shared by all draws in a schedule.
#[derive(Debug, Default)]
pub(crate) struct DrawBuffers {
    /// Opaque root strips rendered in the early depth-writing pass.
    pub(crate) opaque_strips: Vec<GpuStrip>,
    /// Alpha-blended strips selected by each draw's ranges.
    pub(crate) strips: Vec<GpuStrip>,
}

impl DrawBuffers {
    pub(crate) fn clear(&mut self) {
        self.opaque_strips.clear();
        self.strips.clear();
    }
}

/// Target-specific state used while encoding a draw.
#[derive(Debug)]
pub(crate) struct DrawState<T: DrawTarget> {
    /// Destination into which strips will be rendered.
    pub(crate) target: T,
    /// Assigns depth values to opaque strips.
    depth_counter: DepthCounter,
    /// Scene-space bounds visible in the target.
    pub(crate) target_bbox: RectU16,
}

impl<T: DrawTarget> DrawState<T> {
    pub(crate) fn new(target: T, target_bbox: RectU16) -> Self {
        Self {
            target,
            depth_counter: DepthCounter::default(),
            target_bbox,
        }
    }
}

/// Bit 31 of [`GpuStrip::paint_and_rect_flag`] signals that the strip
/// represents a full rectangle.
const RECT_STRIP_FLAG: u32 = 1 << 31;

impl GpuStrip {
    fn from_fill(
        rect: RectU16,
        col_idx: Option<u32>,
        payload: u32,
        paint: u32,
        depth_index: u32,
    ) -> Self {
        let width = rect.width();
        let (dense_width_or_rect_height, col_idx_or_rect_frac) = if let Some(col_idx) = col_idx {
            (width, col_idx)
        } else {
            (0, 0)
        };

        Self {
            x: rect.x0,
            y: rect.y0,
            width,
            dense_width_or_rect_height,
            col_idx_or_rect_frac,
            payload,
            paint_and_rect_flag: paint,
            depth_index,
        }
    }

    fn from_rect(part: RectPart, payload: u32, paint: u32, depth_index: u32) -> Self {
        Self {
            x: part.rect.x0,
            y: part.rect.y0,
            width: part.rect.width(),
            dense_width_or_rect_height: part.rect.height(),
            col_idx_or_rect_frac: part.frac,
            payload,
            paint_and_rect_flag: paint | RECT_STRIP_FLAG,
            depth_index,
        }
    }
}

impl LayerTextureRegion {
    fn payload_at(self, x: u16, y: u16) -> u32 {
        let shift = self.geometry_shift();
        // This should never fail. The shift itself can be negative if the layer bbox doesn't
        // start at 0, but we only sample values that are within the layer bbox.
        let source_x = u16::try_from(x as i32 + shift.0).unwrap();
        let source_y = u16::try_from(y as i32 + shift.1).unwrap();

        pack_u16_pair(source_x, source_y)
    }
}

/// Specifies a run of strips inside a draw that can be drawn with the same external texture
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    /// External texture bound for the run.
    pub(crate) texture_id: TextureId,
    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

/// Assigns monotonically increasing depth values to opaque strips.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DepthCounter {
    /// Number of opaque strips assigned so far.
    count: u32,
}

impl DepthCounter {
    #[inline(always)]
    fn next(&mut self, opaque: bool) -> u32 {
        self.count += opaque as u32;

        self.count
    }
}

trait RectU16Ext {
    fn to_tile_bounds(self) -> RectU16;
}

impl RectU16Ext for RectU16 {
    fn to_tile_bounds(self) -> RectU16 {
        debug_assert!(
            self.x0.is_multiple_of(Tile::WIDTH)
                && self.x1.is_multiple_of(Tile::WIDTH)
                && self.y0.is_multiple_of(Tile::HEIGHT)
                && self.y1.is_multiple_of(Tile::HEIGHT),
            "draw bounding boxes must be tile-aligned"
        );

        Self::new(
            self.x0 / Tile::WIDTH,
            self.y0 / Tile::HEIGHT,
            self.x1 / Tile::WIDTH,
            self.y1 / Tile::HEIGHT,
        )
    }
}

trait StripAlphaFillSegmentExt {
    fn col_idx(self) -> u32;
}

impl StripAlphaFillSegmentExt for StripAlphaFillSegment {
    #[inline(always)]
    fn col_idx(self) -> u32 {
        self.alpha_idx / u32::from(Tile::HEIGHT)
    }
}

#[cfg(test)]
mod tests {
    use super::{Draw, DrawBuffers, DrawBuilder, DrawState, ExternalTextureRun};
    use crate::paint::PaintResolver;
    use crate::scene::{RecordedDraw, RecordedRect};
    use crate::target::{
        DrawTarget, LayerTextureId, LayerTextureRegion, RootTarget, TextureParity, TextureRegion,
    };
    use crate::util::VecExt;
    use alloc::vec::Vec;
    use vello_common::TextureId;
    use vello_common::encode::{EncodedExternalTexture, EncodedPaint};
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, Rect};
    use vello_common::paint::{IndexedPaint, Paint, PremulColor};
    use vello_common::peniko::color::palette::css::BLUE;
    use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
    use vello_common::strip_generator::StripStorage;
    use vello_common::util::Clear;

    struct DrawCase<T: DrawTarget> {
        buffers: DrawBuffers,
        state: DrawState<T>,
        strip_storage: StripStorage,
    }

    impl<T: DrawTarget> DrawCase<T> {
        fn new(target: T, bbox: RectU16) -> Self {
            Self {
                buffers: DrawBuffers::default(),
                state: DrawState::new(target, bbox),
                strip_storage: StripStorage::default(),
            }
        }

        fn rect(
            &mut self,
            draw: &mut Draw,
            rect: Rect,
            paint: Paint,
            paint_resolver: PaintResolver<'_>,
        ) {
            let recorded = RecordedDraw::Rect(RecordedRect { rect, paint });
            DrawBuilder::new(draw, &mut self.buffers, &mut self.state).push_draw(
                &recorded,
                &self.strip_storage,
                paint_resolver,
            );
        }

        fn layer(&mut self, draw: &mut Draw, sample: LayerTextureRegion) {
            DrawBuilder::new(draw, &mut self.buffers, &mut self.state).push_layer_fill(
                sample,
                1.0,
                None,
                &self.strip_storage,
            );
        }
    }

    fn solid(alpha: f32) -> Paint {
        Paint::Solid(PremulColor::from_alpha_color(BLUE.with_alpha(alpha)))
    }

    fn indexed(index: usize) -> Paint {
        Paint::Indexed(IndexedPaint::new(index))
    }

    fn rect(x: f64) -> Rect {
        Rect::new(x, 0.0, x + 4.0, 4.0)
    }

    fn no_paints() -> PaintResolver<'static> {
        PaintResolver::new(&[], &[])
    }

    fn external(texture_id: TextureId) -> EncodedPaint {
        EncodedPaint::ExternalTexture(EncodedExternalTexture {
            texture_id,
            source_region: RectU16::new(0, 0, 8, 8),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
            may_have_transparency: true,
            transform: Affine::IDENTITY,
            tint: None,
        })
    }

    fn layer(layer_bbox: RectU16) -> LayerTextureRegion {
        LayerTextureRegion {
            texture: TextureRegion {
                target: LayerTextureId::new(TextureParity::Even, 0),
                rect: RectU16::new(0, 0, layer_bbox.width(), layer_bbox.height()),
            },
            layer_bbox,
        }
    }

    #[test]
    fn texture_runs() {
        let texture_a = TextureId(10);
        let texture_b = TextureId(20);
        let encoded = [external(texture_a), external(texture_b)];
        let offsets = [0, 0];
        let resolver = PaintResolver::new(&encoded, &offsets);
        let mut case = DrawCase::new(RootTarget::UserSurface, RectU16::new(0, 0, 32, 8));
        let mut draw = Draw::default();
        let mut other = Draw::default();

        for (x, paint_index) in [(0.0, 0), (4.0, 0)] {
            case.rect(&mut draw, rect(x), indexed(paint_index), resolver);
        }
        case.rect(&mut other, rect(8.0), solid(0.5), no_paints());
        for (x, paint_index) in [(12.0, 1), (16.0, 1), (20.0, 0)] {
            case.rect(&mut draw, rect(x), indexed(paint_index), resolver);
        }

        assert_eq!(
            draw.external_texture_runs,
            [
                ExternalTextureRun {
                    texture_id: texture_a,
                    strips_start: 0,
                },
                ExternalTextureRun {
                    texture_id: texture_b,
                    strips_start: 2,
                },
                ExternalTextureRun {
                    texture_id: texture_a,
                    strips_start: 4,
                },
            ]
        );
    }

    #[test]
    fn draw_clear() {
        let texture_id = TextureId(10);
        let encoded = [external(texture_id)];
        let offsets = [0];
        let resolver = PaintResolver::new(&encoded, &offsets);
        let mut case = DrawCase::new(RootTarget::UserSurface, RectU16::new(0, 0, 8, 8));
        let mut draw = Draw::default();

        case.rect(&mut draw, rect(0.0), indexed(0), resolver);
        case.layer(&mut draw, layer(RectU16::new(0, 0, 8, 8)));
        assert_eq!(draw.strip_ranges.len(), 2);
        assert_eq!(draw.external_texture_runs.len(), 1);
        assert!(draw.has_child_layer);

        draw.clear();

        assert_eq!(draw.strip_ranges.len(), 0);
        assert!(draw.external_texture_runs.is_empty());
        assert!(!draw.has_child_layer);

        case.rect(&mut draw, rect(0.0), indexed(0), resolver);
        assert_eq!(draw.strip_ranges.len(), 1);
        assert_eq!(draw.external_texture_runs[0].strips_start, 0);
    }

    #[test]
    fn opaque_routing() {
        let mut user_case = DrawCase::new(RootTarget::UserSurface, RectU16::new(0, 0, 16, 8));
        let mut user_draw = Draw::default();
        user_case.rect(&mut user_draw, rect(0.0), solid(1.0), no_paints());
        user_case.rect(
            &mut user_draw,
            Rect::new(8.25, 0.0, 12.0, 4.0),
            solid(1.0),
            no_paints(),
        );

        assert_eq!(user_case.buffers.opaque_strips.len(), 1);
        assert_eq!(user_draw.strip_ranges.len(), 1);

        let mut atlas_case = DrawCase::new(RootTarget::AtlasLayer, RectU16::new(0, 0, 8, 8));
        let mut atlas_draw = Draw::default();
        atlas_case.rect(&mut atlas_draw, rect(0.0), solid(1.0), no_paints());

        assert!(atlas_case.buffers.opaque_strips.is_empty());
        assert_eq!(atlas_draw.strip_ranges.len(), 1);
    }

    #[test]
    fn child_binding() {
        let mut case = DrawCase::new(RootTarget::UserSurface, RectU16::new(0, 0, 8, 8));
        let mut draw = Draw::default();

        case.layer(&mut draw, layer(RectU16::new(0, 0, 8, 8)));
        assert!(draw.has_child_layer);
        assert_eq!(draw.strip_ranges.len(), 1);
    }

    #[test]
    fn depth_progression() {
        let mut case = DrawCase::new(RootTarget::UserSurface, RectU16::new(0, 0, 64, 8));
        let mut draw = Draw::default();
        let opacity = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5];
        let positions = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0];

        for (x, alpha) in positions.into_iter().zip(opacity) {
            case.rect(&mut draw, rect(x), solid(alpha), no_paints());
        }

        assert_eq!(
            case.buffers
                .strips
                .ranged(&draw.strip_ranges)
                .iter()
                .map(|strip| strip.depth_index)
                .collect::<Vec<_>>(),
            [0, 0, 1, 1, 2]
        );
        assert_eq!(
            case.buffers
                .opaque_strips
                .iter()
                .map(|strip| strip.depth_index)
                .collect::<Vec<_>>(),
            [1, 2]
        );
    }
}
