// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::tile::Tile;

const DEPTH_BUCKET_WIDTH: u16 = 32;

#[derive(Debug, Clone)]
pub(crate) enum Cmd {
    Fill(FillCmd),
    AlphaFill(AlphaFillCmd),
    PushLayer,
    PopBuf,
    Opacity(f32),
    Mask(Mask),
    BlendFill(BlendFillCmd),
    #[allow(
        dead_code,
        reason = "will be used once layer clip alphas are emitted as row commands"
    )]
    BlendAlphaFill(BlendAlphaFillCmd),
}

impl Cmd {
    #[inline(always)]
    fn generated_span(&self) -> Option<(u16, u16)> {
        match self {
            Self::Fill(cmd) => Some((cmd.x, cmd.width)),
            Self::AlphaFill(cmd) => Some((cmd.x, cmd.width)),
            Self::BlendFill(cmd) => Some((cmd.x, cmd.width)),
            Self::BlendAlphaFill(cmd) => Some((cmd.x, cmd.width)),
            Self::PushLayer | Self::PopBuf | Self::Opacity(_) | Self::Mask(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn fill_x(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.x,
            Self::AlphaFill(cmd) => cmd.x,
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn fill_width(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.width,
            Self::AlphaFill(cmd) => cmd.width,
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn fill_attrs_idx(&self) -> u32 {
        match self {
            Self::Fill(cmd) => cmd.attrs_idx,
            Self::AlphaFill(cmd) => cmd.attrs_idx,
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) blend_mode: BlendMode,
}

#[derive(Debug, Clone)]
pub(crate) struct BlendAlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) blend_mode: BlendMode,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedFill {
    pub(crate) x: u16,
    pub(crate) width: u16,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedAlphaFill {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FillAttrs {
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
    pub(crate) path_id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerClip {
    pub(crate) strip_range: Range<usize>,
    pub(crate) bbox: RectU16,
}

#[derive(Debug, Clone)]
struct ActiveLayer {
    mask: Option<Mask>,
    blend_mode: BlendMode,
    opacity: f32,
    clip: Option<LayerClip>,
    bbox: RectU16,
    occupied_rows: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<Cmd>,
    pub(crate) opaque: Vec<FillCmd>,
    bounds: Option<(u16, u16)>,
    opaque_bounds: Option<(u16, u16)>,
    max_opaque_path_id: u32,
    layer_depth: usize,
}

impl RowCommands {
    fn new() -> Self {
        Self {
            cmds: Vec::new(),
            opaque: Vec::new(),
            bounds: None,
            opaque_bounds: None,
            max_opaque_path_id: 0,
            layer_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
        self.bounds = None;
        self.opaque_bounds = None;
        self.max_opaque_path_id = 0;
        self.layer_depth = 0;
    }

    fn push_cmd(&mut self, cmd: Cmd, width: u16) {
        if let Some((x, cmd_width)) = cmd.generated_span() {
            self.include_bounds(x, cmd_width, width);
        }
        self.cmds.push(cmd);
    }

    fn push_layer(&mut self) {
        self.cmds.push(Cmd::PushLayer);
        self.layer_depth += 1;
    }

    fn pop_layer(
        &mut self,
        x: u16,
        width: u16,
        mask: Option<&Mask>,
        opacity: f32,
        blend_mode: BlendMode,
    ) {
        if let Some(mask) = mask {
            self.cmds.push(Cmd::Mask(mask.clone()));
        }
        if opacity != 1.0 {
            self.cmds.push(Cmd::Opacity(opacity));
        }
        self.cmds.push(Cmd::BlendFill(BlendFillCmd {
            x,
            width,
            blend_mode,
        }));
        self.pop_buf();
    }

    fn push_layer_props(&mut self, mask: Option<&Mask>, opacity: f32) {
        if let Some(mask) = mask {
            self.cmds.push(Cmd::Mask(mask.clone()));
        }
        if opacity != 1.0 {
            self.cmds.push(Cmd::Opacity(opacity));
        }
    }

    fn push_blend_fill(&mut self, fill: GeneratedFill, blend_mode: BlendMode, full_width: u16) {
        self.push_cmd(
            Cmd::BlendFill(BlendFillCmd {
                x: fill.x,
                width: fill.width,
                blend_mode,
            }),
            full_width,
        );
    }

    fn push_blend_alpha_fill(
        &mut self,
        fill: GeneratedAlphaFill,
        blend_mode: BlendMode,
        full_width: u16,
    ) {
        self.push_cmd(
            Cmd::BlendAlphaFill(BlendAlphaFillCmd {
                x: fill.x,
                width: fill.width,
                alpha_idx: fill.alpha_idx,
                blend_mode,
            }),
            full_width,
        );
    }

    fn pop_buf(&mut self) {
        self.cmds.push(Cmd::PopBuf);
        self.layer_depth -= 1;
    }

    fn push_opaque(&mut self, cmd: FillCmd, width: u16, path_id: u32) {
        self.include_bounds(cmd.x, cmd.width, width);
        self.include_opaque_bounds(cmd.x, cmd.width, width);
        self.max_opaque_path_id = self.max_opaque_path_id.max(path_id);
        self.opaque.push(cmd);
    }

    pub(crate) fn bounds(&self) -> Option<(u16, u16)> {
        self.bounds
    }

    pub(crate) fn depth_affects(&self, x: u16, cmd_width: u16, path_id: u32) -> bool {
        if path_id >= self.max_opaque_path_id {
            return false;
        }

        let Some((opaque_start, opaque_end)) = self.opaque_bounds else {
            return false;
        };

        let end = x.saturating_add(cmd_width);
        if x >= opaque_end || end <= opaque_start {
            return false;
        }

        true
    }

    fn include_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.bounds = Some(match self.bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }

    fn include_opaque_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.opaque_bounds = Some(match self.opaque_bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    clip_bboxes: Vec<RectU16>,
    rows: Vec<RowCommands>,
    attrs: Vec<FillAttrs>,
    active_layers: Vec<ActiveLayer>,
    next_path_id: u32,
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

    fn ceil_to_tile_width(width: u16) -> u16 {
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
                        bucketer.rows[row_idx].push_blend_alpha_fill(fill, blend_mode, full_width);
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
    fn ensure_row_layers(&mut self, row_idx: usize) {
        let layer_depth = self.rows[row_idx].layer_depth;
        if layer_depth == self.active_layers.len() {
            return;
        }

        for layer_idx in layer_depth..self.active_layers.len() {
            self.rows[row_idx].push_layer();
            self.active_layers[layer_idx].occupied_rows.push(row_idx);
        }
    }

    pub(crate) fn generate_fill(
        &mut self,
        strip_buf: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        let path_id = self.next_path_id;
        self.next_path_id = self
            .next_path_id
            .checked_add(1)
            .expect("row-bucket path ID overflow");
        let attrs_idx = self.attrs.len() as u32;
        self.attrs.push(FillAttrs {
            paint: paint.clone(),
            blend_mode,
            mask: mask.clone(),
            path_id,
        });
        let depth_cull_path_id = (self.active_layers.is_empty()
            && blend_mode == BlendMode::default()
            && mask.is_none()
            && paint_is_opaque(&paint, encoded_paints))
        .then_some(path_id);
        self.generate(
            strip_buf,
            |bucketer, row_idx, fill| {
                bucketer.push_fill(row_idx, fill, attrs_idx, depth_cull_path_id)
            },
            |bucketer, row_idx, fill| {
                bucketer.ensure_row_layers(row_idx);
                let full_width = bucketer.width();
                bucketer.rows[row_idx].push_cmd(
                    Cmd::AlphaFill(AlphaFillCmd {
                        x: fill.x,
                        width: fill.width,
                        alpha_idx: fill.alpha_idx,
                        attrs_idx,
                    }),
                    full_width,
                );
            },
        );
    }

    pub(crate) fn generate<F, A>(
        &mut self,
        strip_buf: &[Strip],
        mut fill_cmd: F,
        mut alpha_fill_cmd: A,
    ) where
        F: FnMut(&mut Self, usize, GeneratedFill),
        A: FnMut(&mut Self, usize, GeneratedAlphaFill),
    {
        if strip_buf.is_empty() {
            return;
        }

        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let clip_x0 = clip_bbox.x0;
        let clip_x1 = clip_bbox.x1;
        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];
            let strip_y = strip.strip_y();
            let row_y = strip_y * Tile::HEIGHT;
            let row_y1 = row_y.saturating_add(Tile::HEIGHT);
            if row_y1 <= clip_bbox.y0 {
                continue;
            }
            if row_y >= clip_bbox.y1 {
                break;
            }

            let row_idx = strip_y as usize;
            if row_idx >= self.rows.len() {
                break;
            }

            let next_strip = &strip_buf[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let x0 = strip.x;
            let x1 = x0.saturating_add(strip_width);
            let clipped_x0 = x0.max(clip_x0);
            let clipped_x1 = x1.min(clip_x1);

            if clipped_x0 < clipped_x1 {
                let alpha_idx =
                    strip.alpha_idx() + u32::from(clipped_x0 - x0) * u32::from(Tile::HEIGHT);
                alpha_fill_cmd(
                    self,
                    row_idx,
                    GeneratedAlphaFill {
                        x: clipped_x0,
                        width: clipped_x1 - clipped_x0,
                        alpha_idx,
                    },
                );
            }

            if next_strip.fill_gap() && strip_y == next_strip.strip_y() {
                let fill_x0 = x1.max(clip_x0);
                let fill_x1 = next_strip.x.min(clip_x1);
                if fill_x0 < fill_x1 {
                    fill_cmd(
                        self,
                        row_idx,
                        GeneratedFill {
                            x: fill_x0,
                            width: fill_x1 - fill_x0,
                        },
                    );
                }
            }
        }
    }

    fn push_fill(
        &mut self,
        row_idx: usize,
        fill: GeneratedFill,
        attrs_idx: u32,
        depth_cull_path_id: Option<u32>,
    ) {
        self.ensure_row_layers(row_idx);
        let full_width = self.width();
        let row = &mut self.rows[row_idx];
        let Some(path_id) = depth_cull_path_id else {
            row.push_cmd(
                Cmd::Fill(FillCmd {
                    x: fill.x,
                    width: fill.width,
                    attrs_idx,
                }),
                full_width,
            );
            return;
        };

        let end = fill.x + fill.width;
        let aligned_x = fill.x.next_multiple_of(DEPTH_BUCKET_WIDTH).min(end);
        let aligned_end = (end / DEPTH_BUCKET_WIDTH) * DEPTH_BUCKET_WIDTH;

        if aligned_x >= aligned_end {
            row.push_cmd(
                Cmd::Fill(FillCmd {
                    x: fill.x,
                    width: fill.width,
                    attrs_idx,
                }),
                full_width,
            );
            return;
        }

        if fill.x < aligned_x {
            row.push_cmd(
                Cmd::Fill(FillCmd {
                    x: fill.x,
                    width: aligned_x - fill.x,
                    attrs_idx,
                }),
                full_width,
            );
        }

        if aligned_x < aligned_end {
            row.push_opaque(
                FillCmd {
                    x: aligned_x,
                    width: aligned_end - aligned_x,
                    attrs_idx,
                },
                full_width,
                path_id,
            );
        }

        if aligned_end < end {
            row.push_cmd(
                Cmd::Fill(FillCmd {
                    x: aligned_end,
                    width: end - aligned_end,
                    attrs_idx,
                }),
                full_width,
            );
        }
    }
}

fn paint_is_opaque(paint: &Paint, encoded_paints: &[EncodedPaint]) -> bool {
    match paint {
        Paint::Solid(color) => color.is_opaque(),
        Paint::Indexed(index) => match &encoded_paints[index.index()] {
            EncodedPaint::Gradient(gradient) => !gradient.may_have_transparency,
            EncodedPaint::Image(image) => !image.may_have_transparency,
            EncodedPaint::ExternalTexture(texture) => !texture.may_have_transparency,
            EncodedPaint::BlurredRoundedRect(_) => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{Cmd, CommandBucketer};
    use alloc::vec::Vec;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::encode::EncodeExt;
    use vello_common::kurbo::Affine;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::{BlendMode, ColorStop, Gradient};
    use vello_common::strip::Strip;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    #[test]
    fn opaque_fill_splits_to_aligned_middle() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(3, 0, 0, false), Strip::new(100, 0, 0, true)];

        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(RED)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].x, 32);
        assert_eq!(row.opaque[0].width, 64);
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 3 && cmd.width == 29));
        assert!(matches!(row.cmds[1], Cmd::Fill(cmd) if cmd.x == 96 && cmd.width == 4));
    }

    #[test]
    fn opaque_indexed_fill_splits_to_aligned_middle() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(3, 0, 0, false), Strip::new(100, 0, 0, true)];
        let mut encoded_paints = Vec::new();
        let paint = Gradient::new_linear((0., 0.), (128., 0.))
            .with_stops([ColorStop::from((0.0, RED)), ColorStop::from((1.0, BLUE))])
            .encode_into(&mut encoded_paints, Affine::IDENTITY, None);

        bucketer.generate_fill(&strips, paint, BlendMode::default(), None, &encoded_paints);

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].x, 32);
        assert_eq!(row.opaque[0].width, 64);
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 3 && cmd.width == 29));
        assert!(matches!(row.cmds[1], Cmd::Fill(cmd) if cmd.x == 96 && cmd.width == 4));
    }

    #[test]
    fn transparent_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(96, 0, 0, true)];

        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(AlphaColor::from_rgba8(255, 0, 0, 128))),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 0 && cmd.width == 96));
    }

    #[test]
    fn alpha_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(8, 0, 32, false)];

        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(BLUE)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], Cmd::AlphaFill(cmd) if cmd.x == 0 && cmd.width == 8 && cmd.alpha_idx == 0)
        );
    }

    #[test]
    fn short_unaligned_opaque_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(8, 0, 0, false), Strip::new(16, 0, 0, true)];

        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(RED)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 8 && cmd.width == 8));
    }
}
