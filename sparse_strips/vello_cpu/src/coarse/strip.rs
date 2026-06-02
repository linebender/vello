// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::DEPTH_BUCKET_WIDTH;
use super::bucketer::CommandBucketer;
use super::cmd::{FillAttrs, FillCmd, FineCmd, Span};
use crate::peniko::BlendMode;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::tile::Tile;

const DEPTH_BUCKET_TILE_WIDTH: u16 = DEPTH_BUCKET_WIDTH / Tile::WIDTH;
const _: () = assert!(
    DEPTH_BUCKET_WIDTH.is_multiple_of(Tile::WIDTH),
    "depth bucket width must be a multiple of tile width"
);

// Note: All methods here assume that strips are horizontally aligned to
// tile boundaries. if that ever changes, the logic will have to be rewritten.

/// A generic alpha fill command that can later on either be
/// turned into a normal fill or a blend fill.
#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedAlphaFill {
    pub(super) span: Span,
    pub(super) alpha_idx: u32,
}

impl CommandBucketer {
    pub(crate) fn generate_fill(
        &mut self,
        strip_buf: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        thread_idx: u8,
        paint_offset: (u16, u16),
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
            thread_idx,
            paint_offset,
        });
        let depth_cull_path_id = (self.active_layers.is_empty()
            && blend_mode == BlendMode::default()
            && mask.is_none()
            && paint_is_opaque(&paint, encoded_paints))
        .then_some(path_id);
        self.generate(
            strip_buf,
            |bucketer, row_idx, fill| {
                bucketer.push_fill(row_idx, fill, attrs_idx, depth_cull_path_id);
            },
            |bucketer, row_idx, fill| {
                bucketer.ensure_row_layers(row_idx);
                let full_width = bucketer.width();
                bucketer.rows[row_idx].push_cmd(
                    FineCmd::Fill(FillCmd::new(fill.span, Some(fill.alpha_idx), attrs_idx)),
                    full_width,
                );
            },
        );
    }

    pub(super) fn generate<F, A>(
        &mut self,
        strip_buf: &[Strip],
        mut fill_cmd: F,
        mut alpha_fill_cmd: A,
    ) where
        F: FnMut(&mut Self, usize, Span),
        A: FnMut(&mut Self, usize, GeneratedAlphaFill),
    {
        if strip_buf.is_empty() {
            return;
        }

        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let clip_x0 = (clip_bbox.x0 / Tile::WIDTH) * Tile::WIDTH;
        let clip_x1 = Self::ceil_to_tile_width(clip_bbox.x1).min(self.width());
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

            if strip_width > 0 && x0 < clip_x1 && x1 > clip_x0 {
                alpha_fill_cmd(
                    self,
                    row_idx,
                    GeneratedAlphaFill {
                        span: Span::new(x0 / Tile::WIDTH, strip_width / Tile::WIDTH),
                        alpha_idx: strip.alpha_idx(),
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
                        Span::new(fill_x0 / Tile::WIDTH, (fill_x1 - fill_x0) / Tile::WIDTH),
                    );
                }
            }
        }
    }

    fn push_fill(
        &mut self,
        row_idx: usize,
        span: Span,
        attrs_idx: u32,
        depth_cull_path_id: Option<u32>,
    ) {
        self.ensure_row_layers(row_idx);
        let full_width = self.width();
        let row = &mut self.rows[row_idx];
        let Some(path_id) = depth_cull_path_id else {
            row.push_cmd(
                FineCmd::Fill(FillCmd::new(span, None, attrs_idx)),
                full_width,
            );
            return;
        };

        let fill_x = span.tile_x();
        let end = span.tile_end();
        let aligned_x = fill_x.next_multiple_of(DEPTH_BUCKET_TILE_WIDTH).min(end);
        let aligned_end = (end / DEPTH_BUCKET_TILE_WIDTH) * DEPTH_BUCKET_TILE_WIDTH;

        if aligned_x >= aligned_end {
            row.push_cmd(
                FineCmd::Fill(FillCmd::new(span, None, attrs_idx)),
                full_width,
            );
            return;
        }

        if fill_x < aligned_x {
            row.push_cmd(
                FineCmd::Fill(FillCmd::new(
                    Span::new(fill_x, aligned_x - fill_x),
                    None,
                    attrs_idx,
                )),
                full_width,
            );
        }

        if aligned_x < aligned_end {
            row.push_opaque(
                FillCmd::new(
                    Span::new(aligned_x, aligned_end - aligned_x),
                    None,
                    attrs_idx,
                ),
                full_width,
                path_id,
            );
        }

        if aligned_end < end {
            row.push_cmd(
                FineCmd::Fill(FillCmd::new(
                    Span::new(aligned_end, end - aligned_end),
                    None,
                    attrs_idx,
                )),
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
    use super::CommandBucketer;
    use super::DEPTH_BUCKET_WIDTH;
    use crate::coarse::cmd::FineCmd;
    use crate::coarse::layer::LayerClip;
    use alloc::vec::Vec;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::encode::EncodeExt;
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Affine;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::{BlendMode, ColorStop, Gradient};
    use vello_common::strip::Strip;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    #[test]
    fn opaque_fill_splits_to_aligned_middle() {
        let end = DEPTH_BUCKET_WIDTH * 3 + 4;
        let mut bucketer = CommandBucketer::new(DEPTH_BUCKET_WIDTH * 4, 4);
        let strips = [Strip::new(4, 0, 0, false), Strip::new(end, 0, 0, true)];

        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(RED)),
            BlendMode::default(),
            None,
            0,
            (0, 0),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].span.pixel_x(), DEPTH_BUCKET_WIDTH);
        assert_eq!(row.opaque[0].span.pixel_width(), DEPTH_BUCKET_WIDTH * 2);
        assert_eq!(row.cmds.len(), 2);
        assert!(
            matches!(row.cmds[0], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 4 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH - 4)
        );
        assert!(
            matches!(row.cmds[1], FineCmd::Fill(cmd) if cmd.span.pixel_x() == DEPTH_BUCKET_WIDTH * 3 && cmd.span.pixel_width() == 4)
        );
    }

    #[test]
    fn opaque_indexed_fill_splits_to_aligned_middle() {
        let end = DEPTH_BUCKET_WIDTH * 3 + 4;
        let mut bucketer = CommandBucketer::new(DEPTH_BUCKET_WIDTH * 4, 4);
        let strips = [Strip::new(4, 0, 0, false), Strip::new(end, 0, 0, true)];
        let mut encoded_paints = Vec::new();
        let paint = Gradient::new_linear((0., 0.), (128., 0.))
            .with_stops([ColorStop::from((0.0, RED)), ColorStop::from((1.0, BLUE))])
            .encode_into(&mut encoded_paints, Affine::IDENTITY, None);

        bucketer.generate_fill(
            &strips,
            paint,
            BlendMode::default(),
            None,
            0,
            (0, 0),
            &encoded_paints,
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].span.pixel_x(), DEPTH_BUCKET_WIDTH);
        assert_eq!(row.opaque[0].span.pixel_width(), DEPTH_BUCKET_WIDTH * 2);
        assert_eq!(row.cmds.len(), 2);
        assert!(
            matches!(row.cmds[0], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 4 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH - 4)
        );
        assert!(
            matches!(row.cmds[1], FineCmd::Fill(cmd) if cmd.span.pixel_x() == DEPTH_BUCKET_WIDTH * 3 && cmd.span.pixel_width() == 4)
        );
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
            0,
            (0, 0),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == 96)
        );
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
            0,
            (0, 0),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == 8 && cmd.alpha_idx() == Some(0))
        );
    }

    #[test]
    fn alpha_fill_keeps_full_strip_when_clipped() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(8, 0, 32, false)];

        bucketer.push_layer(
            BlendMode::default(),
            1.0,
            None,
            Some(LayerClip {
                strip_range: 0..0,
                thread_idx: 0,
                bbox: RectU16::new(1, 0, 7, 4),
            }),
        );
        bucketer.generate_fill(
            &strips,
            Paint::Solid(color(BLUE)),
            BlendMode::default(),
            None,
            0,
            (0, 0),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], FineCmd::PushLayer));
        assert!(
            matches!(row.cmds[1], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == 8 && cmd.alpha_idx() == Some(0))
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
            0,
            (0, 0),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], FineCmd::Fill(cmd) if cmd.span.pixel_x() == 8 && cmd.span.pixel_width() == 8)
        );
    }
}
