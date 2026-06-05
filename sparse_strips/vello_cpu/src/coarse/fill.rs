// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::bucket::CommandBucketer;
use super::cmd::{PaintFill, PaintFillAttrs, RenderCmd};
use super::depth::{self, DepthSegment};
use crate::peniko::BlendMode;
use crate::util::{Span, snap_bbox_to_tile_coordinates};
use vello_common::encode::EncodedPaint;
use vello_common::strip::Strip;
use vello_common::tile::Tile;

// Note: All methods here assume that strips are horizontally aligned to
// tile boundaries. if that ever changes, the logic will have to be rewritten.

#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedAlphaFill {
    pub(super) span: Span,
    pub(super) alpha_idx: u32,
}

impl CommandBucketer {
    pub(crate) fn generate_fill(
        &mut self,
        strip_buf: &[Strip],
        attrs: &PaintFillAttrs,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        assert_ne!(attrs.draw_id, 0, "fill draw IDs should start at 1");

        let pixmap_origin = attrs.pixmap_origin;
        let attrs_idx = self.paint_fill_attrs.len() as u32;
        self.paint_fill_attrs.push(attrs.clone());

        let draw_id =
            // While in certain cases it _might_ be okay to use depth culling while inside of
            // a layer, it can get very finnicky with blend modes etc., so we just outright
            // reject those for now.
            (self.active_layers.is_empty()
            && attrs.blend_mode == BlendMode::default()
            && attrs.mask.is_none()
            && !attrs.paint.may_have_transparency(encoded_paints))
        .then_some(attrs.draw_id);

        self.generate(
            strip_buf,
            pixmap_origin,
            |bucketer, row_idx, fill| {
                bucketer.push_fill(row_idx, fill, attrs_idx, draw_id);
            },
            |bucketer, row_idx, fill| {
                bucketer.ensure_row_layers(row_idx);
                bucketer.rows[row_idx].push_cmd(RenderCmd::PaintFill(PaintFill::new(
                    fill.span,
                    Some(fill.alpha_idx),
                    attrs_idx,
                )));
            },
        );
    }

    pub(super) fn generate<F, A>(
        &mut self,
        strip_buf: &[Strip],
        pixmap_origin: (u16, u16),
        mut fill_cmd: F,
        mut alpha_fill_cmd: A,
    ) where
        F: FnMut(&mut Self, usize, Span),
        A: FnMut(&mut Self, usize, GeneratedAlphaFill),
    {
        if strip_buf.is_empty() {
            return;
        }

        let clip_bbox = snap_bbox_to_tile_coordinates(*self.clip_bboxes.last().unwrap());
        let clip_x0 = clip_bbox.x0;
        let clip_x1 = clip_bbox.x1.min(self.width());
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
            let strip_width = strip.width_to(next_strip);
            let x0 = strip_x(strip);
            let x1 = x0.saturating_add(strip_width);

            if strip_width > 0 && x0 < clip_x1 && x1 > clip_x0 {
                alpha_fill_cmd(
                    self,
                    row_idx,
                    GeneratedAlphaFill {
                        span: Span::new(x0, strip_width),
                        alpha_idx: strip.alpha_idx(),
                    },
                );
            }

            if next_strip.fill_gap() && strip_y == strip_row(next_strip) {
                let fill_x0 = x1.max(clip_x0);
                let fill_x1 = strip_x(next_strip).min(clip_x1);
                if fill_x0 < fill_x1 {
                    fill_cmd(self, row_idx, Span::new(fill_x0, fill_x1 - fill_x0));
                }
            }
        }
    }

    /// Note: If depth-culling should be disabled, pass `None` to `draw_id`.
    fn push_fill(
        &mut self,
        row_idx: usize,
        span: Span,
        attrs_idx: u32,
        draw_id: Option<u32>,
    ) {
        self.ensure_row_layers(row_idx);
        let row = &mut self.rows[row_idx];
        let draw_id = draw_id.filter(|_| row.layer_depth == 0);
        let Some(draw_id) = draw_id else {
            row.push_cmd(RenderCmd::PaintFill(PaintFill::new(span, None, attrs_idx)));
            return;
        };

        depth::split_opaque_span(span, |span, segment| match segment {
            DepthSegment::Regular => {
                row.push_cmd(RenderCmd::PaintFill(PaintFill::new(span, None, attrs_idx)));
            }
            DepthSegment::Opaque => {
                row.push_depth_write(PaintFill::new(span, None, attrs_idx), draw_id);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::CommandBucketer;
    use crate::coarse::cmd::{PaintFillAttrs, RenderCmd};
    use crate::coarse::depth::DEPTH_BUCKET_WIDTH;
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
            pixmap_origin: (0, 0),
        }
    }
    #[test]
    fn opaque_fill_uses_depth_write_when_possible() {
        let end = DEPTH_BUCKET_WIDTH * 2 + 4;
        let mut bucketer = CommandBucketer::new(end, 4);
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
        let mut bucketer = CommandBucketer::new(DEPTH_BUCKET_WIDTH, 4);
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
