// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{BumpAllocators, LineSoup, Path, PathSegment, SegmentCount, Tile};

use super::util::{span, Vec2};

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;
const TILE_SCALE: f32 = 1.0 / 16.0;

pub fn path_tiling(
    bump: &mut BumpAllocators,
    seg_counts: &[SegmentCount],
    lines: &[LineSoup],
    paths: &[Path],
    tiles: &[Tile],
    segments: &mut [PathSegment],
) {
    for seg_ix in 0..bump.seg_counts {
        let seg_count = seg_counts[seg_ix as usize];
        let line = lines[seg_count.line_ix as usize];
        let counts = seg_count.counts;
        let seg_within_slice = counts >> 16;
        let seg_within_line = counts & 0xffff;

        // coarse rasterization logic
        let p0 = Vec2::from_array(line.p0);
        let p1 = Vec2::from_array(line.p1);
        let is_down = p1.y >= p0.y;
        let (mut xy0, mut xy1) = if is_down { (p0, p1) } else { (p1, p0) };
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
        let count = span(s0.x, s1.x) + span(s0.y, s1.y) - 1;

        let dx = (s1.x - s0.x).abs();
        let dy = s1.y - s0.y;
        let idxdy = (dx + dy).recip();
        let a = dx * idxdy;
        let is_positive_slope = s1.x >= s0.x;
        let sign = if is_positive_slope { 1.0 } else { -1.0 };
        let xt0 = (s0.x * sign).floor();
        let c = s0.x * sign - xt0;
        let y0 = s0.y.floor();
        let ytop = if s0.y == s1.y { s0.y.ceil() } else { y0 + 1.0 };
        let b = (dy * c + dx * (ytop - s0.y)) * idxdy;
        let x0 = xt0 * sign + if is_positive_slope { 0.0 } else { -1.0 };
        let z = (a * seg_within_line as f32 + b).floor();
        let x = x0 as i32 + (sign * z) as i32;
        let y = (y0 + seg_within_line as f32 - z) as i32;

        let path = paths[line.path_ix as usize];
        let bbox = path.bbox;
        let bbox = [
            bbox[0] as i32,
            bbox[1] as i32,
            bbox[2] as i32,
            bbox[3] as i32,
        ];
        let stride = bbox[2] - bbox[0];
        let tile_ix = path.tiles as i32 + (y - bbox[1]) * stride + x - bbox[0];
        let tile = tiles[tile_ix as usize];
        let tile_xy = Vec2 {
            x: x as f32 * TILE_WIDTH as f32,
            y: y as f32 * TILE_HEIGHT as f32,
        };
        let tile_xy1 = tile_xy
            + Vec2 {
                x: TILE_WIDTH as f32,
                y: TILE_HEIGHT as f32,
            };
        let mut y_edge = 1e9;

        if seg_within_line > 0 {
            let z_prev = (a * (seg_within_line as f32 - 1.0) + b).floor();
            if z == z_prev {
                // Top edge is clipped
                let mut xt = xy0.x + (xy1.x - xy0.x) * (tile_xy.y - xy0.y) / (xy1.y - xy0.y);
                xt = xt.clamp(tile_xy.x + 1e-3, tile_xy1.x);
                xy0 = Vec2 {
                    x: xt,
                    y: tile_xy.y,
                };
            } else {
                // If is_positive_slope, left edge is clipped, otherwise right
                let x_clip = if is_positive_slope {
                    tile_xy.x
                } else {
                    tile_xy1.x
                };
                let mut yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
                yt = yt.clamp(tile_xy.y + 1e-3, tile_xy1.y);
                xy1 = Vec2 {
                    x: tile_xy.y + 1e-3,
                    y: yt,
                };
                if is_positive_slope {
                    y_edge = yt;
                }
            }
        }
        if seg_within_line < count - 1 {
            let z_next = (a * (seg_within_line as f32 + 1.0) + b).floor();
            if z == z_next {
                // Bottom edge is clipped
                let mut xt = xy0.x + (xy1.x - xy0.x) * (tile_xy1.y - xy0.y) / (xy1.y - xy0.y);
                xt = xt.clamp(tile_xy.x + 1e-3, tile_xy1.x);
                xy1 = Vec2 {
                    x: xt,
                    y: tile_xy1.y,
                };
            } else {
                // If is_positive_slope, right edge is clipped, otherwise left
                let x_clip = if is_positive_slope {
                    tile_xy1.x
                } else {
                    tile_xy.x
                };
                let mut yt = xy0.y + (xy1.y - xy0.y) * (x_clip - xy0.x) / (xy1.x - xy0.x);
                yt = yt.clamp(tile_xy.y + 1e-3, tile_xy1.y);
                xy1 = Vec2 {
                    x: tile_xy.y + 1e-3,
                    y: yt,
                };
                if !is_positive_slope {
                    y_edge = yt;
                }
            }
        }
        if !is_down {
            (xy0, xy1) = (xy1, xy0);
        }
        let segment = PathSegment {
            origin: xy0.to_array(),
            delta: (xy1 - xy0).to_array(),
            y_edge,
        };
        segments[(tile.segments + seg_within_slice) as usize] = segment;
    }
}
