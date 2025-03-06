// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{BumpAllocators, LineSoup, Path, SegmentCount, Tile};

use super::{
    CpuBinding,
    util::{ONE_MINUS_ULP, ROBUST_EPSILON, Vec2, span},
};

const TILE_SCALE: f32 = 1.0 / 16.0;

fn path_count_main(
    bump: &mut BumpAllocators,
    lines: &[LineSoup],
    paths: &[Path],
    tile: &mut [Tile],
    seg_counts: &mut [SegmentCount],
) {
    for line_ix in 0..bump.lines {
        let line = lines[line_ix as usize];
        let p0 = Vec2::from_array(line.p0);
        let p1 = Vec2::from_array(line.p1);
        let is_down = p1.y >= p0.y;
        let (xy0, xy1) = if is_down { (p0, p1) } else { (p1, p0) };
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
        let count_x = span(s0.x, s1.x) - 1;
        let count = count_x + span(s0.y, s1.y);

        let dx = (s1.x - s0.x).abs();
        let dy = s1.y - s0.y;
        if dx + dy == 0.0 {
            continue;
        }
        if dy == 0.0 && s0.y.floor() == s0.y {
            continue;
        }
        let idxdy = 1.0 / (dx + dy);
        let mut a = dx * idxdy;
        let is_positive_slope = s1.x >= s0.x;
        let sign = if is_positive_slope { 1.0 } else { -1.0 };
        let xt0 = (s0.x * sign).floor();
        let c = s0.x * sign - xt0;
        let y0 = s0.y.floor();
        let ytop = if s0.y == s1.y { s0.y.ceil() } else { y0 + 1.0 };
        let b = ((dy * c + dx * (ytop - s0.y)) * idxdy).min(ONE_MINUS_ULP);
        let robust_err = (a * (count as f32 - 1.0) + b).floor() - count_x as f32;
        if robust_err != 0.0 {
            a -= ROBUST_EPSILON.copysign(robust_err);
        }
        let x0 = xt0 * sign + if is_positive_slope { 0.0 } else { -1.0 };

        let path = paths[line.path_ix as usize];
        let bbox = path.bbox;
        let bbox = [
            bbox[0] as i32,
            bbox[1] as i32,
            bbox[2] as i32,
            bbox[3] as i32,
        ];
        let xmin = s0.x.min(s1.x);
        let stride = bbox[2] - bbox[0];
        if s0.y >= bbox[3] as f32 || s1.y < bbox[1] as f32 || xmin >= bbox[2] as f32 || stride == 0
        {
            continue;
        }
        // Clip line to bounding box. Clipping is done in "i" space.
        let mut imin = 0;
        if s0.y < bbox[1] as f32 {
            let mut iminf = ((bbox[1] as f32 - y0 + b - a) / (1.0 - a)).round() - 1.0;
            if y0 + iminf - (a * iminf + b).floor() < bbox[1] as f32 {
                iminf += 1.0;
            }
            imin = iminf as u32;
        }
        let mut imax = count;
        if s1.y > bbox[3] as f32 {
            let mut imaxf = ((bbox[3] as f32 - y0 + b - a) / (1.0 - a)).round() - 1.0;
            if y0 + imaxf - (a * imaxf + b).floor() < bbox[3] as f32 {
                imaxf += 1.0;
            }
            imax = imaxf as u32;
        }
        let delta = if is_down { -1 } else { 1 };
        let mut ymin = 0;
        let mut ymax = 0;
        if s0.x.max(s1.x) < bbox[0] as f32 {
            ymin = s0.y.ceil() as i32;
            ymax = s1.y.ceil() as i32;
            imax = imin;
        } else {
            let fudge = if is_positive_slope { 0.0 } else { 1.0 };
            if xmin < bbox[0] as f32 {
                let mut f = ((sign * (bbox[0] as f32 - x0) - b + fudge) / a).round();
                if (x0 + sign * (a * f + b).floor() < bbox[0] as f32) == is_positive_slope {
                    f += 1.0;
                }
                let ynext = (y0 + f - (a * f + b).floor() + 1.0) as i32;
                if is_positive_slope {
                    if f as u32 > imin {
                        ymin = (y0 + if y0 == s0.y { 0.0 } else { 1.0 }) as i32;
                        ymax = ynext;
                        imin = f as u32;
                    }
                } else if (f as u32) < imax {
                    ymin = ynext;
                    ymax = s1.y.ceil() as i32;
                    imax = f as u32;
                }
            }
            if s0.x.max(s1.x) > bbox[2] as f32 {
                let mut f = ((sign * (bbox[2] as f32 - x0) - b + fudge) / a).round();
                if (x0 + sign * (a * f + b).floor() < bbox[2] as f32) == is_positive_slope {
                    f += 1.0;
                }
                if is_positive_slope {
                    imax = imax.min(f as u32);
                } else {
                    imin = imin.max(f as u32);
                }
            }
        }
        imax = imin.max(imax);
        ymin = ymin.max(bbox[1]);
        ymax = ymax.min(bbox[3]);
        for y in ymin..ymax {
            let base = path.tiles as i32 + (y - bbox[1]) * stride;
            tile[base as usize].backdrop += delta;
        }
        let mut last_z = (a * (imin as f32 - 1.0) + b).floor();
        let seg_base = bump.seg_counts;
        bump.seg_counts += imax - imin;
        for i in imin..imax {
            let zf = a * i as f32 + b;
            let z = zf.floor();
            let y = (y0 + i as f32 - z) as i32;
            let x = (x0 + sign * z) as i32;
            let base = path.tiles as i32 + (y - bbox[1]) * stride - bbox[0];
            let top_edge = if i == 0 { y0 == s0.y } else { last_z == z };
            if top_edge && x + 1 < bbox[2] {
                let x_bump = (x + 1).max(bbox[0]);
                tile[(base + x_bump) as usize].backdrop += delta;
            }
            // .segments is another name for the .count field; it's overloaded
            let seg_within_slice = tile[(base + x) as usize].segment_count_or_ix;
            tile[(base + x) as usize].segment_count_or_ix += 1;
            let counts = (seg_within_slice << 16) | i;
            let seg_count = SegmentCount { line_ix, counts };
            seg_counts[(seg_base + i - imin) as usize] = seg_count;
            last_z = z;
        }
    }
}

pub fn path_count(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    // config is binding 0
    let mut bump = resources[1].as_typed_mut();
    let lines = resources[2].as_slice();
    let paths = resources[3].as_slice();
    let mut tile = resources[4].as_slice_mut();
    let mut seg_counts = resources[5].as_slice_mut();
    path_count_main(&mut bump, &lines, &paths, &mut tile, &mut seg_counts);
}
