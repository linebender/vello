// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reference rasterizer: exact per-sample scanline evaluation of the MSAA16 masks.
//!
//! This is deliberately simple and slow. It is the test oracle for the Skia port, and a
//! functional stand-in for it while the port is in progress. It writes the same mask layout
//! and sample positions as the port.

use super::sink::StripSink;
use super::tile::Line;
use super::{MSAA16_PATTERN, SAMPLES, TILE_SIZE};

const TS: usize = TILE_SIZE as usize;

/// Rasterize one path into `sink`, clipped to the `width` × `height` viewport.
pub fn rasterize_path_reference(
    lines: &[Line],
    even_odd: bool,
    width: u32,
    height: u32,
    sink: &mut StripSink,
) {
    if lines.is_empty() || width == 0 || height == 0 {
        return;
    }
    let width_in_tiles = width.div_ceil(TILE_SIZE as u32) as usize;
    let height_in_tiles = height.div_ceil(TILE_SIZE as u32) as usize;

    // Bucket lines by the tile rows they span (half-open in y, like the sample test below).
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); height_in_tiles];
    for (i, l) in lines.iter().enumerate() {
        let (ymin, ymax) = (l.p0.y.min(l.p1.y), l.p0.y.max(l.p1.y));
        if ymin == ymax || ymax <= 0.0 || ymin >= height as f32 {
            continue;
        }
        let r0 = ((ymin / TS as f32).floor().max(0.0) as usize).min(height_in_tiles);
        let r1 = ((ymax / TS as f32).ceil().max(0.0) as usize).min(height_in_tiles);
        for b in &mut buckets[r0..r1] {
            b.push(i as u32);
        }
    }

    let mut crossings: Vec<(f32, i32)> = Vec::new();
    let mut row_masks: Vec<u16> = Vec::new();
    for (ty, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        // Pixels outside the x extent of the row's lines have winding 0.
        let mut xmin = f32::INFINITY;
        let mut xmax = f32::NEG_INFINITY;
        for &i in bucket {
            let l = &lines[i as usize];
            xmin = xmin.min(l.p0.x.min(l.p1.x));
            xmax = xmax.max(l.p0.x.max(l.p1.x));
        }
        let tx0 = ((xmin / TS as f32).floor().max(0.0) as usize).min(width_in_tiles);
        let tx1 = ((xmax / TS as f32).ceil().max(0.0) as usize).min(width_in_tiles);
        if tx0 >= tx1 {
            continue;
        }
        let px0 = (tx0 * TS) as i64;
        let row_px = (tx1 - tx0) * TS;
        row_masks.clear();
        row_masks.resize(row_px * TS, 0);

        for py in 0..TS {
            for k in 0..SAMPLES {
                let sy = (ty * TS + py) as f32 + (k as f32 + 0.5) / SAMPLES as f32;
                let off = (MSAA16_PATTERN[k] as f32 + 0.5) / SAMPLES as f32;
                crossings.clear();
                for &i in bucket {
                    let l = &lines[i as usize];
                    let (y0, y1) = (l.p0.y, l.p1.y);
                    // Half-open in y so shared vertices are counted once.
                    let hit = if y0 < y1 {
                        y0 <= sy && sy < y1
                    } else {
                        y1 <= sy && sy < y0
                    };
                    if !hit {
                        continue;
                    }
                    let t = (sy - y0) / (y1 - y0);
                    let x = l.p0.x + t * (l.p1.x - l.p0.x);
                    crossings.push((x, if y1 > y0 { 1 } else { -1 }));
                }
                if crossings.is_empty() {
                    continue;
                }
                crossings.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                // The sample at pixel px is at px + off. Its winding is the sum of the
                // crossings strictly to its left. Within (x_j, x_{j+1}] the winding is
                // constant, so fill whole pixel ranges at once.
                let bit = 1_u16 << k;
                let row = &mut row_masks[py * row_px..(py + 1) * row_px];
                let mut winding = 0;
                for j in 0..crossings.len() {
                    winding += crossings[j].1;
                    let inside = if even_odd {
                        winding & 1 != 0
                    } else {
                        winding != 0
                    };
                    if !inside {
                        continue;
                    }
                    let lo = crossings[j].0;
                    let hi = crossings.get(j + 1).map_or(f32::INFINITY, |c| c.0);
                    // px + off > lo  <=>  px >= floor(lo - off) + 1
                    // px + off <= hi <=>  px <= floor(hi - off)
                    let first = ((lo - off).floor() as i64 + 1).max(px0);
                    let last = if hi.is_finite() {
                        ((hi - off).floor() as i64).min(px0 + row_px as i64 - 1)
                    } else {
                        px0 + row_px as i64 - 1
                    };
                    for px in first..=last {
                        row[(px - px0) as usize] |= bit;
                    }
                }
            }
        }

        for tx in tx0..tx1 {
            let col0 = (tx - tx0) * TS;
            let scratch = sink.scratch_mut();
            for y in 0..TS {
                let row = &row_masks[y * row_px + col0..y * row_px + col0 + TS];
                for x in (0..TS).step_by(2) {
                    scratch[y * (TS / 2) + x / 2] = row[x] as u32 | ((row[x + 1] as u32) << 16);
                }
            }
            sink.push_mask_tile(tx as u32, ty as u32);
        }
    }
}
