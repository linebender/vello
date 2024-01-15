// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use std::cmp::Ordering;

use crate::{
    cpu_dispatch::CpuBinding,
    cpu_shader::util::{ONE_MINUS_ULP, ROBUST_EPSILON},
};

use vello_encoding::{BumpAllocators, LineSoup};

use super::util::{span, Vec2};

const TILE_WIDTH: u32 = 1;
const TILE_HEIGHT: u32 = 4;
const TILE_SCALE_X: f32 = 1.0 / (TILE_WIDTH as f32);
const TILE_SCALE_Y: f32 = 1.0 / (TILE_HEIGHT as f32);

// This represents a small tile, analogous to a boundary fragment in Li
// et al. Ultimately we should have a packed representation as in the
// Scanline meets Vello doc.
#[derive(Clone, Debug)]
struct MiniTile {
    path_id: u32,
    // These coordinates are in tile units
    x: u32,
    y: u32,
    delta_wind: i32,
}

fn cmp_minitile(a: &MiniTile, b: &MiniTile) -> Ordering {
    (a.path_id, a.y, a.x).cmp(&(b.path_id, b.y, b.x))
}

fn merge_tiles(tiles: &[MiniTile]) -> Vec<MiniTile> {
    let mut result: Vec<MiniTile> = vec![];
    for tile in tiles {
        if let Some(last) = result.last_mut() {
            if last.path_id == tile.path_id && last.y == tile.y {
                if last.x == tile.x {
                    last.delta_wind += tile.delta_wind;
                } else {
                    let mut new_tile = tile.clone();
                    new_tile.delta_wind += last.delta_wind;
                    result.push(new_tile);
                }
            } else {
                result.push(tile.clone());
            }
        } else {
            result.push(tile.clone());
        }
    }
    result
}

fn make_strips(merged: &[MiniTile]) {
    let mut i = 0;
    let mut n_strips = 0;
    let mut n_li_prims = merged.len();
    while i < merged.len() {
        let first = &merged[i];
        let mut j = i + 1;
        let mut expected_x = first.x + 1;
        let mut strip_end = 0;
        let mut last_wind = first.delta_wind;
        while j < merged.len() {
            let this = &merged[j];
            if first.path_id != this.path_id || first.y != this.y || this.x != expected_x {
                strip_end = if last_wind != 0 { this.x } else { expected_x };
                break;
            }
            j += 1;
            expected_x += 1;
            last_wind = this.delta_wind;
        }
        if strip_end != expected_x {
            n_li_prims += 1;
        }
        println!(
            "strip path_id = {}, y = {}, {}..{} {}",
            first.path_id, first.y, first.x, expected_x, strip_end
        );
        n_strips += 1;
        i = j;
    }
    println!("n_strips = {n_strips}, n_li_prims = {n_li_prims}");
}

fn scanline_main(bump: &BumpAllocators, lines: &[LineSoup]) {
    println!("running scanline, n_lines = {}", bump.lines);
    let mut tiles = vec![];
    for line in &lines[..bump.lines as usize] {
        // coarse rasterization logic
        let p0 = Vec2::from_array(line.p0);
        let p1 = Vec2::from_array(line.p1);
        let is_down = p1.y >= p0.y;
        let (xy0, xy1) = if is_down { (p0, p1) } else { (p1, p0) };
        let s0 = Vec2::new(xy0.x * TILE_SCALE_X, xy0.y * TILE_SCALE_Y);
        let s1 = Vec2::new(xy1.x * TILE_SCALE_X, xy1.y * TILE_SCALE_Y);
        let count_x = span(s0.x, s1.x) - 1;
        let count = count_x + span(s0.y, s1.y);
        println!("line {line:?}, count {count}");

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

        // This is a hack and should be more principled
        let bbox = [0, 0, 2048, 2110];
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
        for _y in ymin..ymax {
            // TODO: handle delta bumps left of viewport
        }
        let mut last_z = (a * (imin as f32 - 1.0) + b).floor();
        for i in imin..imax {
            let zf = a * i as f32 + b;
            let z = zf.floor();
            let y = (y0 + i as f32 - z) as i32;
            let x = (x0 + sign * z) as i32;
            let top_edge = if i == 0 { y0 == s0.y } else { last_z == z };
            let delta_wind = if top_edge && x + 1 < bbox[2] {
                delta
            } else {
                0
            };
            let tile = MiniTile {
                path_id: line.path_ix,
                x: x as u32,
                y: y as u32,
                delta_wind,
            };
            tiles.push(tile);
            // let seg_within_slice = tile[(base + x) as usize].segment_count_or_ix;
            // tile[(base + x) as usize].segment_count_or_ix += 1;
            // let counts = (seg_within_slice << 16) | i;
            // let seg_count = SegmentCount { line_ix, counts };
            // seg_counts[(seg_base + i - imin) as usize] = seg_count;
            last_z = z;
        }
    }
    tiles.sort_by(cmp_minitile);
    //println!("{tiles:#?}");
    let merged = merge_tiles(&tiles);
    println!("{merged:#?}");
    make_strips(&merged);
    println!("n tiles {}, n merged {}", tiles.len(), merged.len());
}

pub fn scanline(_n_wg: u32, resources: &[CpuBinding]) {
    let bump = resources[0].as_typed();
    let lines = resources[1].as_slice();
    scanline_main(&bump, &lines);
}
