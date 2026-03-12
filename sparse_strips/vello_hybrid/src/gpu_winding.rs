// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-accelerated winding number computation.
//!
//! Generates [`GpuTileLine`] instances (analytic + coarse) and [`Strip`]s from sorted tiles,
//! replacing the CPU-side per-pixel alpha computation in [`vello_common::strip::render`].
//! A GPU render pass processes the tile-line instances into a winding texture, which the
//! strip shader then samples.

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::flatten::Line;
use vello_common::peniko::Fill;
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};

const TILE_Y_SHIFT: u32 = 16;
const KIND_BIT: u32 = 1 << 31;

/// A GPU tile-line instance for the winding computation shader.
///
/// Two kinds (distinguished by `KIND_BIT` in `tile_xy_kind`):
///   kind 0: Analytic — `p0`/`p1` are line endpoints in screen coordinates.
///   kind 1: Coarse   — `p0`/`p1` encode per-row winding values to splat.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuTileLine {
    /// Starting column index in the winding texture for this tile location.
    pub winding_col: u32,
    /// Packed tile x (bits 0–15), tile y (bits 16–30), kind (bit 31).
    /// x and y are in tile units (multiply by `Tile::WIDTH`/`Tile::HEIGHT` for pixels).
    pub tile_xy_kind: u32,
    /// Line start (analytic) or per-row winding for rows 0–1 (coarse).
    pub p0: [f32; 2],
    /// Line end (analytic) or per-row winding for rows 2–3 (coarse).
    pub p1: [f32; 2],
}

/// Pack tile coordinates and kind flag into a single `u32`.
fn pack_tile_xy_kind(tx: u16, ty: u16, kind: u32) -> u32 {
    (tx as u32) | ((ty as u32) << TILE_Y_SHIFT) | (kind * KIND_BIT)
}

/// Output of [`render_strips_and_tile_lines`].
#[derive(Debug)]
pub struct GpuWindingOutput {
    /// The generated strips.
    pub strips: Vec<Strip>,
    /// GPU tile-line instances for the winding render pass.
    pub tile_lines: Vec<GpuTileLine>,
    /// Total number of winding values (= number of pixels in the winding texture).
    /// Equal to `num_columns * Tile::HEIGHT`.
    pub winding_value_count: u32,
}

/// Generate strips and GPU tile-line instances from sorted tiles.
///
/// This replaces the CPU alpha computation in `strip::render()`. The strip layout
/// (boundaries, `fill_gap`, `alpha_idx`) is identical; instead of computing per-pixel
/// alpha, we emit [`GpuTileLine`] instances for a subsequent GPU winding pass.
/// `alpha_idx_offset` aligns winding_col values with the cumulative alpha_idx
/// in StripStorage. Pass `strip_storage.alphas.len() as u32` before the
/// corresponding `strip::render()` call.
pub fn render_strips_and_tile_lines(
    tiles: &Tiles,
    fill_rule: Fill,
    lines: &[Line],
    alpha_idx_offset: u32,
) -> GpuWindingOutput {
    let mut strips = Vec::new();
    let mut tile_lines = Vec::new();

    if tiles.is_empty() {
        return GpuWindingOutput {
            strips,
            tile_lines,
            winding_value_count: 0,
        };
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    let mut winding_delta: i32 = 0;
    let mut accumulated_winding = [0.0f32; Tile::HEIGHT as usize];
    let mut prev_tile = *tiles.get(0);
    let mut alpha_idx_counter: u32 = alpha_idx_offset;
    let mut location_winding_col: u32 = 0;

    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, 0);

    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        alpha_idx_counter,
        false,
    );

    for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
        let line = lines[tile.line_idx() as usize];

        // --- Location change: advance alpha counter ---
        if !prev_tile.same_loc(&tile) {
            alpha_idx_counter += u32::from(Tile::WIDTH) * u32::from(Tile::HEIGHT);
        }

        // --- Strip boundary ---
        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            strips.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                if winding_delta != 0 || is_sentinel {
                    strips.push(Strip::new(
                        u16::MAX,
                        prev_tile.y * Tile::HEIGHT,
                        alpha_idx_counter,
                        should_fill(winding_delta),
                    ));
                }

                winding_delta = 0;
                accumulated_winding = [0.0; Tile::HEIGHT as usize];
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                alpha_idx_counter,
                should_fill(winding_delta),
            );
            accumulated_winding = [winding_delta as f32; Tile::HEIGHT as usize];
        }

        // --- New tile position: emit coarse instance ---
        if !prev_tile.same_loc(&tile) || tile_idx == 0 {
            location_winding_col = alpha_idx_counter / u32::from(Tile::HEIGHT);

            if accumulated_winding.iter().any(|&v| v != 0.0) {
                tile_lines.push(GpuTileLine {
                    winding_col: location_winding_col,
                    tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 1),
                    p0: [accumulated_winding[0], accumulated_winding[1]],
                    p1: [accumulated_winding[2], accumulated_winding[3]],
                });
            }
        }

        prev_tile = tile;

        let p0_y = line.p0.y - f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let p1_y = line.p1.y - f32::from(tile.y) * f32::from(Tile::HEIGHT);

        // Skip horizontal lines.
        if p0_y == p1_y {
            continue;
        }

        let sign = (p0_y - p1_y).signum();
        winding_delta += sign as i32 * i32::from(tile.winding());

        let tile_top_y = f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let tile_left_x = f32::from(tile.x) * f32::from(Tile::WIDTH);
        let tile_right_x = tile_left_x + f32::from(Tile::WIDTH);

        let p0_x = line.p0.x - tile_left_x;
        let p1_x = line.p1.x - tile_left_x;
        let (line_left_x, line_left_y, line_right_x) = if p0_x < p1_x {
            (p0_x, line.p0.y - tile_top_y, p1_x)
        } else {
            (p1_x, line.p1.y - tile_top_y, p0_x)
        };

        // --- Viewport left edge: account for winding from left-of-viewport portion ---
        if tile.x == 0 && line_left_x < 0. {
            let (line_top_y_local, line_top_x, line_bottom_y_local) = if p0_y < p1_y {
                (p0_y, p0_x, p1_y)
            } else {
                (p1_y, p1_x, p0_y)
            };
            let y_slope = (line_bottom_y_local - line_top_y_local)
                / (if p0_y < p1_y { p1_x - p0_x } else { p0_x - p1_x });

            let (vp_ymin, vp_ymax) = if line.p0.x == line.p1.x {
                (line_top_y_local, line_bottom_y_local)
            } else {
                let line_viewport_left_y = (line_top_y_local - line_top_x * y_slope)
                    .max(line_top_y_local)
                    .min(line_bottom_y_local);
                (
                    f32::min(line_left_y, line_viewport_left_y),
                    f32::max(line_left_y, line_viewport_left_y),
                )
            };

            // Per-row contribution from left-of-viewport portion.
            let mut left_winding = [0.0f32; Tile::HEIGHT as usize];
            for row in 0..Tile::HEIGHT as usize {
                let row_top = row as f32;
                let row_bottom = row_top + 1.0;
                let h = (vp_ymax.min(row_bottom) - vp_ymin.max(row_top)).max(0.0);
                left_winding[row] = h * sign;
                accumulated_winding[row] += h * sign;
            }

            // Emit coarse instance for the left-of-viewport contribution.
            if left_winding.iter().any(|&v| v != 0.0) {
                tile_lines.push(GpuTileLine {
                    winding_col: location_winding_col,
                    tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 1),
                    p0: [left_winding[0], left_winding[1]],
                    p1: [left_winding[2], left_winding[3]],
                });
            }

            if line_right_x < 0. {
                // Entire line is left of viewport — no analytic instance needed.
                continue;
            }
        }

        // --- Cheap per-row height clamping to track accumulated winding ---
        let global_top_y = line.p0.y.min(line.p1.y);
        let global_bot_y = line.p0.y.max(line.p1.y);

        let dx = line.p1.x - line.p0.x;
        let dy = line.p1.y - line.p0.y;
        let (line_top_y, line_bottom_y) = if dx.abs() < 1e-6 {
            (global_top_y, global_bot_y)
        } else {
            let slope = dy / dx;
            let y_at_left = line.p0.y + (tile_left_x - line.p0.x) * slope;
            let y_at_right = line.p0.y + (tile_right_x - line.p0.x) * slope;
            let ymin = y_at_left.min(y_at_right).max(global_top_y);
            let ymax = y_at_left.max(y_at_right).min(global_bot_y);
            (ymin, ymax)
        };

        for row in 0..Tile::HEIGHT as usize {
            let px_top = tile_top_y + row as f32;
            let px_bottom = px_top + 1.0;
            let ymin = line_top_y.max(px_top);
            let ymax = line_bottom_y.min(px_bottom);
            let h = (ymax - ymin).max(0.0);
            accumulated_winding[row] += h * sign;
        }

        // --- Emit analytic instance ---
        tile_lines.push(GpuTileLine {
            winding_col: location_winding_col,
            tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 0),
            p0: [line.p0.x, line.p0.y],
            p1: [line.p1.x, line.p1.y],
        });
    }

    GpuWindingOutput {
        strips,
        tile_lines,
        winding_value_count: alpha_idx_counter,
    }
}
