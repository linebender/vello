// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hybrid fast-path strip topology and GPU winding payload generation.

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::flatten::Line;
use vello_common::peniko::Fill;
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};

const TILE_Y_SHIFT: u32 = 16;
const KIND_BIT: u32 = 1 << 31;

/// A tile-local winding input for the GPU winding render pass.
///
/// Kind 0 instances represent analytic line segments.
/// Kind 1 instances represent coarse per-row winding carry values.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuTileLine {
    /// Starting column in the winding texture.
    pub winding_col: u32,
    /// Packed tile x (bits 0..15), tile y (bits 16..30), kind (bit 31).
    pub tile_xy_kind: u32,
    /// Analytic line start or coarse rows 0-1.
    pub p0: [f32; 2],
    /// Analytic line end or coarse rows 2-3.
    pub p1: [f32; 2],
}

#[derive(Debug, Default)]
pub(crate) struct WindingOutput {
    pub strips: Vec<Strip>,
    pub tile_lines: Vec<GpuTileLine>,
    /// Number of logical winding values. This is also the next `alpha_idx` offset.
    pub winding_value_count: u32,
}

#[inline(always)]
fn pack_tile_xy_kind(tile_x: u16, tile_y: u16, kind: u32) -> u32 {
    (tile_x as u32) | ((tile_y as u32) << TILE_Y_SHIFT) | (kind * KIND_BIT)
}

/// Convert the logical winding-value count into physical texture dimensions.
pub(crate) fn winding_texture_height(texture_width: u32, winding_value_count: u32) -> u32 {
    let num_columns = winding_value_count.div_ceil(u32::from(Tile::HEIGHT));
    let num_bands = num_columns.div_ceil(texture_width).max(1);
    num_bands * u32::from(Tile::HEIGHT)
}

/// Generate strip topology plus winding pass inputs from sorted tiles.
///
/// This preserves the strip boundary semantics of `vello_common::strip::render`, but
/// replaces CPU alpha generation with GPU winding inputs.
pub(crate) fn build_winding_output(
    tiles: &Tiles,
    fill_rule: Fill,
    lines: &[Line],
    alpha_idx_offset: u32,
) -> WindingOutput {
    let mut output = WindingOutput::default();

    if tiles.is_empty() {
        return output;
    }

    let should_fill = |winding: i32| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    let mut winding_delta: i32 = 0;
    let mut accumulated_winding = [0.0f32; Tile::HEIGHT as usize];
    let mut prev_tile = *tiles.get(0);
    let mut next_alpha_idx = alpha_idx_offset;
    let mut location_winding_col = alpha_idx_offset / u32::from(Tile::HEIGHT);

    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, 0);

    let mut current_strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        next_alpha_idx,
        false,
    );

    for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
        let line = lines[tile.line_idx() as usize];

        if !prev_tile.same_loc(&tile) {
            next_alpha_idx += u32::from(Tile::WIDTH) * u32::from(Tile::HEIGHT);
        }

        if !prev_tile.same_loc(&tile) && !prev_tile.prev_loc(&tile) {
            output.strips.push(current_strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                if winding_delta != 0 || is_sentinel {
                    output.strips.push(Strip::new(
                        u16::MAX,
                        prev_tile.y * Tile::HEIGHT,
                        next_alpha_idx,
                        should_fill(winding_delta),
                    ));
                }

                winding_delta = 0;
                accumulated_winding = [0.0; Tile::HEIGHT as usize];
            }

            if is_sentinel {
                break;
            }

            current_strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                next_alpha_idx,
                should_fill(winding_delta),
            );
            accumulated_winding = [winding_delta as f32; Tile::HEIGHT as usize];
        }

        if tile_idx == 0 || !prev_tile.same_loc(&tile) {
            location_winding_col = next_alpha_idx / u32::from(Tile::HEIGHT);

            if accumulated_winding.iter().any(|value| *value != 0.0) {
                output.tile_lines.push(GpuTileLine {
                    winding_col: location_winding_col,
                    tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 1),
                    p0: [accumulated_winding[0], accumulated_winding[1]],
                    p1: [accumulated_winding[2], accumulated_winding[3]],
                });
            }
        }

        prev_tile = tile;

        let tile_left_x = f32::from(tile.x) * f32::from(Tile::WIDTH);
        let tile_top_y = f32::from(tile.y) * f32::from(Tile::HEIGHT);
        let p0_x = line.p0.x - tile_left_x;
        let p0_y = line.p0.y - tile_top_y;
        let p1_x = line.p1.x - tile_left_x;
        let p1_y = line.p1.y - tile_top_y;

        if p0_y == p1_y {
            continue;
        }

        let sign = (p0_y - p1_y).signum();
        winding_delta += sign as i32 * i32::from(tile.winding());

        let (line_left_x, line_left_y, line_right_x) = if p0_x < p1_x {
            (p0_x, p0_y, p1_x)
        } else {
            (p1_x, p1_y, p0_x)
        };

        // Account for winding that enters from the left of the viewport.
        if tile.x == 0 && line_left_x < 0.0 {
            let (line_top_y, line_top_x, line_bottom_y) = if p0_y < p1_y {
                (p0_y, p0_x, p1_y)
            } else {
                (p1_y, p1_x, p0_y)
            };
            let x_delta = if p0_y < p1_y { p1_x - p0_x } else { p0_x - p1_x };
            let y_slope = (line_bottom_y - line_top_y) / x_delta;

            let (viewport_ymin, viewport_ymax) = if line.p0.x == line.p1.x {
                (line_top_y, line_bottom_y)
            } else {
                let line_viewport_left_y = (line_top_y - line_top_x * y_slope)
                    .max(line_top_y)
                    .min(line_bottom_y);
                (
                    f32::min(line_left_y, line_viewport_left_y),
                    f32::max(line_left_y, line_viewport_left_y),
                )
            };

            let mut left_winding = [0.0f32; Tile::HEIGHT as usize];
            for row in 0..Tile::HEIGHT as usize {
                let row_top = row as f32;
                let row_bottom = row_top + 1.0;
                let h = (viewport_ymax.min(row_bottom) - viewport_ymin.max(row_top)).max(0.0);
                left_winding[row] = h * sign;
                accumulated_winding[row] += h * sign;
            }

            if left_winding.iter().any(|value| *value != 0.0) {
                output.tile_lines.push(GpuTileLine {
                    winding_col: location_winding_col,
                    tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 1),
                    p0: [left_winding[0], left_winding[1]],
                    p1: [left_winding[2], left_winding[3]],
                });
            }

            if line_right_x < 0.0 {
                continue;
            }
        }

        let global_top_y = line.p0.y.min(line.p1.y);
        let global_bottom_y = line.p0.y.max(line.p1.y);

        let dx = line.p1.x - line.p0.x;
        let dy = line.p1.y - line.p0.y;
        let tile_right_x = tile_left_x + f32::from(Tile::WIDTH);
        let (segment_top_y, segment_bottom_y) = if dx.abs() < 1e-6 {
            (global_top_y, global_bottom_y)
        } else {
            let slope = dy / dx;
            let y_at_left = line.p0.y + (tile_left_x - line.p0.x) * slope;
            let y_at_right = line.p0.y + (tile_right_x - line.p0.x) * slope;
            (
                y_at_left.min(y_at_right).max(global_top_y),
                y_at_left.max(y_at_right).min(global_bottom_y),
            )
        };

        for row in 0..Tile::HEIGHT as usize {
            let pixel_top = tile_top_y + row as f32;
            let pixel_bottom = pixel_top + 1.0;
            let ymin = segment_top_y.max(pixel_top);
            let ymax = segment_bottom_y.min(pixel_bottom);
            let h = (ymax - ymin).max(0.0);
            accumulated_winding[row] += h * sign;
        }

        output.tile_lines.push(GpuTileLine {
            winding_col: location_winding_col,
            tile_xy_kind: pack_tile_xy_kind(tile.x, tile.y, 0),
            p0: [line.p0.x, line.p0.y],
            p1: [line.p1.x, line.p1.y],
        });
    }

    output.winding_value_count = next_alpha_idx;
    output
}

#[cfg(test)]
mod tests {
    use super::{build_winding_output, winding_texture_height};
    use alloc::vec::Vec;
    use vello_common::fearless_simd::Level;
    use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
    use vello_common::peniko::Fill;
    use vello_common::strip;
    use vello_common::strip_generator::StripGenerator;
    use vello_common::tile::Tile;

    fn make_generator() -> StripGenerator {
        StripGenerator::new(128, 128, Level::baseline())
    }

    fn compare_fill(path: &BezPath) {
        let mut generator = make_generator();
        generator.prepare_tiles_for_fill(path, Affine::IDENTITY);
        let output = build_winding_output(
            generator.tiles(),
            Fill::NonZero,
            generator.lines(),
            0,
        );

        let mut ref_strips = Vec::new();
        let mut ref_alphas = Vec::new();
        strip::render(
            Level::baseline(),
            generator.tiles(),
            &mut ref_strips,
            &mut ref_alphas,
            Fill::NonZero,
            None,
            generator.lines(),
        );

        assert_eq!(output.strips, ref_strips);
        assert_eq!(output.winding_value_count as usize, ref_alphas.len());
    }

    fn compare_stroke(path: &BezPath, stroke: &Stroke) {
        let mut generator = make_generator();
        generator.prepare_tiles_for_stroke(path, stroke, Affine::IDENTITY);
        let output = build_winding_output(
            generator.tiles(),
            Fill::NonZero,
            generator.lines(),
            0,
        );

        let mut ref_strips = Vec::new();
        let mut ref_alphas = Vec::new();
        strip::render(
            Level::baseline(),
            generator.tiles(),
            &mut ref_strips,
            &mut ref_alphas,
            Fill::NonZero,
            None,
            generator.lines(),
        );

        assert_eq!(output.strips, ref_strips);
        assert_eq!(output.winding_value_count as usize, ref_alphas.len());
    }

    #[test]
    fn strip_topology_matches_fill_render() {
        let mut path = BezPath::new();
        path.move_to((-10.0, 10.0));
        path.line_to((70.0, 20.0));
        path.line_to((30.0, 80.0));
        path.close_path();
        compare_fill(&path);
    }

    #[test]
    fn strip_topology_matches_stroke_render() {
        let rect = Rect::new(10.0, 12.0, 92.0, 76.0).to_path(0.1);
        compare_stroke(&rect, &Stroke::new(3.5));
    }

    #[test]
    fn winding_texture_height_wraps_by_strip_height() {
        let width = 8;
        let values = 9 * u32::from(Tile::HEIGHT);
        assert_eq!(winding_texture_height(width, values), 8);
    }
}
