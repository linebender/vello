// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{ConfigUniform, PathSegment, Tile};

use super::{CMD_COLOR, CMD_END, CMD_FILL, CMD_JUMP, CMD_SOLID, CpuTexture, PTCL_INITIAL_ALLOC};

// These should also move into a common area
const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;
const TILE_SIZE: usize = TILE_WIDTH * TILE_HEIGHT;

fn read_color(ptcl: &[u32], offset: u32) -> u32 {
    ptcl[(offset + 1) as usize]
}

struct CmdFill {
    size_and_rule: u32,
    seg_data: u32,
    backdrop: i32,
}

fn read_fill(ptcl: &[u32], offset: u32) -> CmdFill {
    let size_and_rule = ptcl[(offset + 1) as usize];
    let seg_data = ptcl[(offset + 2) as usize];
    let backdrop = ptcl[(offset + 3) as usize] as i32;
    CmdFill {
        size_and_rule,
        seg_data,
        backdrop,
    }
}

fn unpack4x8unorm(x: u32) -> [f32; 4] {
    let mut result = [0.0; 4];
    for i in 0..4 {
        result[i] = ((x >> (i * 8)) & 0xff) as f32 * (1.0 / 255.0);
    }
    result
}

fn pack4x8unorm(x: [f32; 4]) -> u32 {
    let mut result = 0;
    for i in 0..4 {
        let byte = (x[i].clamp(0.0, 1.0) * 255.0).round() as u32;
        result |= byte << (i * 8);
    }
    result
}

fn fill_path(area: &mut [f32], segments: &[PathSegment], fill: &CmdFill, x_tile: f32, y_tile: f32) {
    let n_segs = fill.size_and_rule >> 1;
    let even_odd = (fill.size_and_rule & 1) != 0;
    let backdrop_f = fill.backdrop as f32;
    for a in area.iter_mut() {
        *a = backdrop_f;
    }
    for segment in &segments[fill.seg_data as usize..][..n_segs as usize] {
        let delta = [
            segment.point1[0] - segment.point0[0],
            segment.point1[1] - segment.point0[1],
        ];
        for yi in 0..TILE_HEIGHT {
            let y = segment.point0[1] - (y_tile + yi as f32);
            let y0 = y.clamp(0.0, 1.0);
            let y1 = (y + delta[1]).clamp(0.0, 1.0);
            let dy = y0 - y1;
            let y_edge =
                delta[0].signum() * (y_tile + yi as f32 - segment.y_edge + 1.0).clamp(0.0, 1.0);
            if dy != 0.0 {
                let vec_y_recip = delta[1].recip();
                let t0 = (y0 - y) * vec_y_recip;
                let t1 = (y1 - y) * vec_y_recip;
                let startx = segment.point0[0] - x_tile;
                let x0 = startx + t0 * delta[0];
                let x1 = startx + t1 * delta[0];
                let xmin0 = x0.min(x1);
                let xmax0 = x0.max(x1);
                for i in 0..TILE_WIDTH {
                    let i_f = i as f32;
                    let xmin = (xmin0 - i_f).min(1.0) - 1.0e-6;
                    let xmax = xmax0 - i_f;
                    let b = xmax.min(1.0);
                    let c = b.max(0.0);
                    let d = xmin.max(0.0);
                    let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                    area[yi * TILE_WIDTH + i] += y_edge + a * dy;
                }
            } else if y_edge != 0.0 {
                for i in 0..TILE_WIDTH {
                    area[yi * TILE_WIDTH + i] += y_edge;
                }
            }
        }
    }
    if even_odd {
        for a in area.iter_mut() {
            {
                *a = (*a - 2.0 * (0.5 * *a).round()).abs();
            }
        }
    } else {
        for a in area.iter_mut() {
            {
                *a = a.abs().min(1.0);
            }
        }
    }
}

#[expect(unused, reason = "Draft code as textures not wired up")]
fn fine_main(
    config: &ConfigUniform,
    tiles: &[Tile],
    segments: &[PathSegment],
    output: &mut CpuTexture,
    ptcl: &[u32],
    info: &[u32],
    // TODO: image texture resources
    // TODO: masks?
) {
    let width_in_tiles = config.width_in_tiles;
    let height_in_tiles = config.height_in_tiles;
    let n_tiles = width_in_tiles * height_in_tiles;
    let mut area = vec![0.0_f32; TILE_SIZE];
    let mut rgba = vec![[0.0_f32; 4]; TILE_SIZE];
    for tile_ix in 0..n_tiles {
        rgba.fill([0.0; 4]);
        area.fill(0.0);
        let tile_x = tile_ix % width_in_tiles;
        let tile_y = tile_ix / width_in_tiles;
        let mut cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
        // skip over blend stack allocation
        cmd_ix += 1;
        loop {
            let tag = ptcl[cmd_ix as usize];
            if tag == CMD_END {
                break;
            }
            match tag {
                CMD_FILL => {
                    let fill = read_fill(ptcl, cmd_ix);
                    // x0 and y0 will go away when we do tile-relative coords
                    let x0 = (tile_x as usize * TILE_WIDTH) as f32;
                    let y0 = (tile_y as usize * TILE_HEIGHT) as f32;
                    fill_path(&mut area, segments, &fill, x0, y0);
                    cmd_ix += 4;
                }
                CMD_SOLID => {
                    area.fill(1.0);
                    cmd_ix += 2;
                }
                CMD_COLOR => {
                    let color = read_color(ptcl, cmd_ix);
                    let fg = unpack4x8unorm(color);
                    let fg = [fg[3], fg[2], fg[1], fg[0]];
                    for i in 0..TILE_SIZE {
                        let ai = area[i];
                        let fg_i = [fg[0] * ai, fg[1] * ai, fg[2] * ai, fg[3] * ai];
                        for j in 0..4 {
                            rgba[i][j] = rgba[i][j] * (1.0 - fg_i[3]) + fg_i[j];
                        }
                    }
                    cmd_ix += 2;
                }
                CMD_JUMP => {
                    cmd_ix = ptcl[(cmd_ix + 1) as usize];
                }
                _ => todo!("unhandled ptcl command {tag}"),
            }
        }
        // Write tile (in rgba)
        for y in 0..TILE_HEIGHT {
            let base =
                output.width * (tile_y as usize * TILE_HEIGHT + y) + tile_x as usize * TILE_WIDTH;
            for x in 0..TILE_WIDTH {
                let rgba32 = pack4x8unorm(rgba[y * TILE_WIDTH + x]);
                output.pixels[base + x] = rgba32;
            }
        }
    }
}
