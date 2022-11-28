// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    target_width: u32,
    target_height: u32,

    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,

    // offsets within scene buffer (in u32 units)
    // Note: this is a difference from piet-gpu, which is in bytes
    pathtag_base: u32,
    pathdata_base: u32,

    drawtag_base: u32,
    drawdata_base: u32,

    transform_base: u32,
    linewidth_base: u32,
}

// Geometry of tiles and bins

let TILE_WIDTH = 16u;
let TILE_HEIGHT = 16u;
// Number of tiles per bin
let N_TILE_X = 16u;
let N_TILE_Y = 16u;
//let N_TILE = N_TILE_X * N_TILE_Y;
let N_TILE = 256u;
