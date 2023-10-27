// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This must be kept in sync with the struct in src/encoding/resolve.rs
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    target_width: u32,
    target_height: u32,

    // The initial color applied to the pixels in a tile during the fine stage.
    // This is only used in the full pipeline. The format is packed RGBA8 in MSB
    // order.
    base_color: u32,

    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,

    // To reduce the number of bindings, info and bin data are combined
    // into one buffer.
    bin_data_start: u32,

    // offsets within scene buffer (in u32 units)
    pathtag_base: u32,
    pathdata_base: u32,

    drawtag_base: u32,
    drawdata_base: u32,

    transform_base: u32,
    style_base: u32,

    // Sizes of bump allocated buffers (in element size units)
    binning_size: u32,
    tiles_size: u32,
    segments_size: u32,    
    ptcl_size: u32,
}

// Geometry of tiles and bins

let TILE_WIDTH = 16u;
let TILE_HEIGHT = 16u;
// Number of tiles per bin
let N_TILE_X = 16u;
let N_TILE_Y = 16u;
//let N_TILE = N_TILE_X * N_TILE_Y;
let N_TILE = 256u;

// Not currently supporting non-square tiles
let TILE_SCALE = 0.0625;

let BLEND_STACK_SPLIT = 4u;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
let RAD_GRAD_KIND_CIRCULAR = 1u;
let RAD_GRAD_KIND_STRIP = 2u;
let RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3u;
let RAD_GRAD_KIND_CONE = 4u;

// Radial gradient flags
let RAD_GRAD_SWAPPED = 1u;
