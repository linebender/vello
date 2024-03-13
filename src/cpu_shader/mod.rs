// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

//! CPU implementations of shader stages.

// Allow un-idiomatic Rust to more closely match shaders
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

mod backdrop;
mod bbox_clear;
mod binning;
mod clip_leaf;
mod clip_reduce;
mod coarse;
mod draw_leaf;
mod draw_reduce;
mod euler;
mod fine;
mod flatten;
mod path_count;
mod path_count_setup;
mod path_tiling;
mod path_tiling_setup;
mod pathtag_reduce;
mod pathtag_scan;
mod tile_alloc;
mod util;

pub use backdrop::backdrop;
pub use bbox_clear::bbox_clear;
pub use binning::binning;
pub use clip_leaf::clip_leaf;
pub use clip_reduce::clip_reduce;
pub use coarse::coarse;
pub use draw_leaf::draw_leaf;
pub use draw_reduce::draw_reduce;
pub use flatten::flatten;
pub use path_count::path_count;
pub use path_count_setup::path_count_setup;
pub use path_tiling::path_tiling;
pub use path_tiling_setup::path_tiling_setup;
pub use pathtag_reduce::pathtag_reduce;
pub use pathtag_scan::pathtag_scan;
pub use tile_alloc::tile_alloc;

// Common definitions

const PTCL_INITIAL_ALLOC: u32 = 64;

// Tags for PTCL commands
const CMD_END: u32 = 0;
const CMD_FILL: u32 = 1;
//const CMD_STROKE: u32 = 2;
const CMD_SOLID: u32 = 3;
const CMD_COLOR: u32 = 5;
const CMD_LIN_GRAD: u32 = 6;
const CMD_RAD_GRAD: u32 = 7;
const CMD_SWEEP_GRAD: u32 = 8;
const CMD_IMAGE: u32 = 9;
const CMD_BEGIN_CLIP: u32 = 10;
const CMD_END_CLIP: u32 = 11;
const CMD_JUMP: u32 = 12;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
const RAD_GRAD_KIND_CIRCULAR: u32 = 1;
const RAD_GRAD_KIND_STRIP: u32 = 2;
const RAD_GRAD_KIND_FOCAL_ON_CIRCLE: u32 = 3;
const RAD_GRAD_KIND_CONE: u32 = 4;

// Radial gradient flags
const RAD_GRAD_SWAPPED: u32 = 1;
