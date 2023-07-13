// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU implementations of shader stages.

mod backdrop;
mod bbox_clear;
mod binning;
mod coarse;
mod draw_leaf;
mod draw_reduce;
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

pub use pathtag_reduce::pathtag_reduce;

// Common definitions

const PTCL_INITIAL_ALLOC: u32 = 64;

// Tags for PTCL commands
const CMD_END: u32 = 0;
const CMD_FILL: u32 = 1;
const CMD_STROKE: u32 = 2;
const CMD_SOLID: u32 = 3;
const CMD_COLOR: u32 = 5;
const CMD_LIN_GRAD: u32 = 6;
const CMD_RAD_GRAD: u32 = 7;
const CMD_IMAGE: u32 = 8;
const CMD_BEGIN_CLIP: u32 = 9;
const CMD_END_CLIP: u32 = 10;
const CMD_JUMP: u32 = 11;
