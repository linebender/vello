// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Bitflags for each stage that can fail allocation.
let STAGE_BINNING: u32 = 0x1u;
let STAGE_TILE_ALLOC: u32 = 0x2u;
let STAGE_PATH_COARSE: u32 = 0x4u;
let STAGE_COARSE: u32 = 0x8u;

// This must be kept in sync with the struct in src/render.rs
struct BumpAllocators {
    // Bitmask of stages that have failed allocation.
    failed: atomic<u32>,
    binning: atomic<u32>,
    ptcl: atomic<u32>,
    tile: atomic<u32>,
    segments: atomic<u32>,
    blend: atomic<u32>,
}
