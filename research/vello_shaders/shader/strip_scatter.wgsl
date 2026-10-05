// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Scatter the CPU sparse strip records (`AaConfig::SparseMsaa16`) into the tile
// array allocated by `tile_alloc`.
//
// See `vello_shaders::cpu::sparse_strips` for the CPU -> GPU contract. Each record
// is either a mask tile, which sets `segment_count_or_ix = block + 1` (coarse then
// emits `CMD_MASK`), or a backdrop delta, which `backdrop_dyn` (run afterwards)
// prefix-sums into backdrops, so tiles inside solid spans get a backdrop of 1.
//
// The tile rectangle of a path comes from its clip-intersected bbox, so records
// outside it are dropped. A delta left of the rectangle is moved to its left edge,
// which keeps spans that start left of the rectangle and end inside it correct.

#import bump
#import tile

struct StripScatterConfig {
    n_records: u32,
    n_drawobj: u32,
    // Invocations per row of the dispatch. The dispatch is split over y when it
    // needs more than 65535 workgroups.
    row_stride: u32,
    _padding: u32,
}

// Must match `StripRecord` in `vello_shaders::cpu::sparse_strips::sink`.
struct StripRecord {
    path_ix: u32,
    // tile_x | (tile_y << 16)
    xy: u32,
    // Mask tile: the mask block index. Backdrop delta: KIND_DELTA | (delta as u16).
    payload: u32,
}

// Same layout as `Tile`, but with atomic fields.
struct AtomicTile {
    backdrop: atomic<i32>,
    segment_count_or_ix: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> scatter_config: StripScatterConfig;

@group(0) @binding(1)
var<storage> records: array<StripRecord>;

@group(0) @binding(2)
var<storage> paths: array<Path>;

@group(0) @binding(3)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(4)
var<storage, read_write> tiles: array<AtomicTile>;

const KIND_DELTA = 0x80000000u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Abort if any of the prior stages failed: the tile allocation can't be trusted.
    if atomicLoad(&bump.failed) != 0u {
        return;
    }
    let ix = global_id.y * scatter_config.row_stride + global_id.x;
    if ix >= scatter_config.n_records {
        return;
    }
    let record = records[ix];
    if record.path_ix >= scatter_config.n_drawobj {
        return;
    }
    let path = paths[record.path_ix];
    let x = record.xy & 0xffffu;
    let y = record.xy >> 16u;
    if y < path.bbox.y || y >= path.bbox.w {
        return;
    }
    let row = path.tiles + (y - path.bbox.y) * (path.bbox.z - path.bbox.x);
    if (record.payload & KIND_DELTA) == 0u {
        if x >= path.bbox.x && x < path.bbox.z {
            atomicStore(&tiles[row + x - path.bbox.x].segment_count_or_ix, record.payload + 1u);
        }
    } else {
        let cx = max(x, path.bbox.x);
        if cx < path.bbox.z {
            // Sign extend the low 16 bits.
            let delta = bitcast<i32>(record.payload << 16u) >> 16u;
            atomicAdd(&tiles[row + cx - path.bbox.x].backdrop, delta);
        }
    }
}
