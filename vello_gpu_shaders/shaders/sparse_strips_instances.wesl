// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The instance pass of GPU sparse strips: one thread per strip descriptor emits the strip
//! instances `vello_common::strip::visit_strip_fill_segments` would emit for it (an alpha
//! segment, and a fill segment for the gap to the next strip), clipped to the draw target's
//! bounds and shifted into the target, for paths the scheduler marked as drawn directly from
//! GPU buffers. Instances land in the path's slot range, in no particular order (strips of one
//! path never overlap); unused slots stay zero (the buffer is cleared) and draw nothing.

import package::helpers::sparse_strips::{
    StripDesc, DrawInfo, Instance, config, paths, DRAW_DIRECT, TILE_W, TILE_H, TILE_PIXELS,
    STRIP_FILL_GAP, STRIP_SENTINEL,
};

const WG: u32 = 64u;

@group(1) @binding(0) var<storage, read> strip_descs: array<StripDesc>;
@group(1) @binding(1) var<storage, read> draw_info: array<DrawInfo>;
@group(1) @binding(2) var<storage, read_write> instances: array<Instance>;
// Per-path instance counters, zero at dispatch.
@group(1) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

fn shift_coord(v: u32, shift: i32) -> u32 {
    return u32(clamp(i32(v) + shift, 0, 0xffff));
}

// Emit the pixel rectangle `[x0, x1) x [y, y + TILE_H)` of the target, shifted like
// `RectU16::shift` (each coordinate clamped to u16).
fn emit(info: DrawInfo, path: u32, x0: u32, x1: u32, y: u32, col_idx: u32, dense: bool) {
    let sx0 = shift_coord(x0, info.shift_x);
    let sx1 = shift_coord(x1, info.shift_x);
    let sy = shift_coord(y, info.shift_y);
    let width = sx1 - sx0;
    let slot = info.instance_base + atomicAdd(&counters[path], 1u);
    let dense_width = select(0u, width, dense);
    instances[slot] = Instance(
        sx0 | (sy << 16u),
        width | (dense_width << 16u),
        select(0u, col_idx, dense),
        info.payload,
        info.paint,
        info.depth_index,
    );
}

@compute @workgroup_size(WG)
fn build_instances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    let n_strips = paths[config.n_paths - 1u].strip_end;
    if s >= n_strips {
        return;
    }
    let d = strip_descs[s];
    if (d.info & 3u) == STRIP_SENTINEL {
        return;
    }
    let info = draw_info[d.path];
    if (info.flags & DRAW_DIRECT) == 0u {
        return;
    }
    let x = d.xy & 0xffffu;
    let y = d.xy >> 16u;
    let bx0 = info.bounds_x & 0xffffu;
    let bx1 = info.bounds_x >> 16u;
    let by0 = info.bounds_y & 0xffffu;
    let by1 = info.bounds_y >> 16u;
    let tile_y = y / TILE_H;
    if tile_y < by0 || tile_y >= by1 {
        return;
    }
    // Every path ends with a sentinel, so the next descriptor belongs to the same path.
    let next = strip_descs[s + 1u];
    let tile_x0 = x / TILE_W;
    // One alpha block per tile column.
    let tile_x1 = tile_x0 + (next.block_start - d.block_start);
    let cx0 = max(tile_x0, bx0);
    let cx1 = min(tile_x1, bx1);
    if cx0 < cx1 {
        let alpha_idx = (d.block_start + config.alpha_base_blocks + (cx0 - tile_x0)) * TILE_PIXELS;
        emit(info, d.path, cx0 * TILE_W, cx1 * TILE_W, y, alpha_idx / TILE_H, true);
    }
    if (next.info & STRIP_FILL_GAP) != 0u && (next.xy >> 16u) == y {
        let fx0 = max(tile_x1, bx0);
        let fx1 = min((next.xy & 0xffffu) / TILE_W, bx1);
        if fx0 < fx1 {
            emit(info, d.path, fx0 * TILE_W, fx1 * TILE_W, y, 0u, false);
        }
    }
}
