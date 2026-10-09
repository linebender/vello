// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The merge pass of GPU sparse strips: every entry rasterizes its line into per-pixel winding
//! contributions for its tile, and the contributions of a run (all entries at one tile
//! location) are summed. Runs are contiguous, so the sum is a segmented reduction: within a
//! workgroup through shared-memory atomics into the run head's slot, across workgroups through
//! global atomics into the run's accumulation block (which starts zeroed). The resolve pass
//! turns the sums into alphas.
//!
//! See `vello_gpu::sparse_strips::model::line_payload` for the per-line math.

import package::helpers::sparse_strips::{
    config, paths, entries, lines, path_of_entry, entry_xy, entry_line, TILE_W, TILE_H,
    TILE_PIXELS, ACC_WORDS, FIXED_ONE,
};

const WG: u32 = 128u;

@group(1) @binding(0) var<storage, read> entry_block: array<u32>;
@group(1) @binding(1) var<storage, read_write> acc: array<atomic<i32>>;

var<workgroup> sh_slot: array<atomic<i32>, WG * ACC_WORDS>;
var<workgroup> sh_head: array<u32, WG>;

@compute @workgroup_size(WG)
fn merge(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i = wg_id.x * WG + t;
    let valid = i < config.n_entries;

    // Run heads: first entry of a path, or a different tile location than the previous entry.
    // The first entry of a workgroup is a pseudo-head collecting its workgroup's part of a run.
    var e = vec2(0u, 0u);
    var p_ix = 0u;
    var is_head = t == 0u;
    if valid {
        e = entries[i];
        p_ix = path_of_entry(i);
        if i == paths[p_ix].entry_start || entries[i - 1u].y != e.y {
            is_head = true;
        }
    }

    // Each entry's head lane: prefix max of the head lanes.
    var h = select(0u, t, is_head);
    sh_head[t] = h;
    for (var d = 1u; d < WG; d <<= 1u) {
        workgroupBarrier();
        var other = 0u;
        if t >= d {
            other = sh_head[t - d];
        }
        workgroupBarrier();
        h = max(h, other);
        sh_head[t] = h;
    }
    let head = h;

    // Per-line payload in fixed point.
    var payload: array<i32, ACC_WORDS>;
    for (var k = 0u; k < ACC_WORDS; k++) {
        payload[k] = 0;
    }
    if valid {
        let line = entry_line(paths[p_ix], e);
        let xy = entry_xy(e);
        let p0x = line.x - f32(xy.x * TILE_W);
        let p0y = line.y - f32(xy.y * TILE_H);
        let p1x = line.z - f32(xy.x * TILE_W);
        let p1y = line.w - f32(xy.y * TILE_H);
        if p0y != p1y {
            let sign = select(-1.0, 1.0, p0y > p1y);
            let down = p0y < p1y;
            let top_y = select(p1y, p0y, down);
            let top_x = select(p1x, p0x, down);
            let bot_y = select(p0y, p1y, down);
            let bot_x = select(p0x, p1x, down);
            let dx = bot_x - top_x;
            let dy = bot_y - top_y;
            let vertical = dx == 0.0;
            let y_slope = dy / select(dx, 1.0, vertical);
            let x_slope = dx / dy;
            for (var row = 0u; row < TILE_H; row++) {
                let ymin = max(top_y, f32(row));
                let ymax = min(bot_y, f32(row) + 1.0);
                var acc_w = 0.0;
                for (var col = 0u; col < TILE_W; col++) {
                    let px_left = f32(col);
                    let px_right = f32(col) + 1.0;
                    let tl = px_left - top_x;
                    let tr = px_right - top_x;
                    var raw_l = top_y + tl * y_slope;
                    var raw_r = top_y + tr * y_slope;
                    if vertical {
                        raw_l = select(1e30, -1e30, tl <= 0.0);
                        raw_r = select(1e30, -1e30, tr <= 0.0);
                    }
                    let yl = min(max(raw_l, ymin), ymax);
                    let yr = min(max(raw_r, ymin), ymax);
                    let xl = top_x + (yl - top_y) * x_slope;
                    let xr = top_x + (yr - top_y) * x_slope;
                    let hh = abs(yr - yl);
                    let area = hh * (px_right - 0.5 * (xl + xr));
                    payload[col * TILE_H + row] = i32(round((sign * area + acc_w) * FIXED_ONE));
                    acc_w += sign * hh;
                }
                payload[TILE_PIXELS + row] = i32(round(acc_w * FIXED_ONE));
            }
        }
    }

    // Segmented sum into the head's slot.
    if is_head {
        for (var k = 0u; k < ACC_WORDS; k++) {
            atomicStore(&sh_slot[t * ACC_WORDS + k], payload[k]);
        }
    }
    workgroupBarrier();
    if valid && !is_head {
        for (var k = 0u; k < ACC_WORDS; k++) {
            atomicAdd(&sh_slot[head * ACC_WORDS + k], payload[k]);
        }
    }
    workgroupBarrier();
    if valid && is_head {
        let base = entry_block[i] * ACC_WORDS;
        for (var k = 0u; k < ACC_WORDS; k++) {
            let v = atomicLoad(&sh_slot[t * ACC_WORDS + k]);
            if v != 0 {
                atomicAdd(&acc[base + k], v);
            }
        }
    }
}
