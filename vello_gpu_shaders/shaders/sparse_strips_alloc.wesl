// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Strip allocation for GPU sparse strips: a reduce-then-scan over the sorted tile entries
//! with the monoid of `helpers::sparse_strips`, then the alloc pass that emits strip
//! descriptors. See `vello_gpu::sparse_strips::model` for the algorithm.

import package::helpers::sparse_strips::{
    Agg, PathInfo, StripDesc, config, paths, aux, entries, agg_empty, agg_multi, combine,
    path_of_entry, path_should_fill, row_coarse, row_bg_prefix, entry_xy, entry_line,
    winding_delta, TILE_W, TILE_H, PATH_CULLED, STRIP_REAL, STRIP_SOLID, STRIP_TERMINATOR,
    STRIP_SENTINEL, STRIP_FILL_GAP, AGG_NONEMPTY, AGG_MULTI_ROW,
};

const WG: u32 = 256u;
const RESOLVE_WG: u32 = 64u;

const L_PATH_HEAD: u32 = 1u;
const L_PATH_LAST: u32 = 2u;
const L_ROW_HEAD: u32 = 4u;
const L_ROW_LAST: u32 = 8u;
const L_RUN_HEAD: u32 = 16u;
const L_STRIP_HEAD: u32 = 32u;
const L_CAPTIVE: u32 = 64u;

@group(1) @binding(0) var<storage, read_write> wg_aggs: array<Agg>;
@group(1) @binding(1) var<storage, read_write> entry_block: array<u32>;
@group(1) @binding(2) var<storage, read_write> strip_descs: array<StripDesc>;
// Dispatch arguments of the resolve pass `(x, 1, 1)`, then the strip and block totals.
@group(1) @binding(3) var<storage, read_write> indirect: array<u32>;

var<workgroup> sh_agg: array<Agg, WG>;

// Per-entry facts that do not depend on the prefix; see `model::Local`.
struct Leaf {
    agg: Agg,
    path: u32,
    x: u32,
    y: u32,
    flags: u32,
    d: i32,
    coarse: i32,
    bg_lo: u32,
    n_bg: u32,
    trail_lo: u32,
    n_trail: u32,
}

fn leaf(i: u32) -> Leaf {
    let path_ix = path_of_entry(i);
    let p = paths[path_ix];
    let e = entries[i];
    let xy = entry_xy(e);
    let line = entry_line(p, e);

    let path_head = i == p.entry_start;
    let path_last = i + 1u == p.entry_end;
    var prev = vec2(0xffffffffu, 0xffffffffu);
    if !path_head {
        prev = entry_xy(entries[i - 1u]);
    }
    let row_head = path_head || prev.y != xy.y;
    let run_head = row_head || prev.x != xy.x;
    let strip_head = run_head && (row_head || prev.x + 1u != xy.x);
    var row_last = path_last;
    if !path_last {
        row_last = entry_xy(entries[i + 1u]).y != xy.y;
    }

    var l: Leaf;
    l.path = path_ix;
    l.x = xy.x;
    l.y = xy.y;
    l.flags = select(0u, L_PATH_HEAD, path_head) | select(0u, L_PATH_LAST, path_last)
        | select(0u, L_ROW_HEAD, row_head) | select(0u, L_ROW_LAST, row_last)
        | select(0u, L_RUN_HEAD, run_head) | select(0u, L_STRIP_HEAD, strip_head);
    l.d = winding_delta(line, e);
    l.coarse = 0;
    l.bg_lo = 0u;
    l.n_bg = 0u;
    l.trail_lo = 0u;
    l.n_trail = 0u;
    var captive = false;

    if (p.flags & PATH_CULLED) != 0u {
        if row_head {
            l.coarse = row_coarse(p.row_base, xy.y);
            var lo = 0u;
            if !path_head {
                lo = row_bg_prefix(p.row_base, prev.y + 1u);
            }
            l.bg_lo = lo;
            l.n_bg = row_bg_prefix(p.row_base, xy.y) - lo;
            captive = path_should_fill(p.flags, l.coarse) && xy.x != 0u;
            l.flags |= select(0u, L_CAPTIVE, captive);
        }
        if path_last {
            var lo = p.n_fill_rows;
            if xy.y + 1u < config.rows {
                lo = row_bg_prefix(p.row_base, xy.y + 1u);
            }
            l.trail_lo = lo;
            l.n_trail = p.n_fill_rows - lo;
        }
    }

    var strips = 2u * l.n_bg + u32(captive) + u32(strip_head);
    var blocks = l.n_bg + u32(captive) + u32(run_head);
    if path_last {
        strips += 2u * l.n_trail + 1u;
        blocks += l.n_trail;
    }
    let w = l.d + l.coarse;
    l.agg = Agg(vec2(path_ix, xy.y), vec2(path_ix, xy.y), w, w, strips, blocks, AGG_NONEMPTY);
    return l;
}

// Inclusive workgroup scan of `v`; afterwards `sh_agg[t]` holds thread `t`'s inclusive prefix.
fn workgroup_scan(t: u32, v: Agg) -> Agg {
    var acc = v;
    sh_agg[t] = acc;
    for (var d = 1u; d < WG; d <<= 1u) {
        workgroupBarrier();
        var other = agg_empty();
        if t >= d {
            other = sh_agg[t - d];
        }
        workgroupBarrier();
        acc = combine(other, acc);
        sh_agg[t] = acc;
    }
    workgroupBarrier();
    return acc;
}

@compute @workgroup_size(WG)
fn reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let i = wg_id.x * WG + local_id.x;
    var v = agg_empty();
    if i < config.n_entries {
        v = leaf(i).agg;
    }
    let incl = workgroup_scan(local_id.x, v);
    if local_id.x == WG - 1u {
        wg_aggs[wg_id.x] = incl;
    }
}

// Single workgroup: turn the workgroup aggregates into exclusive prefixes, in place.
@compute @workgroup_size(WG)
fn scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let t = local_id.x;
    let n = config.n_wg;
    let k = (n + WG - 1u) / WG;
    let lo = t * k;
    let hi = min(lo + k, n);
    var v = agg_empty();
    for (var j = lo; j < hi; j++) {
        v = combine(v, wg_aggs[j]);
    }
    workgroup_scan(t, v);
    var prefix = agg_empty();
    if t > 0u {
        prefix = sh_agg[t - 1u];
    }
    for (var j = lo; j < hi; j++) {
        let a = wg_aggs[j];
        wg_aggs[j] = prefix;
        prefix = combine(prefix, a);
    }
}

fn make_desc(x: u32, y: u32, block_start: u32, info: u32, winding: i32, path: u32) -> StripDesc {
    return StripDesc(x | (y << 16u), block_start, info, winding, path);
}

fn emit_solid_row(row: u32, s: u32, blk: u32, path: u32) {
    strip_descs[s] = make_desc(0u, row * TILE_H, blk, STRIP_SOLID, 0, path);
    strip_descs[s + 1u] = make_desc(
        config.viewport_width, row * TILE_H, blk + 1u, STRIP_TERMINATOR | STRIP_FILL_GAP, 0, path);
}

// The alloc pass for one entry; see `model::emit`.
fn emit(i: u32, p: Agg, l: Leaf) {
    let path = paths[l.path];
    let row_head = (l.flags & L_ROW_HEAD) != 0u;

    // Resolve the exclusive prefix; see `model::before`.
    var terminators = 0u;
    if (p.flags & AGG_NONEMPTY) != 0u {
        if agg_multi(p) {
            terminators += u32(p.w_first != 0);
            terminators += u32(row_head && p.w_last != 0);
        } else {
            terminators += u32(row_head && p.w_first != 0);
        }
    }
    let winding = select(p.w_last, l.coarse, row_head);
    var s = p.strips + terminators;
    var blk = p.blocks;

    if (l.flags & L_PATH_HEAD) != 0u {
        paths[l.path].strip_start = s;
        paths[l.path].block_start = blk;
    }
    if row_head && (path.flags & PATH_CULLED) != 0u {
        for (var k = 0u; k < l.n_bg; k++) {
            let row = aux[path.fill_row_base + l.bg_lo + k];
            emit_solid_row(row, s, blk, l.path);
            s += 2u;
            blk += 1u;
        }
        if (l.flags & L_CAPTIVE) != 0u {
            strip_descs[s] = make_desc(0u, l.y * TILE_H, blk, STRIP_SOLID, 0, l.path);
            s += 1u;
            blk += 1u;
        }
    }
    if (l.flags & L_STRIP_HEAD) != 0u {
        let fill_gap = path_should_fill(path.flags, winding) && l.x != 0u;
        let info = STRIP_REAL | select(0u, STRIP_FILL_GAP, fill_gap);
        strip_descs[s] = make_desc(l.x * TILE_W, l.y * TILE_H, blk, info, winding, l.path);
        s += 1u;
    }
    if (l.flags & L_RUN_HEAD) != 0u {
        entry_block[i] = blk;
        blk += 1u;
    } else {
        entry_block[i] = p.blocks - 1u;
    }
    if (l.flags & L_ROW_LAST) != 0u {
        let total = winding + l.d;
        if total != 0 {
            let info = STRIP_TERMINATOR
                | select(0u, STRIP_FILL_GAP, path_should_fill(path.flags, total));
            strip_descs[s] = make_desc(config.viewport_width, l.y * TILE_H, blk, info, total, l.path);
            s += 1u;
        }
    }
    if (l.flags & L_PATH_LAST) != 0u {
        var last_y = l.y;
        for (var k = 0u; k < l.n_trail; k++) {
            let row = aux[path.fill_row_base + l.trail_lo + k];
            emit_solid_row(row, s, blk, l.path);
            s += 2u;
            blk += 1u;
            last_y = row;
        }
        strip_descs[s] = make_desc(0xffffu, last_y * TILE_H, blk, STRIP_SENTINEL, 0, l.path);
        s += 1u;
        paths[l.path].strip_end = s;
        paths[l.path].block_end = blk;
        if i + 1u == config.n_entries {
            indirect[0] = (s + RESOLVE_WG - 1u) / RESOLVE_WG;
            indirect[1] = 1u;
            indirect[2] = 1u;
            indirect[3] = s;
            indirect[4] = blk;
        }
    }
}

@compute @workgroup_size(WG)
fn alloc(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i = wg_id.x * WG + t;
    var l: Leaf;
    var v = agg_empty();
    if i < config.n_entries {
        l = leaf(i);
        v = l.agg;
    }
    workgroup_scan(t, v);
    var excl = wg_aggs[wg_id.x];
    if t > 0u {
        excl = combine(excl, sh_agg[t - 1u]);
    }
    if i < config.n_entries {
        emit(i, excl, l);
    }
}
