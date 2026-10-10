// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared declarations of the GPU sparse strip passes.
//!
//! The Rust side of these structures is `vello_gpu::sparse_strips`; the algorithm is documented
//! and modelled in `vello_gpu::sparse_strips::model`, which these shaders follow function by
//! function.

// Tile dimensions: 4x4 pixels, 16 alphas per block.
const TILE_W: u32 = 4u;
const TILE_H: u32 = 4u;
const TILE_PIXELS: u32 = 16u;
// `i32` words per accumulation block: 16 pixel windings and 4 scanline carries.
const ACC_WORDS: u32 = 20u;
const FIXED_ONE: f32 = 65536.0;
const ROW_WORDS: u32 = 8u;

const PATH_EVEN_ODD: u32 = 1u;
const PATH_CULLED: u32 = 2u;
const PATH_THRESHOLD: u32 = 4u;

const STRIP_REAL: u32 = 0u;
const STRIP_SOLID: u32 = 1u;
const STRIP_TERMINATOR: u32 = 2u;
const STRIP_SENTINEL: u32 = 3u;
const STRIP_FILL_GAP: u32 = 4u;

const AGG_NONEMPTY: u32 = 1u;
const AGG_MULTI_ROW: u32 = 2u;

struct Config {
    n_entries: u32,
    n_paths: u32,
    rows: u32,
    viewport_width: u32,
    n_wg: u32,
    max_strips: u32,
    max_blocks: u32,
    // Added to the strips' alpha block indices.
    alpha_base_blocks: u32,
}

struct PathInfo {
    entry_start: u32,
    entry_end: u32,
    line_base: u32,
    row_base: u32,
    fill_row_base: u32,
    n_fill_rows: u32,
    flags: u32,
    _pad: u32,
    // Written by the alloc pass.
    strip_start: u32,
    strip_end: u32,
    block_start: u32,
    block_end: u32,
}

const DRAW_DIRECT: u32 = 1u;

// Per-path draw information for the instance pass; see `GpuDrawInfo`.
struct DrawInfo {
    paint: u32,
    payload: u32,
    depth_index: u32,
    flags: u32,
    instance_base: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// A strip instance, matching `StripInstance` in `render.wesl` and `GpuStrip`.
struct Instance {
    xy: u32,
    widths: u32,
    col_idx: u32,
    payload: u32,
    paint: u32,
    depth: u32,
}

struct StripDesc {
    xy: u32,
    block_start: u32,
    info: u32,
    winding: i32,
    path: u32,
}

// Aggregate of the allocation monoid; see `model::Agg`.
struct Agg {
    first_row: vec2<u32>,
    last_row: vec2<u32>,
    w_first: i32,
    w_last: i32,
    strips: u32,
    blocks: u32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> paths: array<PathInfo>;
@group(0) @binding(2) var<storage, read> aux: array<u32>;
// Sorted tiles: `(packed_winding_line_idx, x | y << 16)`.
@group(0) @binding(3) var<storage, read> entries: array<vec2<u32>>;
// Lines as `(x0, y0, x1, y1)`.
@group(0) @binding(4) var<storage, read> lines: array<vec4<f32>>;

fn agg_empty() -> Agg {
    return Agg(vec2(0u), vec2(0u), 0, 0, 0u, 0u, 0u);
}

fn agg_multi(a: Agg) -> bool {
    return (a.flags & AGG_MULTI_ROW) != 0u;
}

// The monoid operation; see `model::combine`.
fn combine(l: Agg, r: Agg) -> Agg {
    if (l.flags & AGG_NONEMPTY) == 0u {
        return r;
    }
    if (r.flags & AGG_NONEMPTY) == 0u {
        return l;
    }
    var w_first: i32;
    var w_last: i32;
    var extra = 0u;
    var multi = true;
    if all(l.last_row == r.first_row) {
        let merged = l.w_last + r.w_first;
        w_last = select(merged, r.w_last, agg_multi(r));
        if agg_multi(l) {
            w_first = l.w_first;
            extra = u32(agg_multi(r) && merged != 0);
        } else {
            w_first = merged;
            multi = agg_multi(r);
        }
    } else {
        w_first = l.w_first;
        w_last = r.w_last;
        extra = u32(agg_multi(l) && l.w_last != 0) + u32(agg_multi(r) && r.w_first != 0);
    }
    return Agg(
        l.first_row,
        r.last_row,
        w_first,
        w_last,
        l.strips + r.strips + extra,
        l.blocks + r.blocks,
        AGG_NONEMPTY | select(0u, AGG_MULTI_ROW, multi),
    );
}

// Index of the path containing entry `i`: the last path with `entry_start <= i`.
fn path_of_entry(i: u32) -> u32 {
    var lo = 0u;
    var hi = config.n_paths;
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if paths[mid].entry_start <= i {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn path_should_fill(flags: u32, winding: i32) -> bool {
    if (flags & PATH_EVEN_ODD) != 0u {
        return (winding & 1) != 0;
    }
    return winding != 0;
}

fn row_coarse(row_base: u32, row: u32) -> i32 {
    return bitcast<i32>(aux[row_base + row * ROW_WORDS]);
}

fn row_bg_prefix(row_base: u32, row: u32) -> u32 {
    return aux[row_base + row * ROW_WORDS + 1u];
}

fn row_partial(row_base: u32, row: u32) -> vec4<f32> {
    let b = row_base + row * ROW_WORDS + 4u;
    return vec4(
        bitcast<f32>(aux[b]),
        bitcast<f32>(aux[b + 1u]),
        bitcast<f32>(aux[b + 2u]),
        bitcast<f32>(aux[b + 3u]),
    );
}

fn entry_xy(e: vec2<u32>) -> vec2<u32> {
    return vec2(e.y & 0xffffu, e.y >> 16u);
}

fn entry_line(p: PathInfo, e: vec2<u32>) -> vec4<f32> {
    return lines[p.line_base + (e.x >> 5u)];
}

// The line's winding contribution; see `model::winding_delta`.
fn winding_delta(line: vec4<f32>, e: vec2<u32>) -> i32 {
    if line.y == line.w || (e.x & 0x10u) == 0u {
        return 0;
    }
    return select(-1, 1, line.y > line.w);
}
