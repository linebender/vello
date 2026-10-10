// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The resolve passes of GPU sparse strips: `resolve_strips` writes one strip record per
//! descriptor; `resolve_blocks` turns one alpha block per thread into alphas, carrying the
//! per-scanline winding from the blocks before it in the strip and applying the fill rule. See
//! `vello_gpu::sparse_strips::model::resolve`.

import package::helpers::sparse_strips::{
    StripDesc, config, paths, row_partial, TILE_H, TILE_PIXELS, ACC_WORDS, FIXED_ONE,
    PATH_EVEN_ODD, PATH_CULLED, PATH_THRESHOLD, STRIP_SOLID, STRIP_FILL_GAP,
};

const WG: u32 = 64u;

@group(1) @binding(0) var<storage, read> strip_descs: array<StripDesc>;
@group(1) @binding(1) var<storage, read> acc: array<i32>;
// Strips as `(x | y << 16, alpha_idx | fill_gap << 31)`.
@group(1) @binding(2) var<storage, read_write> strips: array<vec2<u32>>;
// Alphas, 16 bytes per block, column-major within the tile.
@group(1) @binding(3) var<storage, read_write> alphas: array<u32>;

fn resolve_alpha(flags: u32, w: f32) -> u32 {
    var coverage = abs(w);
    if (flags & PATH_EVEN_ODD) != 0u {
        let im1 = floor(w * 0.5 + 0.5);
        coverage = abs(w - 2.0 * im1);
    }
    var alpha = u32(min(coverage * 255.0 + 0.5, 255.0));
    if (flags & PATH_THRESHOLD) != 0u {
        alpha = select(0u, 255u, alpha >= ((flags >> 8u) & 0xffu));
    }
    return alpha;
}

// One thread per strip: write the strip record.
@compute @workgroup_size(WG)
fn resolve_strips(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    let n_strips = paths[config.n_paths - 1u].strip_end;
    if s >= n_strips {
        return;
    }
    let d = strip_descs[s];
    let alpha_idx = (d.block_start + config.alpha_base_blocks) * TILE_PIXELS;
    strips[s] = vec2(d.xy, alpha_idx | ((d.info & STRIP_FILL_GAP) << 29u));
}

// The strip owning alpha block `b`: the last descriptor whose `block_start <= b`. Terminator
// and sentinel descriptors start at the block after the strip they follow, so they never win.
fn strip_of_block(b: u32, n_strips: u32) -> u32 {
    var lo = 0u;
    var hi = n_strips;
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if strip_descs[mid].block_start <= b {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// One thread per alpha block: resolve its 16 alphas. The per-scanline winding carried into
// the block is the strip's left-edge winding plus the right-edge carries of the blocks before
// it in the strip.
@compute @workgroup_size(WG)
fn resolve_blocks(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let n_strips = paths[config.n_paths - 1u].strip_end;
    let n_blocks = paths[config.n_paths - 1u].block_end;
    if b >= n_blocks {
        return;
    }
    let d = strip_descs[strip_of_block(b, n_strips)];
    let kind = d.info & 3u;
    if kind == STRIP_SOLID {
        for (var k = 0u; k < 4u; k++) {
            alphas[b * 4u + k] = 0xffffffffu;
        }
        return;
    }
    let path = paths[d.path];
    var carry = vec4(f32(d.winding));
    if (d.xy & 0xffffu) == 0u && (path.flags & PATH_CULLED) != 0u {
        carry += row_partial(path.row_base, (d.xy >> 16u) / TILE_H);
    }
    for (var j = d.block_start; j < b; j++) {
        let base = j * ACC_WORDS + TILE_PIXELS;
        carry += vec4(
            f32(acc[base]),
            f32(acc[base + 1u]),
            f32(acc[base + 2u]),
            f32(acc[base + 3u]),
        ) / FIXED_ONE;
    }
    let acc_base = b * ACC_WORDS;
    for (var col = 0u; col < 4u; col++) {
        var word = 0u;
        for (var row = 0u; row < 4u; row++) {
            let w = carry[row] + f32(acc[acc_base + col * 4u + row]) / FIXED_ONE;
            word |= resolve_alpha(path.flags, w) << (8u * row);
        }
        alphas[b * 4u + col] = word;
    }
}
