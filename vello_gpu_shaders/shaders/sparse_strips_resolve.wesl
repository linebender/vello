// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The resolve pass of GPU sparse strips: one thread per strip descriptor writes the strip
//! and, for real strips, walks the strip's tiles left to right carrying the per-scanline
//! winding across tiles, applying the fill rule to produce alphas. See
//! `vello_gpu::sparse_strips::model::resolve`.

import package::helpers::sparse_strips::{
    StripDesc, config, paths, aux, row_partial, TILE_H, TILE_PIXELS, ACC_WORDS, FIXED_ONE,
    PATH_EVEN_ODD, PATH_CULLED, PATH_THRESHOLD, STRIP_REAL, STRIP_SOLID, STRIP_FILL_GAP,
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

@compute @workgroup_size(WG)
fn resolve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    let n_strips = paths[config.n_paths - 1u].strip_end;
    if s >= n_strips {
        return;
    }
    let d = strip_descs[s];
    strips[s] = vec2(d.xy, (d.block_start * TILE_PIXELS) | ((d.info & STRIP_FILL_GAP) << 29u));
    let kind = d.info & 3u;
    if kind == STRIP_SOLID {
        let base = d.block_start * 4u;
        for (var k = 0u; k < 4u; k++) {
            alphas[base + k] = 0xffffffffu;
        }
    } else if kind == STRIP_REAL {
        let path = paths[d.path];
        let n_blocks = strip_descs[s + 1u].block_start - d.block_start;
        var carry = vec4(f32(d.winding));
        if (d.xy & 0xffffu) == 0u && (path.flags & PATH_CULLED) != 0u {
            carry += row_partial(path.row_base, (d.xy >> 16u) / TILE_H);
        }
        for (var b = d.block_start; b < d.block_start + n_blocks; b++) {
            let acc_base = b * ACC_WORDS;
            for (var col = 0u; col < 4u; col++) {
                var word = 0u;
                for (var row = 0u; row < 4u; row++) {
                    let w = carry[row] + f32(acc[acc_base + col * 4u + row]) / FIXED_ONE;
                    word |= resolve_alpha(path.flags, w) << (8u * row);
                }
                alphas[b * 4u + col] = word;
            }
            carry += vec4(
                f32(acc[acc_base + TILE_PIXELS]),
                f32(acc[acc_base + TILE_PIXELS + 1u]),
                f32(acc[acc_base + TILE_PIXELS + 2u]),
                f32(acc[acc_base + TILE_PIXELS + 3u]),
            ) / FIXED_ONE;
        }
    }
}
