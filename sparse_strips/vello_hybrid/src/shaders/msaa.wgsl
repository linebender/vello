// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
struct Config {
    // Count of pre merge tiles
    pmt_count: u32,
    // Count of tiles which will write to the alpha texture
    end_tile_count: u32,
    workgroups: u32,
    d: u32,
};

struct PreMergeTile {
    /// The index into the alpha buffer that an "end tile" should write to.
    alpha_index: u32,

    /// Contains the Intersection Data and the Line Buffer Index.
    /// MSB                                                           LSB
    /// 31------------------------------------------------------6|5----0|
    /// |             Line Buffer Index (26 bits)                | Ints |
    ///
    /// - **Ints**: Intersection Data (6 bits)
    packed_winding_line_idx: u32,

    /// Contains the Scanned Winding, Tile Start ID, and Flags.
    /// MSB                                                          LSB
    /// 31-----------------23|22-------16|15--------8|7-----------2|1|0|
    /// |   Start ID (9b)    |   Unused  | Scan Wind |   Unused    |F|E|
    ///
    /// - **Start ID**: u16 (Tile Start ID, 9 bits)
    /// - **Scan Wind**: i8 (Exclusive prefix sum)
    /// - **F**: Fill Rule (bool)
    /// - **E**: Is End Tile (bool)
    packed_scanned_winding: u32,

    /// Contains the packed u16 tile x and y coordinates.
    /// MSB                  LSB
    /// 31-------16|15--------0|
    /// |  Tile Y  |  Tile X   |
    packed_xy: u32,

    line: vec4<f32>,
};

const INTERSECTS_TOP_MASK: u32          = 1 << 0;
const INTERSECTS_BOTTOM_MASK: u32       = 1 << 1;
const INTERSECTS_LEFT_MASK: u32         = 1 << 2;
const INTERSECTS_RIGHT_MASK: u32        = 1 << 3;
const PERFECT_MASK: u32                 = 1 << 4;
const WINDING_MASK: u32                 = 1 << 5;
const STRICT_INTERSECTION_MASK: u32     = 0x3fu;
const LINE_SHIFT: u32                   = 6u;
const SCANNED_WINDING_SHIFT_DOWN: u32   = 24u;
const SCANNED_WINDING_SHIFT_UP: u32     = 16u;
const INVALID_ID: u32                   = 256u;

fn get_intersection_data(packed_winding_line_idx: u32) -> u32 {
    return packed_winding_line_idx & STRICT_INTERSECTION_MASK;
}

fn get_line_idx(packed_winding_line_idx: u32) -> u32 {
    return packed_winding_line_idx >> LINE_SHIFT;
}

fn get_start_tile_idx(packed_scanned_winding: u32) -> u32 {
    return packed_scanned_winding >> 23;
}

fn get_scanned_winding(packed_scanned_winding: u32) -> u32 {
    return u32(i32(packed_scanned_winding << SCANNED_WINDING_SHIFT_UP) >>
        SCANNED_WINDING_SHIFT_DOWN);
}

fn is_end_tile(packed_scanned_winding: u32) -> bool {
    return bool(packed_scanned_winding & 1);
}

fn get_tile_x(packed_xy: u32) -> u32 {
    return packed_xy & 0xffffu;
}

fn get_tile_y(packed_xy: u32) -> u32 {
    return packed_xy >> 16;
}

const TILE_HEIGHT = 4u;
const TILE_WIDTH = 4u;

@group(0) @binding(0)
var<uniform> config : Config;

@group(0) @binding(1)
var<storage, read> msaa_lut: array<u32>;

@group(0) @binding(2)
var<storage, read> pmt_in: array<PreMergeTile>;

// One per workgroup partition
@group(0) @binding(3)
var<storage, read_write> part_indicator: array<u32>;

@group(0) @binding(4)
var<storage, read_write> part_red: array<array<array<vec2<u32>, VEC2_COLS>, TILE_HEIGHT>>;

@group(0) @binding(5)
var output: texture_storage_2d<rgba32uint, write>;

@group(0) @binding(6)
//var<storage, read_write> debug: array<vec2<f32>>;
var<storage, read_write> debug: array<vec2<u32>>;

const BLOCK_DIM = 256u;
const LG_BLOCK_DIM = 8u;
const SCAN_CONST = LG_BLOCK_DIM - 1u;
const TILE_SIZE = TILE_HEIGHT * TILE_WIDTH;
const PART_SIZE = BLOCK_DIM / 4; // 1 thread : row
const EPSILON = 1e-6;

const VEC2_COLS: u32 = TILE_WIDTH * 2 / 2; // packed vec2s per row

// MUST EXACTLY MATCH LUT GENERATION ON CPU
const MASK_WIDTH: u32 = 64;
const MASK_HEIGHT: u32 = 64;
const PACKING_SCALE: f32 = 0.5;
fn get_msaa_mask(local_p0: vec2f, local_p1: vec2f) -> u32 {
    let dir = local_p1 - local_p0;
    let n_unnorm = vec2f(dir.y, -dir.x);
    let len = max(length(n_unnorm), 1e-9);
    let n = n_unnorm / len;

    let c_raw = dot(n, local_p0) - 0.5 * (n.x + n.y);
    let sign_c = select(1.0, -1.0, c_raw < 0.0);
    let c2 = max(0.0, 1.0 - c_raw * PACKING_SCALE * sign_c);
    let n_rev = n * (c2 * sign_c);

    var uv = n_rev * 0.5 + 0.5;
    if (sign_c < 0.0 && uv.x == 0.5) {
        uv.x = 0.5 - 1e-7;
    }

    let mask_dims = vec2f(f32(MASK_WIDTH), f32(MASK_HEIGHT));
    let coord_raw = vec2i(floor(uv * mask_dims));

    let max_coord = vec2i(i32(MASK_WIDTH) - 1, i32(MASK_HEIGHT) - 1);
    let coord = clamp(coord_raw, vec2i(0), max_coord);

    let index = u32(coord.y) * u32(MASK_WIDTH) + u32(coord.x);
    let shift = (index & 3u) * 8u;

    return (msaa_lut[index >> 2u] >> shift) & 0xffu;
}

// 2048 at BLOCK_DIM = 256 and 8xMSAA
// 4096 at BLOCK_DIM = 256 and 16xMSAA
var<workgroup> wg_scan: array<array<array<vec2<u32>, VEC2_COLS>, TILE_HEIGHT>, PART_SIZE>;

// 1 thread per pre_pmt
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn msaa(@builtin(local_invocation_id) tid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>)  {
    var pmt: PreMergeTile;
    if ((gid.x >> 2) < config.pmt_count) {
        pmt = pmt_in[gid.x >> 2];
    } else {
        pmt = PreMergeTile(
            0xffffffffu,
            0u,
            0u,
            0u,
            vec4f(0.0f),
        );
    }

    let p0 = pmt.line.xy;
    let p1 = pmt.line.zw;

    let d = p1 - p0;
    let is_vertical = abs(d.x) <= EPSILON;
    let is_horizontal = abs(d.y) <= EPSILON;
    let idx = select(1.0f / d.x, 0.0f, is_vertical);
    let idy = select(1.0f / d.y, 0.0f, is_horizontal);
    let dxdy = d.x * idy;
    let dydx = d.y * idx;

    let canonical_x_dir = p1.x >= p0.x;
    let canonical_y_dir = p1.y >= p0.y;

    let right = select(p0.x, p1.x, canonical_x_dir);
    let right_in_viewport = right >= 0.0f;
    //let right_in_viewport = false;

    var clipped_top = vec2f(0.0f, 0.0f);
    var clipped_bot = vec2f(0.0f, 0.0f);
    let intersection_data = get_intersection_data(pmt.packed_winding_line_idx);
    if (right_in_viewport) {
        var p_entry = p0;
        var p_exit = p1;

        let tile_min_x_u32 = get_tile_x(pmt.packed_xy) * TILE_WIDTH;
        let tile_min_y_u32 = get_tile_y(pmt.packed_xy) * TILE_HEIGHT;
        let tile_min = vec2f(f32(tile_min_x_u32), f32(tile_min_y_u32));
        let tile_max = vec2f(f32(tile_min_x_u32 + TILE_WIDTH),
                                        f32(tile_min_y_u32 + TILE_HEIGHT));

        let x_canon = vec2<bool>(canonical_x_dir);
        let v_masks = select(vec2u(INTERSECTS_RIGHT_MASK, INTERSECTS_LEFT_MASK),
                                        vec2u(INTERSECTS_LEFT_MASK, INTERSECTS_RIGHT_MASK),
                                        x_canon);
        let v_bounds = select(vec2f(tile_max.x, tile_min.x),
                                         vec2f(tile_min.x, tile_max.x),
                                         x_canon);
        let mask_v_in = v_masks.x;
        let mask_v_out = v_masks.y;
        let bound_v_in = v_bounds.x;
        let bound_v_out = v_bounds.y;

        let y_canon = vec2<bool>(canonical_y_dir);
        let h_masks = select(
            vec2u(INTERSECTS_BOTTOM_MASK, INTERSECTS_TOP_MASK),
            vec2u(INTERSECTS_TOP_MASK, INTERSECTS_BOTTOM_MASK),
            y_canon,
        );
        let h_bounds = select(
            vec2f(tile_max.y, tile_min.y),
            vec2f(tile_min.y, tile_max.y),
            y_canon,
        );
        let mask_h_in = h_masks.x;
        let mask_h_out = h_masks.y;
        let bound_h_in = h_bounds.x;
        let bound_h_out = h_bounds.y;

        let entry_hits = intersection_data & (mask_v_in | mask_h_in);
        {
            let use_h = (intersection_data & mask_h_in) != 0u;
            let p_oriented = select(p0.yx, p0.xy, use_h);
            let bound = select(bound_v_in, bound_h_in, use_h);
            let slope = select(dydx, dxdy, use_h);
            let calculated = p_oriented.x + (bound - p_oriented.y) * slope;
            let candidate = select(vec2f(bound, calculated), vec2f(calculated, bound), use_h);
            p_entry = select(p_entry, candidate, entry_hits != 0u);
        }

        let exit_hits = intersection_data & (mask_v_out | mask_h_out);
        {
            let use_h = (intersection_data & mask_h_out) != 0u;
            let p_oriented = select(p0.yx, p0.xy, use_h);
            let bound = select(bound_v_out, bound_h_out, use_h);
            let slope = select(dydx, dxdy, use_h);
            let calculated = p_oriented.x + (bound - p_oriented.y) * slope;
            let candidate = select(vec2f(bound, calculated), vec2f(calculated, bound), use_h);
            p_exit = select(p_exit, candidate, exit_hits != 0u);
        }

        if ((intersection_data & PERFECT_MASK) != 0u && (exit_hits ^ entry_hits) != 0u) {
            let entry_is_empty = entry_hits == 0u;

            let target_val = select(p_exit, p_entry, entry_is_empty);
            let out_of_bounds = any(target_val < tile_min) || any(target_val > tile_max);

            p_entry = select(p_entry, p_exit, out_of_bounds && entry_is_empty);
            p_exit = select(p_exit, p_entry, out_of_bounds && !entry_is_empty);
        }

        {
            let exit_is_min = p_exit.y < p_entry.y;
            clipped_top = select(p_entry, p_exit, exit_is_min);
            clipped_bot = select(p_exit, p_entry, exit_is_min);

            clipped_top -= tile_min;
            clipped_bot -= tile_min;

            let tile_size = vec2f(f32(TILE_WIDTH), f32(TILE_HEIGHT));
            let zero_vec = vec2f(0.0);

            clipped_top = clamp(clipped_top, zero_vec, tile_size);
            clipped_bot = clamp(clipped_bot, zero_vec, tile_size);
        }
    }

    // Start tiles get the coarse winding, others get seeded with biased 0, per sample, 4 samples
    // to a u32
    var mask: array<vec2u, TILE_WIDTH>;
    {
        let winding = get_scanned_winding(pmt.packed_scanned_winding);
        let w = vec2u(0x80808080u + winding *
            select(0, 0x1010101u, get_start_tile_idx(pmt.packed_scanned_winding) == (tid.x >> 2)));
        for (var col = 0u; col < TILE_WIDTH; col++) {
            mask[col] = w;
        }
    }

    if (right_in_viewport) {
        let left = (intersection_data & INTERSECTS_LEFT_MASK) != 0;
        if (left) {
            let y_edge = select(clipped_bot.y, clipped_top.y, clipped_top.x <= clipped_bot.x);
            let y_cross = u32(ceil(y_edge));
            if ((tid.x & 3) >= y_cross) {
                let v = vec2u(select(0x1010101u, 0xfefefeffu, canonical_x_dir));
                for (var col = 0u; col < TILE_WIDTH; col++) {
                    mask[col] += v;
                }
            }
        }

        // TODO dont diverge the subgroup
        let start_y_f32 = floor(clipped_top.y);
        if !is_horizontal || clipped_top.y != start_y_f32 {
            let start_y = u32(start_y_f32);
            let end_y = u32(ceil(clipped_bot.y));
            let x_dir = clipped_top.x <= clipped_bot.x;
            let row = tid.x & 3;
            if row >= start_y && row < end_y {
                let row_top = f32(row);
                let row_bot = f32(row + 1u);
                let x_cross_top = clipped_top.x + (row_top - clipped_top.y) * dxdy;
                let p_top = select(vec2f(x_cross_top, row_top),
                                              clipped_top,
                                              clipped_top.y >= row_top);
                let x_cross_bot = clipped_top.x + (row_bot - clipped_top.y) * dxdy;
                let p_bot = select(vec2f(x_cross_bot, row_bot),
                                              clipped_bot,
                                              clipped_bot.y <= row_bot);
                if p_top.y <= p_bot.y && p_top.y < row_bot {
                    let y = row;
                    let x_min = min(p_top.x, p_bot.x);
                    let x_max = max(p_top.x, p_bot.x);

                    let x_start = clamp(i32(floor(x_min)), 0, i32(TILE_WIDTH - 1));
                    let x_end = clamp(i32(floor(x_max)), 0, i32(TILE_WIDTH - 1));

                    let pixel_top_touch = p_top.y == floor(p_top.y);
                    let crossed_top = y > start_y || pixel_top_touch;
                    for (var x = x_start; x < i32(TILE_WIDTH); x += 1i) {
                        if x <= x_end {
                            let px = f32(x);
                            let py = f32(y);

                            var msaa_mask = get_msaa_mask(
                                vec2f(p_top.x - px, p_top.y - py),
                                vec2f(p_bot.x - px, p_bot.y - py));

                            let is_start = x == x_start;
                            let is_end = x == x_end;
                            let cannonical_start = (x_dir && is_start) || (!x_dir && is_end);
                            let cannonical_end = (x_dir && is_end) || (!x_dir && is_start);
                            let line_top = cannonical_start && y == start_y;

                            let bumped = (line_top && p_top.x == 0.0 && !pixel_top_touch)
                                || (!line_top && !pixel_top_touch && x_dir);

                            if line_top && !bumped {
                                let mask_shift = u32(round(8.0 * (p_top.y - py)));
                                msaa_mask &= 0xffu << mask_shift;
                            }

                            if cannonical_end && y == end_y - 1 && p_bot.x != 0.0 {
                                let mask_shift = u32(round(8.0 * (p_bot.y - py)));
                                msaa_mask &= ~(0xffu << mask_shift);
                            }

                            let mask_a = msaa_mask ^ (msaa_mask << 7u);
                            let mask_b = mask_a ^ (mask_a << 14u);
                            var exp_mask =
                                vec2u(mask_b & 0x1010101u, mask_b >> 4 & 0x1010101u);

                            if bumped {
                                exp_mask -= 0x1010101u;
                            }

                            if canonical_y_dir {
                                mask[x] += exp_mask;
                            } else {
                                mask[x] -= exp_mask;
                            };
                        } else {
                            if crossed_top {
                                if canonical_y_dir {
                                    mask[x] += 0x1010101u;
                                } else {
                                    mask[x] += 0xfefefeffu;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // load into shared
    let row_index = tid.x & 3;
    let tile_index = tid.x >> 2;
    for (var col = 0u; col < TILE_WIDTH; col += 1u) {
        wg_scan[tile_index][row_index][col] = mask[col];
    }

    // Scan
    for (var i = 1u; i < PART_SIZE; i <<= 1u) {
        workgroupBarrier();
        let ii = tile_index - i;
        if (tile_index >= i) {
            for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                let unbiased = wg_scan[ii][row_index][col] - vec2u(0x80808080u);
                mask[col] += unbiased;
            }
        }
        workgroupBarrier();

        if (tile_index >= i) {
            for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                wg_scan[tile_index][row_index][col] = mask[col];
            }
        }
    }
    workgroupBarrier();

    let tile_start_id = get_start_tile_idx(pmt.packed_scanned_winding);
    let is_valid = tile_start_id != INVALID_ID;
    let is_last_tile = (tid.x >> 2) == PART_SIZE - 1;
    let is_end = is_end_tile(pmt.packed_scanned_winding);
    if (is_end || is_last_tile) {
        if (is_valid && tile_start_id != 0) {
            let pred_id = tile_start_id - 1;
            for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                let unbiased = wg_scan[pred_id][row_index][col] - vec2u(0x80808080u);
                mask[col] -= unbiased;
            }
        }
    }
    workgroupBarrier();

    if (is_end && is_valid && tile_start_id != 0) {
        for (var col = 0u; col < TILE_WIDTH; col += 1u) {
            wg_scan[tile_index][row_index][col] = mask[col];
        }
    }
    workgroupBarrier();

    if (is_last_tile) {
        if (row_index == 0) {
            part_indicator[wgid.x] = u32(is_valid);
        }
        for (var col = 0u; col < TILE_WIDTH; col += 1u) {
            part_red[wgid.x][row_index][col] = mask[col];
        }
    }

    if (is_end) {
        // Signal when cross workgroup reduction is necessary.
        //
        // The first partition will always be valid, so setting the alpha index convienently
        // correctly signals when an end tile requires stitching.
        //
        // Only a single end_tile can be invalid in a workgroup partition.
        //
        // Writing the out the reduction is only necessary if stitching is required, but posting
        // the validity is always necessary.
        let alpha_shifted = pmt.alpha_index >> 4;
        if (row_index == 0 && !is_valid) {
            part_indicator[wgid.x + config.workgroups] = select(alpha_shifted, 0, is_valid);
        }
        if (!is_valid) {
            for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                part_red[wgid.x + config.workgroups][row_index][col] = mask[col];
            }
        }
        if (is_valid && row_index == 0) {
            var final_alphas = vec4u(0);
            if (right_in_viewport) {
                for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                    var col_pixels: vec4<f32>;
                    for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                        let cov_mask = wg_scan[tile_index][y][col];
                        let diff = cov_mask ^ vec2u(0x80808080u);
                        let subbed = diff - 0x01010101u;
                        let zeros = subbed & (~diff) & vec2u(0x80808080u);
                        let actives = zeros ^ vec2u(0x80808080u);
                        let counts = ((actives >> vec2u(7)) * 0x01010101u) >> vec2u(24);
                        col_pixels[y] = f32(counts.x + counts.y) * 0.125;
                    }

                    final_alphas[col] = pack4x8unorm(abs(col_pixels));
                }
            }

            let tex_dims = textureDimensions(output);
            let tex_width = tex_dims.x;
            let output_coords = vec2<u32>(
                alpha_shifted % tex_width,
                alpha_shifted / tex_width
            );
            if (output_coords.y < tex_dims.y) {
                textureStore(output, output_coords, final_alphas);
            }
        }
    }
}

// 1 thread per end_tile
// TODO once subgroups, this should be 1 subgroup : tile
@compute @workgroup_size(128, 1, 1)
fn msaa_stitch(@builtin(local_invocation_id) tid: vec3<u32>,
               @builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(workgroup_id) wgid: vec3<u32>)  {
    if (gid.x >= config.workgroups) {
        return;
    }

    let alpha_index = part_indicator[gid.x + config.workgroups];
    if (alpha_index != 0u) {
        var lookback_id = gid.x - 1;
        var mask = part_red[gid.x + config.workgroups];
        while (true) {
            var prev = part_red[lookback_id];
            for (var row = 0u; row < TILE_HEIGHT; row += 1u) {
                for (var col = 0u; col < TILE_WIDTH; col += 1u) {
                    let unbiased = prev[row][col] - vec2u(0x80808080u);
                    mask[row][col] += unbiased;
                }
            }

            if (part_indicator[lookback_id] != 0u || lookback_id == 0u) {
                break;
            } else {
                lookback_id -= 1u;
            }
        }

        var final_alphas = vec4u(0);
        for (var col = 0u; col < TILE_WIDTH; col += 1u) {
            var col_pixels: vec4<f32>;
            for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                let diff = mask[y][col] ^ vec2u(0x80808080u);
                let subbed = diff - 0x01010101u;
                let zeros = subbed & (~diff) & vec2u(0x80808080u);
                let actives = zeros ^ vec2u(0x80808080u);
                let counts = ((actives >> vec2u(7)) * 0x01010101u) >> vec2u(24);
                col_pixels[y] = f32(counts.x + counts.y) * 0.125;
            }
            final_alphas[col] = pack4x8unorm(abs(col_pixels));
        }
        let tex_dims = textureDimensions(output);
        let tex_width = tex_dims.x;
        let output_coords = vec2<u32>(
            alpha_index % tex_width,
            alpha_index / tex_width
        );
        if (output_coords.y < tex_dims.y) {
            textureStore(output, output_coords, final_alphas);
        }
    }
}
