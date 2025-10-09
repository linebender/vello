// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
struct Config {
    // Count of pre merge tiles
    pmt_count: u32,
    // Count of tiles which will write to the alpha texture
    end_tile_count: u32,
    c: u32,
    d: u32,
};

struct PreMergeTile {
    // The index into the alpha buffer that an "end tile" should write to.
    alpha_index: u32,
    // Contains tile location information
    packed_info: u32,
    // The exclusive prefix sum of the signed winding number.
    // Uploading it saves doing the scan here
    scanned_winding: i32,
    padding: u32,
    // Line points, adjusted by the tile position
    p0: vec2f,
    p1: vec2f,
};

const TILE_HEIGHT = 4u;
const TILE_WIDTH = 4u;

@group(0) @binding(0)
var<uniform> config : Config;

@group(0) @binding(1)
var<storage, read> pmt_in: array<PreMergeTile>;

@group(0) @binding(2)
var<storage, read_write> stitch_indicator: array<u32>;

@group(0) @binding(3)
var<storage, read_write> part_indicator: array<u32>;

@group(0) @binding(4)
var<storage, read_write> stitch_loc: array<array<array<f32, TILE_HEIGHT>, TILE_WIDTH>>;

// This is double strided. This is because we need either:
// 1) The reduction from the last segment start to the end of the partition
// 2) The reduction from the last segment start to the last tile start
// These are not mutually exclusive! Stitching requires both!
@group(0) @binding(5)
var<storage, read_write> part_acc: array<array<f32, TILE_HEIGHT>>;

@group(0) @binding(6)
var<storage, read_write> part_loc: array<array<array<f32, TILE_HEIGHT>, TILE_WIDTH>>;

@group(0) @binding(7)
var output: texture_storage_2d<rgba32uint, write>;

// @group(0) @binding(7)
// var<storage, read_write> alpha_buff: array<vec4<u32>>;

// MSB                                                                  LSB
// 31------------------21|20------------------12|11-------------------3|2|1|0|
// |      Unused (11)    |  Seg Start ID (9)    |  Tile Start ID (9)   |F|T|E|
//
// F = FILL_RULE_MASK
// T = IS_TILE_FIRST_COL_MASK
// E = IS_END_TILE_MASK
const IS_END_TILE_MASK: u32 = 1u;
const IS_TILE_FIRST_COL_MASK: u32 = 2u;
const FILL_RULE_MASK: u32 = 4u;
const TILE_START_ID_SHIFT: u32 = 3u;
const SEG_START_ID_SHIFT: u32 = 12u;
const ID_MASK: u32 = 0x1ffu;
const INVALID_ID: u32 = 256u;

fn is_end_tile(packed_info: u32) -> bool {
    return (packed_info & IS_END_TILE_MASK) != 0u;
}

fn is_tile_first_col(packed_info: u32) -> bool {
    return (packed_info & IS_TILE_FIRST_COL_MASK) != 0u;
}

fn is_fill_rule_non_zero(packed_info: u32) -> bool {
    return (packed_info & FILL_RULE_MASK) != 0u;
}

fn get_tile_start_id(packed_info: u32) -> u32 {
    return (packed_info >> TILE_START_ID_SHIFT) & ID_MASK;
}

fn get_seg_start_id(packed_info: u32) -> u32 {
    return (packed_info >> SEG_START_ID_SHIFT) & ID_MASK;
}

// MSB                                                                  LSB
// 31|30|29--------------------------------------------------------------0|
// |L|A|                     Partition ID (30 bits)                       |
//
// T = STITCH_LOC_MASK
// A = STITCH_ACC_MASK
const STITCH_LOC_MASK = 1u << 31u;
const STITCH_ACC_MASK = 1u << 30u;
const PART_ID_MASK = 0x3fffffffu;

fn loc_stitch_required(in: u32) -> bool {
    return (in & STITCH_LOC_MASK) != 0u;
}

fn acc_stitch_required(in: u32) -> bool {
    return (in & STITCH_ACC_MASK) != 0u;
}

fn get_part_id(in: u32) -> u32 {
    return in & PART_ID_MASK;
}

// MSB                                                                  LSB
// 31-----------------------------------------------------------------3|1|0|
// |                      Unused (30 bits)                             |T|S|
//
// S = PART_SEG_START_MASK
// T = PART_TILE_START_MASK
const PART_SEG_START_MASK: u32 = 1u << 0u;
const PART_TILE_START_MASK: u32 = 1u << 1u;
fn part_has_seg_start(in: u32) -> bool {
    return (in & PART_SEG_START_MASK) != 0u;
}

fn part_has_tile_start(in: u32) -> bool {
    return (in & PART_TILE_START_MASK) != 0u;
}

const BLOCK_DIM = 256u;
const LG_BLOCK_DIM = 8u;
const SCAN_CONST = LG_BLOCK_DIM - 1u;
const TILE_SIZE = TILE_HEIGHT * TILE_WIDTH;
const EPSILON = 1e-6;

// TODO! Do not use more than 4096
var<workgroup> wg_acc: array<array<f32, TILE_HEIGHT>, BLOCK_DIM>;
var<workgroup> wg_loc: array<array<array<f32, TILE_HEIGHT>, TILE_WIDTH>, BLOCK_DIM>;

// 1 thread per pre_pmt
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn merge(@builtin(local_invocation_id) tid: vec3<u32>,
         @builtin(global_invocation_id) gid: vec3<u32>,
         @builtin(workgroup_id) wgid: vec3<u32>)  {
    var pmt: PreMergeTile;
    if (gid.x < config.pmt_count) {
        pmt = pmt_in[gid.x];
    } else {
        pmt = PreMergeTile(
            0xffffffffu,
            0u,
            0,
            0u,
            vec2f(0.0, 0.0),
            vec2f(0.0, 0.0),
        );
    }

    var acc = array<f32, TILE_HEIGHT>();
    var loc = array<array<f32, TILE_HEIGHT>, TILE_WIDTH>();
    if (abs(pmt.p0.y - pmt.p1.y) >= EPSILON) { // If not horizontal. . .
        var line_top_y: f32;
        var line_top_x: f32;
        var line_bottom_y: f32;
        var line_bottom_x: f32;
        if (pmt.p0.y < pmt.p1.y) {
            line_top_y = pmt.p0.y;
            line_top_x = pmt.p0.x;
            line_bottom_y = pmt.p1.y;
            line_bottom_x = pmt.p1.x;
        } else {
            line_top_y = pmt.p1.y;
            line_top_x = pmt.p1.x;
            line_bottom_y = pmt.p0.y;
            line_bottom_x = pmt.p0.x;
        }

        var line_left_x: f32;
        var line_left_y: f32;
        var line_right_x: f32;
        if (pmt.p0.x < pmt.p1.x) {
            line_left_x = pmt.p0.x;
            line_left_y = pmt.p0.y;
            line_right_x = pmt.p1.x;
        } else {
            line_left_x = pmt.p1.x;
            line_left_y = pmt.p1.y;
            line_right_x = pmt.p0.x;
        }

        let dx = line_bottom_x - line_top_x;
        let dy = line_bottom_y - line_top_y;
        let is_vertical = abs(dx) < EPSILON;
        let y_slope = dy / dx;
        let x_slope = dx / dy;

        // Unnecessary, we remove vertical and horizontal cases, so a comparison will be sufficient;
        // i.e. dont care about sign of zero
        //let sign = select(1.0f, -1.0f, (bitcast<u32>(pmt.p0.y - pmt.p1.y) & 0x80000000u) != 0u);
        let sign = select(1.0f, -1.0f, pmt.p0.y < pmt.p1.y);

        if (is_tile_first_col(pmt.packed_info) && line_left_x < 0.0f) {
            var ymin: f32;
            var ymax: f32;
            if (is_vertical) {
                ymin = line_top_y;
                ymax = line_bottom_y;
            } else {
                let line_viewport_left_y =
                    min(max((line_top_y - line_top_x * y_slope), line_top_y), line_bottom_y);
                ymin = min(line_left_y, line_viewport_left_y);
                ymax = max(line_left_y, line_viewport_left_y);
            }

            var h = array<f32, TILE_HEIGHT>();
            for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                let px_top_y = f32(y);
                let px_bottom_y = 1.0f + px_top_y;
                let ymin_clamped = max(ymin, px_top_y);
                let ymax_clamped = min(ymax, px_bottom_y);
                h[y] = max(ymax_clamped - ymin_clamped, 0.0f);
                acc[y] = sign * h[y];
            }
        } else {
            // No need to clear in wgsl, but other shader languages required.
            for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                acc[y] = 0.0f;
            }
        }

        if (line_right_x >= 0.0f) {
            if (is_vertical) {
                let x_int = u32(floor(line_top_x));
                for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                    for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                        var h = 0.0f;
                        var area = 0.0f;
                        if x == x_int {
                            let px_top_y = f32(y);
                            let px_bottom_y = 1.0f + px_top_y;
                            let ymin = max(px_top_y, line_top_y);
                            let ymax = min(px_bottom_y, line_bottom_y);
                            let coverage_right = f32(x) + 1.0f - line_top_x;
                            h = max(ymax - ymin, 0.0f);
                            area = h * coverage_right;
                        }
                        loc[x][y] = acc[y] + sign * area;
                        acc[y] += sign * h;
                    }
                }
            }

            if (!is_vertical) {
                for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                    let px_left_x = f32(x);
                    let px_right_x = 1.0f + px_left_x;
                    for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                        let px_top_y = f32(y);
                        let px_bottom_y = 1.0f + px_top_y;
                        let ymin = max(line_top_y, px_top_y);
                        let ymax = min(line_bottom_y, px_bottom_y);

                        let line_px_left_y =
                            min(max((line_top_y + (px_left_x - line_top_x) * y_slope), ymin), ymax);
                        let line_px_right_y =
                            min(max((line_top_y + (px_right_x - line_top_x) * y_slope), ymin), ymax);
                        let line_px_left_yx =
                            line_top_x + (line_px_left_y - line_top_y) * x_slope;
                        let line_px_right_yx =
                            line_top_x + (line_px_right_y - line_top_y) * x_slope;
                        let h = abs(line_px_right_y - line_px_left_y);
                        let area =
                            0.5f * h * (2.0f * px_right_x - line_px_right_yx - line_px_left_yx);
                        loc[x][y] = acc[y] + sign * area;
                        acc[y] += sign * h;
                    }
                }
            }

            wg_loc[tid.x] = loc;
        } else {
            for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                wg_loc[tid.x][x] = acc;
            }
        }
        wg_acc[tid.x] = acc;
    } else {
        // TODO For non wgsl this needs to clear
        wg_acc[tid.x] = acc;
        wg_loc[tid.x] = loc;
    }

    for (var i = 1u; i <= BLOCK_DIM; i <<= 1u) {
        workgroupBarrier();
        let ii = tid.x - i;
        if (tid.x >= i) {
            for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                acc[y] += wg_acc[ii][y];
            }

            for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    loc[x][y] += wg_loc[ii][x][y];
                }
            }
        }
        workgroupBarrier();

        if (tid.x >= i) {
            wg_acc[tid.x] = acc;
            wg_loc[tid.x] = loc;
        }
    }
    workgroupBarrier();

    var scanned_winding = 0.0f;
    let end_tile = is_end_tile(pmt.packed_info);
    let seg_start_id = get_seg_start_id(pmt.packed_info);
    let tile_start_id = get_tile_start_id(pmt.packed_info);
    let seg_is_valid = seg_start_id != INVALID_ID;
    let tile_is_valid = tile_start_id != INVALID_ID;
    if (end_tile || tid.x == BLOCK_DIM - 1u) {
        if (tile_is_valid) {
            if (tile_start_id != 0u) {
                for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                    for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                        loc[x][y] -= wg_loc[tile_start_id - 1u][x][y];
                    }
                }
                acc = wg_acc[tile_start_id - 1u];
            } else {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    acc[y] = 0.0f;
                }
            }
        }

        if (seg_is_valid) {
            scanned_winding = f32(pmt_in[(wgid.x << LG_BLOCK_DIM) + seg_start_id].scanned_winding);
            if (seg_start_id != 0u) {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    acc[y] += scanned_winding - wg_acc[seg_start_id - 1u][y];
                }
            } else {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    acc[y] += scanned_winding;
                }
            }
        }

        // At this point, if this is the last thread in the threadBlock, this contains the reduction
        // from the tile start
        if (tid.x == BLOCK_DIM - 1u) {
            part_loc[wgid.x] = loc;
        }

        // This will add in the acc for the cases we want: tile_is_valid, tile_is_valid && seg_is_valid
        if (tile_is_valid) {
            for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    loc[x][y] += acc[y];
                }
            }
        }

        // Only end tiles participate in the write out
        if (end_tile) {
            var s_ind = 0u;
            if (tile_is_valid && seg_is_valid) { // Safe to write out
                var final_alphas: vec4<u32>;
                if (is_fill_rule_non_zero(pmt.packed_info)) {
                    for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                        final_alphas[x] = pack4x8unorm(abs(vec4<f32>(loc[x][0], loc[x][1],
                                                                     loc[x][2], loc[x][3])));
                    }
                } else {
                    // EvenOdd fill rule logic
                    for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                        let area = vec4<f32>(loc[x][0], loc[x][1], loc[x][2], loc[x][3]);
                        let im1 = floor(area * 0.5 + 0.5);
                        let coverage = abs(area - 2.0 * im1);
                        final_alphas[x] = pack4x8unorm(coverage);
                    }
                }
                let tex_dims = textureDimensions(output);
                let tex_width = tex_dims.x;
                let output_coords = vec2<u32>(
                    (pmt.alpha_index >> 4) % tex_width,
                    (pmt.alpha_index >> 4) / tex_width,
                );
                if (output_coords.y < tex_dims.y) {
                    textureStore(output, output_coords, final_alphas);
                }
            } else {
                // TODO fill rule on stitch
                s_ind = wgid.x |
                    select(0u, STITCH_ACC_MASK, tile_is_valid && !seg_is_valid) |
                    select(0u, STITCH_LOC_MASK, !tile_is_valid);
                stitch_loc[pmt.alpha_index >> 4u] = loc;
            }
            stitch_indicator[pmt.alpha_index >> 4u] = s_ind;
        }
    }

    if (tid.x == BLOCK_DIM - 1) {
        part_acc[wgid.x << 1u] = acc;
        var seg_acc = wg_acc[tid.x];
        if (seg_is_valid) {
            if (seg_start_id != 0u) {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    seg_acc[y] += scanned_winding - wg_acc[seg_start_id - 1u][y];
                }
            } else {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    seg_acc[y] += scanned_winding;
                }
            }
        }
        part_acc[(wgid.x << 1u) + 1u] = seg_acc;

        part_indicator[wgid.x] = select(0u, PART_TILE_START_MASK, tile_is_valid) |
                                 select(0u, PART_SEG_START_MASK, seg_is_valid);
    }
}

// 1 thread per end_tile
// TODO once subgroups, this should be 1 subgroup : tile
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn stitch(@builtin(local_invocation_id) tid: vec3<u32>,
          @builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(workgroup_id) wgid: vec3<u32>)  {
    var s_indicator = 0u;
    if (gid.x < config.end_tile_count) {
        s_indicator = stitch_indicator[gid.x];
    }

    // No stitching needed or oob
    if (s_indicator == 0u) {
        return;
    }
    let part_id = get_part_id(s_indicator);
    var loc = stitch_loc[gid.x];
    var acc = array<f32, TILE_HEIGHT>();
    if (loc_stitch_required(s_indicator)) {
        var lookback_id = part_id - 1u;
        var part_ind: u32;
        while (true) {
            let p_loc = part_loc[lookback_id];
            for (var x = 0u; x < TILE_WIDTH; x += 1u) {
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    loc[x][y] += p_loc[x][y];
                }
            }

            // Did we hit a tile start?
            part_ind = part_indicator[lookback_id];
            if (part_has_tile_start(part_ind)) {
                break;
            } else {
                lookback_id -= 1u;
            }
        }

        lookback_id <<= 1u;
        acc = part_acc[lookback_id];

        // If the tile start also included a seg start, we're done. Else we will have to traverse to
        // the previous seg start
        if (!part_has_seg_start(part_ind)) {
            lookback_id -= 1u; // Down to the upper
            while (true) {
                let p_acc = part_acc[lookback_id];
                for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                    acc[y] += p_acc[y];
                }

                if (part_has_seg_start(part_indicator[lookback_id >> 1u])) {
                    break;
                } else {
                    lookback_id -= 2u;
                }
            }
        }
    }

    if (acc_stitch_required(s_indicator)) {
        var lookback_id = (part_id - 1u) << 1u;
        while (true) {
            var s_acc = part_acc[lookback_id + 1u];
            for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
                acc[y] += s_acc[y];
            }

            if (part_has_seg_start(part_indicator[lookback_id >> 1u])) {
                break;
            } else {
                lookback_id -= 2u;
            }
        }
    }

    // Combine acc and loc
    for (var x = 0u; x < TILE_WIDTH; x += 1u) {
        for (var y = 0u; y < TILE_HEIGHT; y += 1u) {
            loc[x][y] += acc[y];
        }
    }

    var final_alphas: vec4<u32>;
    if (true) { // TODO put fill rule onto the stitch
        for (var x = 0u; x < TILE_WIDTH; x += 1u) {
            final_alphas[x] = pack4x8unorm(abs(vec4<f32>(loc[x][0], loc[x][1],
                                                         loc[x][2], loc[x][3])));
        }
    } else {
        // EvenOdd fill rule logic
        for (var x = 0u; x < TILE_WIDTH; x += 1u) {
            let area = vec4<f32>(loc[x][0], loc[x][1], loc[x][2], loc[x][3]);
            let im1 = floor(area * 0.5 + 0.5);
            let coverage = abs(area - 2.0 * im1);
            final_alphas[x] = pack4x8unorm(coverage);
        }
    }
    let tex_dims = textureDimensions(output);
    let tex_width = tex_dims.x;
    let output_coords = vec2<u32>(
        gid.x % tex_width,
        gid.x / tex_width,
    );
    if (output_coords.y < tex_dims.y) {
        textureStore(output, output_coords, final_alphas);
    }
}
