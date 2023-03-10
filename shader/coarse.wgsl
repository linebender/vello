// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// The coarse rasterization stage.

#import config
#import bump
#import drawtag
#import ptcl
#import tile

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> draw_monoids: array<DrawMonoid>;

// TODO: dedup
struct BinHeader {
    element_count: u32,
    chunk_offset: u32,
}

@group(0) @binding(3)
var<storage> bin_headers: array<BinHeader>;

@group(0) @binding(4)
var<storage> info_bin_data: array<u32>;

@group(0) @binding(5)
var<storage> paths: array<Path>;

@group(0) @binding(6)
var<storage> tiles: array<Tile>;

@group(0) @binding(7)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(8)
var<storage, read_write> ptcl: array<u32>;



// Much of this code assumes WG_SIZE == N_TILE. If these diverge, then
// a fair amount of fixup is needed.
let WG_SIZE = 256u;
//let N_SLICE = WG_SIZE / 32u;
let N_SLICE = 8u;

var<workgroup> sh_bitmaps: array<array<atomic<u32>, N_TILE>, N_SLICE>;
var<workgroup> sh_part_count: array<u32, WG_SIZE>;
var<workgroup> sh_part_offsets: array<u32, WG_SIZE>;
var<workgroup> sh_drawobj_ix: array<u32, WG_SIZE>;
var<workgroup> sh_tile_stride: array<u32, WG_SIZE>;
var<workgroup> sh_tile_width: array<u32, WG_SIZE>;
var<workgroup> sh_tile_x0y0: array<u32, WG_SIZE>;
var<workgroup> sh_tile_count: array<u32, WG_SIZE>;
var<workgroup> sh_tile_base: array<u32, WG_SIZE>;

// helper functions for writing ptcl

var<private> cmd_offset: u32;
var<private> cmd_limit: u32;

// Make sure there is space for a command of given size, plus a jump if needed
fn alloc_cmd(size: u32) {
    if cmd_offset + size >= cmd_limit {
        // We might be able to save a little bit of computation here
        // by setting the initial value of the bump allocator.
        let ptcl_dyn_start = config.width_in_tiles * config.height_in_tiles * PTCL_INITIAL_ALLOC;
        var new_cmd = ptcl_dyn_start + atomicAdd(&bump.ptcl, PTCL_INCREMENT);
        if new_cmd + PTCL_INCREMENT > config.ptcl_size {
            new_cmd = 0u;
            atomicOr(&bump.failed, STAGE_COARSE);
        }
        ptcl[cmd_offset] = CMD_JUMP;
        ptcl[cmd_offset + 1u] = new_cmd;
        cmd_offset = new_cmd;
        cmd_limit = cmd_offset + (PTCL_INCREMENT - PTCL_HEADROOM);
    }
}

fn write_path(tile: Tile, linewidth: f32) -> bool {
    // TODO: take flags
    alloc_cmd(3u);
    if linewidth < 0.0 {
        let even_odd = linewidth < -1.0;
        if tile.segments != 0u {
            let fill = CmdFill(tile.segments, tile.backdrop);
            ptcl[cmd_offset] = CMD_FILL;
            let segments_and_rule = select(fill.tile << 1u, (fill.tile << 1u) | 1u, even_odd);
            ptcl[cmd_offset + 1u] = segments_and_rule;
            ptcl[cmd_offset + 2u] = u32(fill.backdrop);
            cmd_offset += 3u;
        } else {
            if even_odd && (abs(tile.backdrop) & 1) == 0 {
                return false;
            }
            ptcl[cmd_offset] = CMD_SOLID;
            cmd_offset += 1u;
        }
    } else {
        let stroke = CmdStroke(tile.segments, 0.5 * linewidth);
        ptcl[cmd_offset] = CMD_STROKE;
        ptcl[cmd_offset + 1u] = stroke.tile;
        ptcl[cmd_offset + 2u] = bitcast<u32>(stroke.half_width);
        cmd_offset += 3u;
    }
    return true;
}

fn write_color(color: CmdColor) {
    alloc_cmd(2u);
    ptcl[cmd_offset] = CMD_COLOR;
    ptcl[cmd_offset + 1u] = color.rgba_color;
    cmd_offset += 2u;
}

fn write_grad(ty: u32, index: u32, info_offset: u32) {
    alloc_cmd(3u);
    ptcl[cmd_offset] = ty;
    ptcl[cmd_offset + 1u] = index;
    ptcl[cmd_offset + 2u] = info_offset;
    cmd_offset += 3u;
}

fn write_image(info_offset: u32) {
    alloc_cmd(2u);
    ptcl[cmd_offset] = CMD_IMAGE;
    ptcl[cmd_offset + 1u] = info_offset;
    cmd_offset += 2u;
}

fn write_begin_clip() {
    alloc_cmd(1u);
    ptcl[cmd_offset] = CMD_BEGIN_CLIP;
    cmd_offset += 1u;
}

fn write_end_clip(end_clip: CmdEndClip) {
    alloc_cmd(3u);
    ptcl[cmd_offset] = CMD_END_CLIP;
    ptcl[cmd_offset + 1u] = end_clip.blend;
    ptcl[cmd_offset + 2u] = bitcast<u32>(end_clip.alpha);
    cmd_offset += 3u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Exit early if prior stages failed, as we can't run this stage.
    // We need to check only prior stages, as if this stage has failed in another workgroup, 
    // we still want to know this workgroup's memory requirement.   
#ifdef have_uniform
    if local_id.x == 0u {
        // Reuse sh_part_count to hold failed flag, shmem is tight
        sh_part_count[0] = atomicLoad(&bump.failed);
    }
    let failed = workgroupUniformLoad(&sh_part_count[0]);
#else
    let failed = atomicLoad(&bump.failed);
#endif
    if (failed & (STAGE_BINNING | STAGE_TILE_ALLOC | STAGE_PATH_COARSE)) != 0u {
        return;
    }
    let width_in_bins = (config.width_in_tiles + N_TILE_X - 1u) / N_TILE_X;
    let bin_ix = width_in_bins * wg_id.y + wg_id.x;
    let n_partitions = (config.n_drawobj + N_TILE - 1u) / N_TILE;

    // Coordinates of the top left of this bin, in tiles.
    let bin_tile_x = N_TILE_X * wg_id.x;
    let bin_tile_y = N_TILE_Y * wg_id.y;

    let tile_x = local_id.x % N_TILE_X;
    let tile_y = local_id.x / N_TILE_X;
    let this_tile_ix = (bin_tile_y + tile_y) * config.width_in_tiles + bin_tile_x + tile_x;
    cmd_offset = this_tile_ix * PTCL_INITIAL_ALLOC;
    cmd_limit = cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM);

    // clip state
    var clip_zero_depth = 0u;
    var clip_depth = 0u;

    var partition_ix = 0u;
    var rd_ix = 0u;
    var wr_ix = 0u;
    var part_start_ix = 0u;
    var ready_ix = 0u;

    // blend state
    var render_blend_depth = 0u;
    var max_blend_depth = 0u;

    let blend_offset = cmd_offset;
    cmd_offset += 1u;

    while true {
        for (var i = 0u; i < N_SLICE; i += 1u) {
            atomicStore(&sh_bitmaps[i][local_id.x], 0u);
        }

        while true {
            if ready_ix == wr_ix && partition_ix < n_partitions {
                part_start_ix = ready_ix;
                var count = 0u;
                if partition_ix + local_id.x < n_partitions {
                    let in_ix = (partition_ix + local_id.x) * N_TILE + bin_ix;
                    let bin_header = bin_headers[in_ix];
                    count = bin_header.element_count;
                    sh_part_offsets[local_id.x] = bin_header.chunk_offset;
                }
                // prefix sum the element counts
                for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
                    sh_part_count[local_id.x] = count;
                    workgroupBarrier();
                    if local_id.x >= (1u << i) {
                        count += sh_part_count[local_id.x - (1u << i)];
                    }
                    workgroupBarrier();
                }
                sh_part_count[local_id.x] = part_start_ix + count;
#ifdef have_uniform
                ready_ix = workgroupUniformLoad(&sh_part_count[WG_SIZE - 1u]);
#else
                workgroupBarrier();
                ready_ix = sh_part_count[WG_SIZE - 1u];
#endif
                partition_ix += WG_SIZE;
            }
            // use binary search to find draw object to read
            var ix = rd_ix + local_id.x;
            if ix >= wr_ix && ix < ready_ix {
                var part_ix = 0u;
                for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
                    let probe = part_ix + ((N_TILE / 2u) >> i);
                    if ix >= sh_part_count[probe - 1u] {
                        part_ix = probe;
                    }
                }
                ix -= select(part_start_ix, sh_part_count[part_ix - 1u], part_ix > 0u);
                let offset = config.bin_data_start + sh_part_offsets[part_ix];
                sh_drawobj_ix[local_id.x] = info_bin_data[offset + ix];
            }
            wr_ix = min(rd_ix + N_TILE, ready_ix);
            if wr_ix - rd_ix >= N_TILE || (wr_ix >= ready_ix && partition_ix >= n_partitions) {
                break;
            }
        }
        // At this point, sh_drawobj_ix[0.. wr_ix - rd_ix] contains merged binning results.
        var tag = DRAWTAG_NOP;
        var drawobj_ix: u32;
        if local_id.x + rd_ix < wr_ix {
            drawobj_ix = sh_drawobj_ix[local_id.x];
            tag = scene[config.drawtag_base + drawobj_ix];
        }

        var tile_count = 0u;
        // I think this predicate is the same as the last, maybe they can be combined
        if tag != DRAWTAG_NOP {
            let path_ix = draw_monoids[drawobj_ix].path_ix;
            let path = paths[path_ix];
            let stride = path.bbox.z - path.bbox.x;
            sh_tile_stride[local_id.x] = stride;
            let dx = i32(path.bbox.x) - i32(bin_tile_x);
            let dy = i32(path.bbox.y) - i32(bin_tile_y);
            let x0 = clamp(dx, 0, i32(N_TILE_X));
            let y0 = clamp(dy, 0, i32(N_TILE_Y));
            let x1 = clamp(i32(path.bbox.z) - i32(bin_tile_x), 0, i32(N_TILE_X));
            let y1 = clamp(i32(path.bbox.w) - i32(bin_tile_y), 0, i32(N_TILE_Y));
            sh_tile_width[local_id.x] = u32(x1 - x0);
            sh_tile_x0y0[local_id.x] = u32(x0) | u32(y0 << 16u);
            tile_count = u32(x1 - x0) * u32(y1 - y0);
            // base relative to bin
            let base = path.tiles - u32(dy * i32(stride) + dx);
            sh_tile_base[local_id.x] = base;
            // TODO: there's a write_tile_alloc here in the source, not sure what it's supposed to do
        }

        // Prefix sum of tile counts
        sh_tile_count[local_id.x] = tile_count;
        for (var i = 0u; i < firstTrailingBit(N_TILE); i += 1u) {
            workgroupBarrier();
            if local_id.x >= (1u << i) {
                tile_count += sh_tile_count[local_id.x - (1u << i)];
            }
            workgroupBarrier();
            sh_tile_count[local_id.x] = tile_count;
        }
        workgroupBarrier();
        let total_tile_count = sh_tile_count[N_TILE - 1u];
        // Parallel iteration over all tiles
        for (var ix = local_id.x; ix < total_tile_count; ix += N_TILE) {
            // Binary search to find draw object which contains this tile
            var el_ix = 0u;
            for (var i = 0u; i < firstTrailingBit(N_TILE); i += 1u) {
                let probe = el_ix + ((N_TILE / 2u) >> i);
                if ix >= sh_tile_count[probe - 1u] {
                    el_ix = probe;
                }
            }
            drawobj_ix = sh_drawobj_ix[el_ix];
            tag = scene[config.drawtag_base + drawobj_ix];
            let seq_ix = ix - select(0u, sh_tile_count[el_ix - 1u], el_ix > 0u);
            let width = sh_tile_width[el_ix];
            let x0y0 = sh_tile_x0y0[el_ix];
            let x = (x0y0 & 0xffffu) + seq_ix % width;
            let y = (x0y0 >> 16u) + seq_ix / width;
            let tile_ix = sh_tile_base[el_ix] + sh_tile_stride[el_ix] * y + x;
            let tile = tiles[tile_ix];
            let is_clip = (tag & 1u) != 0u;
            var is_blend = false;
            if is_clip {
                let BLEND_CLIP = (128u << 8u) | 3u;
                let scene_offset = draw_monoids[drawobj_ix].scene_offset;
                let dd = config.drawdata_base + scene_offset;
                let blend = scene[dd];
                is_blend = blend != BLEND_CLIP;
            }
            let include_tile = tile.segments != 0u || (tile.backdrop == 0) == is_clip || is_blend;
            if include_tile {
                let el_slice = el_ix / 32u;
                let el_mask = 1u << (el_ix & 31u);
                atomicOr(&sh_bitmaps[el_slice][y * N_TILE_X + x], el_mask);
            }
        }
        workgroupBarrier();
        // At this point bit drawobj % 32 is set in sh_bitmaps[drawobj / 32][y * N_TILE_X + x]
        // if drawobj touches tile (x, y).

        // Write per-tile command list for this tile
        var slice_ix = 0u;
        var bitmap = atomicLoad(&sh_bitmaps[0u][local_id.x]);
        while true {
            if bitmap == 0u {
                slice_ix += 1u;
                // potential optimization: make iteration limit dynamic
                if slice_ix == N_SLICE {
                    break;
                }
                bitmap = atomicLoad(&sh_bitmaps[slice_ix][local_id.x]);
                if bitmap == 0u {
                    continue;
                }
            }
            let el_ix = slice_ix * 32u + firstTrailingBit(bitmap);
            drawobj_ix = sh_drawobj_ix[el_ix];
            // clear LSB of bitmap, using bit magic
            bitmap &= bitmap - 1u;
            let drawtag = scene[config.drawtag_base + drawobj_ix];
            let dm = draw_monoids[drawobj_ix];
            let dd = config.drawdata_base + dm.scene_offset;
            let di = dm.info_offset;
            if clip_zero_depth == 0u {
                let tile_ix = sh_tile_base[el_ix] + sh_tile_stride[el_ix] * tile_y + tile_x;
                let tile = tiles[tile_ix];
                switch drawtag {
                    // DRAWTAG_FILL_COLOR
                    case 0x44u: {
                        let linewidth = bitcast<f32>(info_bin_data[di]);
                        if write_path(tile, linewidth) {
                            let rgba_color = scene[dd];
                            write_color(CmdColor(rgba_color));
                        }
                    }
                    // DRAWTAG_FILL_LIN_GRADIENT
                    case 0x114u: {
                        let linewidth = bitcast<f32>(info_bin_data[di]);
                        if write_path(tile, linewidth) {
                            let index = scene[dd];
                            let info_offset = di + 1u;
                            write_grad(CMD_LIN_GRAD, index, info_offset);
                        }
                    }
                    // DRAWTAG_FILL_RAD_GRADIENT
                    case 0x2dcu: {
                        let linewidth = bitcast<f32>(info_bin_data[di]);
                        if write_path(tile, linewidth) {
                            let index = scene[dd];
                            let info_offset = di + 1u;
                            write_grad(CMD_RAD_GRAD, index, info_offset);
                        }
                    }
                    // DRAWTAG_FILL_IMAGE
                    case 0x248u: {
                        let linewidth = bitcast<f32>(info_bin_data[di]);
                        if write_path(tile, linewidth) {                            
                            write_image(di + 1u);
                        }
                    }
                    // DRAWTAG_BEGIN_CLIP
                    case 0x9u: {
                        if tile.segments == 0u && tile.backdrop == 0 {
                            clip_zero_depth = clip_depth + 1u;
                        } else {
                            write_begin_clip();
                            render_blend_depth += 1u;
                            max_blend_depth = max(max_blend_depth, render_blend_depth);
                        }
                        clip_depth += 1u;
                    }
                    // DRAWTAG_END_CLIP
                    case 0x21u: {
                        clip_depth -= 1u;
                        write_path(tile, -1.0);
                        let blend = scene[dd];
                        let alpha = bitcast<f32>(scene[dd + 1u]);
                        write_end_clip(CmdEndClip(blend, alpha));
                        render_blend_depth -= 1u;
                    }
                    default: {}
                }
            } else {
                // In "clip zero" state, suppress all drawing
                switch drawtag {
                    // DRAWTAG_BEGIN_CLIP
                    case 0x9u: {
                        clip_depth += 1u;
                    }
                    // DRAWTAG_END_CLIP
                    case 0x21u: {
                        if clip_depth == clip_zero_depth {
                            clip_zero_depth = 0u;
                        }
                        clip_depth -= 1u;
                    }
                    default: {}
                }
            }
        }

        rd_ix += N_TILE;
        if rd_ix >= ready_ix && partition_ix >= n_partitions {
            break;
        }
        workgroupBarrier();
    }
    if bin_tile_x + tile_x < config.width_in_tiles && bin_tile_y + tile_y < config.height_in_tiles {
        ptcl[cmd_offset] = CMD_END;
        if max_blend_depth > BLEND_STACK_SPLIT {
            let scratch_size = max_blend_depth * TILE_WIDTH * TILE_HEIGHT;
            ptcl[blend_offset] = atomicAdd(&bump.blend, scratch_size);
        }
    }
}
