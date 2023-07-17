// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{
    BinHeader, BumpAllocators, ConfigUniform, DrawMonoid, DrawTag, Path, RenderConfig, Tile,
};

use crate::cpu_dispatch::CpuBinding;

use super::{CMD_COLOR, CMD_END, CMD_FILL, CMD_JUMP, CMD_SOLID, PTCL_INITIAL_ALLOC};

const N_TILE_X: usize = 16;
const N_TILE_Y: usize = 16;
const N_TILE: usize = N_TILE_X * N_TILE_Y;

const PTCL_INCREMENT: u32 = 256;
const PTCL_HEADROOM: u32 = 2;

// Modeled in the WGSL as private-scoped variables
struct TileState {
    cmd_offset: u32,
    cmd_limit: u32,
}

impl TileState {
    fn new(tile_ix: u32) -> TileState {
        let cmd_offset = tile_ix * PTCL_INITIAL_ALLOC;
        let cmd_limit = cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM);
        TileState {
            cmd_offset,
            cmd_limit,
        }
    }

    fn alloc_cmd(
        &mut self,
        size: u32,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
    ) {
        if self.cmd_offset + size >= self.cmd_limit {
            let ptcl_dyn_start =
                config.width_in_tiles * config.height_in_tiles * PTCL_INITIAL_ALLOC;
            let chunk_size = PTCL_INCREMENT.max(size + PTCL_HEADROOM);
            let new_cmd = ptcl_dyn_start + bump.ptcl;
            bump.ptcl += chunk_size;
            ptcl[self.cmd_offset as usize] = CMD_JUMP;
            ptcl[self.cmd_offset as usize + 1] = new_cmd;
            self.cmd_offset = new_cmd;
            self.cmd_limit = new_cmd + (PTCL_INCREMENT - PTCL_HEADROOM);
        }
    }

    fn write(&mut self, ptcl: &mut [u32], offset: u32, value: u32) {
        ptcl[(self.cmd_offset + offset) as usize] = value;
    }

    fn write_color(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        rgba_color: u32,
    ) {
        self.alloc_cmd(2, config, bump, ptcl);
        self.write(ptcl, 0, CMD_COLOR);
        self.write(ptcl, 1, rgba_color);
    }
}

fn coarse_main(
    config: &ConfigUniform,
    scene: &[u32],
    draw_monoids: &[DrawMonoid],
    bin_headers: &[BinHeader],
    info_bin_data: &[u32],
    paths: &[Path],
    tiles: &[Tile],
    bump: &mut BumpAllocators,
    ptcl: &mut [u32],
) {
    let width_in_tiles = config.width_in_tiles;
    let height_in_tiles = config.height_in_tiles;
    let width_in_bins = (width_in_tiles + N_TILE_X as u32 - 1) / N_TILE_X as u32;
    let height_in_bins = (height_in_tiles + N_TILE_Y as u32 - 1) / N_TILE_Y as u32;
    let n_bins = width_in_bins * height_in_bins;
    let bin_data_start = config.layout.bin_data_start;
    let drawtag_base = config.layout.draw_tag_base;
    let mut compacted = vec![vec![]; N_TILE];
    let n_partitions = (config.layout.n_draw_objects + N_TILE as u32 - 1) / N_TILE as u32;
    for bin in 0..n_bins {
        for v in &mut compacted {
            v.clear();
        }
        let bin_x = bin % width_in_bins;
        let bin_y = bin / width_in_bins;
        let bin_tile_x = N_TILE_X as u32 * bin_x;
        let bin_tile_y = N_TILE_Y as u32 * bin_y;
        for part in 0..n_partitions {
            let in_ix = part * N_TILE as u32 + bin;
            let bin_header = bin_headers[in_ix as usize];
            let start = bin_data_start + bin_header.chunk_offset;
            for i in 0..bin_header.element_count {
                let drawobj_ix = info_bin_data[(start + i) as usize];
                let tag = scene[(drawtag_base + drawobj_ix) as usize];
                if DrawTag(tag) != DrawTag::NOP {
                    let draw_monoid = draw_monoids[drawobj_ix as usize];
                    let path_ix = draw_monoid.path_ix;
                    let path = paths[path_ix as usize];
                    let dx = path.bbox[0] as i32 - bin_tile_x as i32;
                    let dy = path.bbox[1] as i32 - bin_tile_y as i32;
                    let x0 = dx.clamp(0, N_TILE_X as i32);
                    let y0 = dy.clamp(0, N_TILE_Y as i32);
                    let x1 = (path.bbox[2] as i32 - bin_tile_x as i32).clamp(0, N_TILE_X as i32);
                    let y1 = (path.bbox[3] as i32 - bin_tile_y as i32).clamp(0, N_TILE_Y as i32);
                    for y in y0..y1 {
                        for x in x0..x1 {
                            compacted[(y * N_TILE_X as i32 + x) as usize].push(drawobj_ix);
                        }
                    }
                }
            }
        }
        // compacted now has the list of draw objects for each tile.
        // While the WGSL source does at most 256 draw objects at a time,
        // this version does all the draw objects in a tile.
        for tile_ix in 0..N_TILE {
            let tile_x = (tile_ix % N_TILE_X) as u32;
            let tile_y = (tile_ix / N_TILE_X) as u32;
            let this_tile_ix = (bin_tile_y + tile_y) * width_in_tiles + bin_tile_x + tile_x;
            let mut tile_state = TileState::new(this_tile_ix);
            let blend_offset = tile_state.cmd_offset;
            tile_state.cmd_offset += 1;
            for drawobj_ix in &compacted[tile_ix] {
                let drawtag = scene[(drawtag_base + drawobj_ix) as usize];
                let draw_monoid = draw_monoids[*drawobj_ix as usize];
                let path_ix = draw_monoid.path_ix;
                let path = paths[path_ix as usize];
                let bbox = path.bbox;
                let stride = bbox[2] - bbox[0];
                let x = bin_tile_x + tile_x - bbox[0];
                let y = bin_tile_y + tile_y - bbox[1];
                let tile = tiles[(path.tiles + y * stride + x) as usize];
                // TODO: clip-related logic
                let n_segs = tile.segments;
                let include_tile = n_segs != 0 || tile.backdrop != 0;
                if include_tile {
                    let dd = config.layout.draw_data_base + draw_monoid.scene_offset;
                    // TODO: get drawinfo (linewidth for fills)
                    match DrawTag(drawtag) {
                        DrawTag::COLOR => {
                            if n_segs != 0 {
                                let seg_ix = bump.segments;
                                bump.segments += n_segs;
                                tile_state.alloc_cmd(4, config, bump, ptcl);
                                tile_state.write(ptcl, 0, CMD_FILL);
                                let even_odd = false; // TODO
                                let size_and_rule = (n_segs << 1) | (even_odd as u32);
                                tile_state.write(ptcl, 1, size_and_rule);
                                tile_state.write(ptcl, 2, seg_ix);
                                tile_state.write(ptcl, 3, tile.backdrop as u32);
                            } else {
                                tile_state.alloc_cmd(1, config, bump, ptcl);
                                tile_state.write(ptcl, 0, CMD_SOLID);
                            }
                            let rgba_color = scene[dd as usize];
                            tile_state.write_color(config, bump, ptcl, rgba_color);
                        }
                        _ => todo!(),
                    }
                }
            }

            if bin_tile_x + tile_x < width_in_tiles && bin_tile_y + tile_y < height_in_tiles {
                ptcl[tile_state.cmd_offset as usize] = CMD_END;
                let scratch_size = 0; // TODO: actually compute
                ptcl[blend_offset as usize] = bump.blend;
                bump.blend += scratch_size;
            }
        }
    }
}

pub fn coarse(n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let r1 = resources[1].as_buf();
    let r2 = resources[2].as_buf();
    let r3 = resources[3].as_buf();
    let r4 = resources[4].as_buf();
    let r5 = resources[5].as_buf();
    let r6 = resources[6].as_buf();
    let mut r7 = resources[7].as_buf();
    let mut r8 = resources[8].as_buf();
    let config = bytemuck::from_bytes(&r0);
    let scene = bytemuck::cast_slice(&r1);
    let draw_monoids = bytemuck::cast_slice(&r2);
    let bin_headers = bytemuck::cast_slice(&r3);
    let info_bin_data = bytemuck::cast_slice(&r4);
    let paths = bytemuck::cast_slice(&r5);
    let tiles = bytemuck::cast_slice(&r6);
    let bump = bytemuck::from_bytes_mut(r7.as_mut());
    let ptcl = bytemuck::cast_slice_mut(r8.as_mut());
    coarse_main(
        config,
        scene,
        draw_monoids,
        bin_headers,
        info_bin_data,
        paths,
        tiles,
        bump,
        ptcl,
    );
}
