// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use std::cmp::max;

use vello_encoding::{
    BinHeader, BumpAllocators, ConfigUniform, DRAW_INFO_FLAGS_FILL_RULE_BIT, DrawMonoid, DrawTag,
    Path, Tile,
};

use super::{
    CMD_BEGIN_CLIP, CMD_BLUR_RECT, CMD_COLOR, CMD_END, CMD_END_CLIP, CMD_FILL, CMD_IMAGE, CMD_JUMP,
    CMD_LIN_GRAD, CMD_RAD_GRAD, CMD_SOLID, CMD_SWEEP_GRAD, CpuBinding, PTCL_INITIAL_ALLOC,
};

// Tiles per bin
const N_TILE_X: usize = 16;
const N_TILE_Y: usize = 16;
const N_TILE: usize = N_TILE_X * N_TILE_Y;

// If changing also change in config.wgsl
const BLEND_STACK_SPLIT: u32 = 4;

// Pixels per tile
const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

const PTCL_INCREMENT: u32 = 256;
const PTCL_HEADROOM: u32 = 2;

// Modeled in the WGSL as private-scoped variables
struct TileState {
    cmd_offset: u32,
    cmd_limit: u32,
}

impl TileState {
    fn new(tile_ix: u32) -> Self {
        let cmd_offset = tile_ix * PTCL_INITIAL_ALLOC;
        let cmd_limit = cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM);
        Self {
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

    fn write_path(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        tile: &mut Tile,
        draw_flags: u32,
    ) {
        let n_segs = tile.segment_count_or_ix;
        if n_segs != 0 {
            let seg_ix = bump.segments;
            tile.segment_count_or_ix = !seg_ix;
            bump.segments += n_segs;
            self.alloc_cmd(4, config, bump, ptcl);
            self.write(ptcl, 0, CMD_FILL);
            let even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
            let size_and_rule = (n_segs << 1) | (even_odd as u32);
            self.write(ptcl, 1, size_and_rule);
            self.write(ptcl, 2, seg_ix);
            self.write(ptcl, 3, tile.backdrop as u32);
            self.cmd_offset += 4;
        } else {
            self.alloc_cmd(1, config, bump, ptcl);
            self.write(ptcl, 0, CMD_SOLID);
            self.cmd_offset += 1;
        }
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
        self.cmd_offset += 2;
    }

    fn write_image(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        info_offset: u32,
    ) {
        self.alloc_cmd(2, config, bump, ptcl);
        self.write(ptcl, 0, CMD_IMAGE);
        self.write(ptcl, 1, info_offset);
        self.cmd_offset += 2;
    }

    fn write_grad(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        ty: u32,
        index: u32,
        info_offset: u32,
    ) {
        self.alloc_cmd(3, config, bump, ptcl);
        self.write(ptcl, 0, ty);
        self.write(ptcl, 1, index);
        self.write(ptcl, 2, info_offset);
        self.cmd_offset += 3;
    }

    fn write_blur_rect(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        rgba_color: u32,
        info_offset: u32,
    ) {
        self.alloc_cmd(3, config, bump, ptcl);
        self.write(ptcl, 0, CMD_BLUR_RECT);
        self.write(ptcl, 1, info_offset);
        self.write(ptcl, 2, rgba_color);
        self.cmd_offset += 3;
    }

    fn write_begin_clip(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
    ) {
        self.alloc_cmd(1, config, bump, ptcl);
        self.write(ptcl, 0, CMD_BEGIN_CLIP);
        self.cmd_offset += 1;
    }

    fn write_end_clip(
        &mut self,
        config: &ConfigUniform,
        bump: &mut BumpAllocators,
        ptcl: &mut [u32],
        blend: u32,
        alpha: f32,
    ) {
        self.alloc_cmd(3, config, bump, ptcl);
        self.write(ptcl, 0, CMD_END_CLIP);
        self.write(ptcl, 1, blend);
        self.write(ptcl, 2, f32::to_bits(alpha));
        self.cmd_offset += 3;
    }
}

fn coarse_main(
    config: &ConfigUniform,
    scene: &[u32],
    draw_monoids: &[DrawMonoid],
    bin_headers: &[BinHeader],
    info_bin_data: &[u32],
    paths: &[Path],
    tiles: &mut [Tile],
    bump: &mut BumpAllocators,
    ptcl: &mut [u32],
) {
    let width_in_tiles = config.width_in_tiles;
    let height_in_tiles = config.height_in_tiles;
    let width_in_bins = width_in_tiles.div_ceil(N_TILE_X as u32);
    let height_in_bins = height_in_tiles.div_ceil(N_TILE_Y as u32);
    let n_bins = width_in_bins * height_in_bins;
    let bin_data_start = config.layout.bin_data_start;
    let drawtag_base = config.layout.draw_tag_base;
    let mut compacted = vec![vec![]; N_TILE];
    let n_partitions = config.layout.n_draw_objects.div_ceil(N_TILE as u32);
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
            let mut clip_depth = 0;
            let mut render_blend_depth = 0;
            let mut max_blend_depth = 0_u32;
            let mut clip_zero_depth = 0;
            for drawobj_ix in &compacted[tile_ix] {
                let drawtag = scene[(drawtag_base + drawobj_ix) as usize];
                if clip_zero_depth == 0 {
                    let draw_monoid = draw_monoids[*drawobj_ix as usize];
                    let path_ix = draw_monoid.path_ix;
                    let path = paths[path_ix as usize];
                    let bbox = path.bbox;
                    let stride = bbox[2] - bbox[0];
                    let x = bin_tile_x + tile_x - bbox[0];
                    let y = bin_tile_y + tile_y - bbox[1];
                    let tile = &mut tiles[(path.tiles + y * stride + x) as usize];
                    let is_clip = (drawtag & 1) != 0;
                    let mut is_blend = false;
                    let dd = config.layout.draw_data_base + draw_monoid.scene_offset;
                    let di = draw_monoid.info_offset;
                    if is_clip {
                        const BLEND_CLIP: u32 = (128 << 8) | 3;
                        let blend = scene[dd as usize];
                        is_blend = blend != BLEND_CLIP;
                    }

                    let draw_flags = info_bin_data[di as usize];
                    let even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
                    let n_segs = tile.segment_count_or_ix;

                    // If this draw object represents an even-odd fill and we know that no line segment
                    // crosses this tile and then this draw object should not contribute to the tile if its
                    // backdrop (i.e. the winding number of its top-left corner) is even.
                    let backdrop_clear = if even_odd {
                        tile.backdrop.abs() & 1
                    } else {
                        tile.backdrop
                    } == 0;
                    let include_tile = n_segs != 0 || (backdrop_clear == is_clip) || is_blend;
                    if include_tile {
                        match DrawTag(drawtag) {
                            DrawTag::COLOR => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let rgba_color = scene[dd as usize];
                                tile_state.write_color(config, bump, ptcl, rgba_color);
                            }
                            DrawTag::IMAGE => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                tile_state.write_image(config, bump, ptcl, di + 1);
                            }
                            DrawTag::LINEAR_GRADIENT => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let index = scene[dd as usize];
                                tile_state.write_grad(
                                    config,
                                    bump,
                                    ptcl,
                                    CMD_LIN_GRAD,
                                    index,
                                    di + 1,
                                );
                            }
                            DrawTag::RADIAL_GRADIENT => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let index = scene[dd as usize];
                                tile_state.write_grad(
                                    config,
                                    bump,
                                    ptcl,
                                    CMD_RAD_GRAD,
                                    index,
                                    di + 1,
                                );
                            }
                            DrawTag::SWEEP_GRADIENT => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let index = scene[dd as usize];
                                tile_state.write_grad(
                                    config,
                                    bump,
                                    ptcl,
                                    CMD_SWEEP_GRAD,
                                    index,
                                    di + 1,
                                );
                            }
                            DrawTag::BLUR_RECT => {
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let rgba_color = scene[dd as usize];
                                tile_state.write_blur_rect(config, bump, ptcl, rgba_color, di + 1);
                            }
                            DrawTag::BEGIN_CLIP => {
                                let even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
                                let backdrop_clear = if even_odd {
                                    tile.backdrop.abs() & 1 == 0
                                } else {
                                    tile.backdrop == 0
                                };
                                if tile.segment_count_or_ix == 0 && backdrop_clear {
                                    clip_zero_depth = clip_depth + 1;
                                } else {
                                    tile_state.write_begin_clip(config, bump, ptcl);
                                    // TODO: Do we need to track this separately, seems like it
                                    // is always the same as clip_depth in this code path
                                    render_blend_depth += 1;
                                    max_blend_depth = max(render_blend_depth, max_blend_depth);
                                }
                                clip_depth += 1;
                            }
                            DrawTag::END_CLIP => {
                                clip_depth -= 1;
                                tile_state.write_path(config, bump, ptcl, tile, draw_flags);
                                let blend = scene[dd as usize];
                                let alpha = f32::from_bits(scene[dd as usize + 1]);
                                tile_state.write_end_clip(config, bump, ptcl, blend, alpha);
                                render_blend_depth -= 1;
                            }
                            _ => todo!(),
                        }
                    }
                } else {
                    // In "clip zero" state, suppress all drawing
                    match DrawTag(drawtag) {
                        DrawTag::BEGIN_CLIP => clip_depth += 1,
                        DrawTag::END_CLIP => {
                            if clip_depth == clip_zero_depth {
                                clip_zero_depth = 0;
                            }
                            clip_depth -= 1;
                        }
                        _ => (),
                    }
                }
            }

            if bin_tile_x + tile_x < width_in_tiles && bin_tile_y + tile_y < height_in_tiles {
                ptcl[tile_state.cmd_offset as usize] = CMD_END;
                let scratch_size =
                    (max_blend_depth.saturating_sub(BLEND_STACK_SPLIT)) * TILE_WIDTH * TILE_HEIGHT;
                ptcl[blend_offset as usize] = bump.blend;
                bump.blend += scratch_size;
            }
        }
    }
}

pub fn coarse(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let draw_monoids = resources[2].as_slice();
    let bin_headers = resources[3].as_slice();
    let info_bin_data = resources[4].as_slice();
    let paths = resources[5].as_slice();
    let mut tiles = resources[6].as_slice_mut();
    let mut bump = resources[7].as_typed_mut();
    let mut ptcl = resources[8].as_slice_mut();
    coarse_main(
        &config,
        &scene,
        &draw_monoids,
        &bin_headers,
        &info_bin_data,
        &paths,
        &mut tiles,
        &mut bump,
        &mut ptcl,
    );
}
