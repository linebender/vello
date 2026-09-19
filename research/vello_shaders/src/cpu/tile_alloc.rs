// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{BumpAllocators, ConfigUniform, DrawTag, Path, Tile};

use super::CpuBinding;

const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;
const SX: f32 = 1.0 / (TILE_WIDTH as f32);
const SY: f32 = 1.0 / (TILE_HEIGHT as f32);

fn tile_alloc_main(
    config: &ConfigUniform,
    scene: &[u32],
    draw_bboxes: &[[f32; 4]],
    bump: &mut BumpAllocators,
    paths: &mut [Path],
    tiles: &mut [Tile],
) {
    let drawtag_base = config.layout.draw_tag_base;
    let width_in_tiles = config.width_in_tiles as i32;
    let height_in_tiles = config.height_in_tiles as i32;
    for drawobj_ix in 0..config.layout.n_draw_objects {
        let drawtag = DrawTag(scene[(drawtag_base + drawobj_ix) as usize]);
        let mut x0 = 0;
        let mut y0 = 0;
        let mut x1 = 0;
        let mut y1 = 0;
        if drawtag != DrawTag::NOP && drawtag != DrawTag::END_CLIP {
            let bbox = draw_bboxes[drawobj_ix as usize];
            if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
                x0 = (bbox[0] * SX).floor() as i32;
                y0 = (bbox[1] * SY).floor() as i32;
                x1 = (bbox[2] * SX).ceil() as i32;
                y1 = (bbox[3] * SY).ceil() as i32;
            }
        }
        let ux0 = x0.clamp(0, width_in_tiles) as u32;
        let uy0 = y0.clamp(0, height_in_tiles) as u32;
        let ux1 = x1.clamp(0, width_in_tiles) as u32;
        let uy1 = y1.clamp(0, height_in_tiles) as u32;
        let tile_count = (ux1 - ux0) * (uy1 - uy0);
        let offset = bump.tile;
        bump.tile += tile_count;
        // We construct it this way because padding is private.
        let mut path = Path::default();
        path.bbox = [ux0, uy0, ux1, uy1];
        path.tiles = offset;
        paths[drawobj_ix as usize] = path;
        for i in 0..tile_count {
            tiles[(offset + i) as usize] = Tile::default();
        }
    }
}

pub fn tile_alloc(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let draw_bboxes = resources[2].as_slice();
    let mut bump = resources[3].as_typed_mut();
    let mut paths = resources[4].as_slice_mut();
    let mut tiles = resources[5].as_slice_mut();
    tile_alloc_main(
        &config,
        &scene,
        &draw_bboxes,
        &mut bump,
        &mut paths,
        &mut tiles,
    );
}
