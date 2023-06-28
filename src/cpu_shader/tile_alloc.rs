// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{BumpAllocators, DrawTag, Path, RenderConfig, Tile};

const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;
const SX: f32 = 1.0 / (TILE_WIDTH as f32);
const SY: f32 = 1.0 / (TILE_HEIGHT as f32);

pub fn tile_alloc(
    config: &RenderConfig,
    scene: &[u32],
    draw_bboxes: &[[f32; 4]],
    bump: &mut BumpAllocators,
    paths: &mut [Path],
    tiles: &mut [Tile],
) {
    let drawtag_base = config.gpu.layout.draw_tag_base;
    let width_in_tiles = config.gpu.width_in_tiles as i32;
    let height_in_tiles = config.gpu.height_in_tiles as i32;
    for drawobj_ix in 0..config.gpu.layout.n_draw_objects {
        let drawtag = DrawTag(scene[(drawtag_base + drawobj_ix) as usize]);
        let mut x0 = 0;
        let mut y0 = 0;
        let mut x1 = 0;
        let mut y1 = 0;
        if drawtag != DrawTag::NOP && drawtag != DrawTag::END_CLIP {
            let bbox = draw_bboxes[drawobj_ix as usize];
            x0 = (bbox[0] * SX).floor() as i32;
            y0 = (bbox[1] * SY).floor() as i32;
            x1 = (bbox[2] * SX).ceil() as i32;
            y1 = (bbox[3] * SY).ceil() as i32;
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
