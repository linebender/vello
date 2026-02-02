// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU coarse stage regression tests.

use std::cell::RefCell;

use bytemuck::{bytes_of, cast_slice};
use vello_encoding::{
    BinHeader, BumpAllocators, ConfigUniform, DrawMonoid, DrawTag, Layout, Path, Tile,
};
use vello_shaders::cpu::{CpuBinding, coarse};

fn count_cmd(ptcl_u32: &[u32], cmd: u32) -> usize {
    ptcl_u32.iter().filter(|&&x| x == cmd).count()
}

#[test]
fn coarse_emits_set_composite_before_color() {
    // Multiply + SrcOver, matching `vello_shaders/shader/shared/blend.wgsl`.
    let blend_mode = (1_u32 << 8) | 3_u32;
    let composite = vello_encoding::pack_style_composite(blend_mode, 0.5);
    assert_eq!(composite & 1, 0, "style composite must keep bit 0 clear");

    let layout = Layout {
        n_draw_objects: 1,
        n_paths: 1,
        n_clips: 0,
        bin_data_start: 1,
        path_tag_base: 0,
        path_data_base: 0,
        draw_tag_base: 0,
        draw_data_base: 1,
        transform_base: 0,
        style_base: 0,
    };
    let config = ConfigUniform {
        width_in_tiles: 1,
        height_in_tiles: 1,
        target_width: 16,
        target_height: 16,
        base_color: 0,
        layout,
        lines_size: 0,
        binning_size: 0,
        tiles_size: 1,
        seg_counts_size: 0,
        segments_size: 0,
        blend_size: 0,
        ptcl_size: 64,
    };

    let scene: Vec<u32> = vec![
        DrawTag::COLOR.0, // draw_tag_base + 0
        0xFF00_00FF,      // draw_data_base + 0: opaque red in unpack4x8unorm order
    ];
    let draw_monoids = vec![DrawMonoid {
        path_ix: 0,
        clip_ix: 0,
        scene_offset: 0,
        info_offset: 0,
    }];
    let bin_headers = vec![BinHeader {
        element_count: 1,
        chunk_offset: 0,
    }];
    // info[0] = draw_flags, bin_data_start=1 so info[1] holds drawobj_ix = 0
    let info_bin_data: Vec<u32> = vec![composite, 0];
    let mut path = Path::default();
    path.bbox = [0, 0, 1, 1];
    path.tiles = 0;
    let paths = vec![path];

    let tiles_bytes = RefCell::new(
        cast_slice::<Tile, u8>(&[Tile {
            backdrop: 1,
            segment_count_or_ix: 0,
        }])
        .to_vec(),
    );
    let bump_bytes = RefCell::new(bytes_of(&BumpAllocators::default()).to_vec());
    let ptcl_bytes = RefCell::new(cast_slice::<u32, u8>(&vec![0_u32; 64]).to_vec());

    let resources: [CpuBinding<'_>; 9] = [
        CpuBinding::Buffer(bytes_of(&config)),
        CpuBinding::Buffer(cast_slice(&scene)),
        CpuBinding::Buffer(cast_slice(&draw_monoids)),
        CpuBinding::Buffer(cast_slice(&bin_headers)),
        CpuBinding::Buffer(cast_slice(&info_bin_data)),
        CpuBinding::Buffer(cast_slice(&paths)),
        CpuBinding::BufferRW(&tiles_bytes),
        CpuBinding::BufferRW(&bump_bytes),
        CpuBinding::BufferRW(&ptcl_bytes),
    ];

    coarse(0, &resources);

    let ptcl_u32: Vec<u32> = {
        let bytes = ptcl_bytes.borrow();
        cast_slice::<u8, u32>(&bytes).to_vec()
    };

    // ptcl[0] reserved for blend spill offset (written at end of coarse)
    assert_eq!(ptcl_u32[1], 3, "CMD_SOLID");
    assert_eq!(ptcl_u32[2], 14, "CMD_SET_COMPOSITE");
    assert_eq!(ptcl_u32[3], composite, "composite payload");
    assert_eq!(ptcl_u32[4], 5, "CMD_COLOR");
    assert_eq!(ptcl_u32[6], 0, "CMD_END");
}

#[test]
fn coarse_dedupes_set_composite_when_unchanged() {
    // Multiply + SrcOver, matching `vello_shaders/shader/shared/blend.wgsl`.
    let blend_mode = (1_u32 << 8) | 3_u32;
    let composite = vello_encoding::pack_style_composite(blend_mode, 0.5);

    let layout = Layout {
        n_draw_objects: 2,
        n_paths: 2,
        n_clips: 0,
        bin_data_start: 2,
        path_tag_base: 0,
        path_data_base: 0,
        draw_tag_base: 0,
        draw_data_base: 2,
        transform_base: 0,
        style_base: 0,
    };
    let config = ConfigUniform {
        width_in_tiles: 1,
        height_in_tiles: 1,
        target_width: 16,
        target_height: 16,
        base_color: 0,
        layout,
        lines_size: 0,
        binning_size: 0,
        tiles_size: 1,
        seg_counts_size: 0,
        segments_size: 0,
        blend_size: 0,
        ptcl_size: 64,
    };

    let scene: Vec<u32> = vec![
        DrawTag::COLOR.0, // draw_tag_base + 0
        DrawTag::COLOR.0, // draw_tag_base + 1
        0xFF00_00FF,      // draw_data_base + 0
        0xFFFF_00FF,      // draw_data_base + 1
    ];
    let draw_monoids = vec![
        DrawMonoid {
            path_ix: 0,
            clip_ix: 0,
            scene_offset: 0,
            info_offset: 0,
        },
        DrawMonoid {
            path_ix: 1,
            clip_ix: 0,
            scene_offset: 1,
            info_offset: 1,
        },
    ];
    let bin_headers = vec![BinHeader {
        element_count: 2,
        chunk_offset: 0,
    }];
    // info[0..2]=draw_flags, bin_data_start=2 so info[2..] holds drawobj_ix
    let info_bin_data: Vec<u32> = vec![composite, composite, 0, 1];
    let mut path0 = Path::default();
    path0.bbox = [0, 0, 1, 1];
    path0.tiles = 0;
    let mut path1 = path0;
    path1.tiles = 0;
    let paths = vec![path0, path1];

    let tiles_bytes = RefCell::new(
        cast_slice::<Tile, u8>(&[Tile {
            backdrop: 1,
            segment_count_or_ix: 0,
        }])
        .to_vec(),
    );
    let bump_bytes = RefCell::new(bytes_of(&BumpAllocators::default()).to_vec());
    let ptcl_bytes = RefCell::new(cast_slice::<u32, u8>(&vec![0_u32; 64]).to_vec());

    let resources: [CpuBinding<'_>; 9] = [
        CpuBinding::Buffer(bytes_of(&config)),
        CpuBinding::Buffer(cast_slice(&scene)),
        CpuBinding::Buffer(cast_slice(&draw_monoids)),
        CpuBinding::Buffer(cast_slice(&bin_headers)),
        CpuBinding::Buffer(cast_slice(&info_bin_data)),
        CpuBinding::Buffer(cast_slice(&paths)),
        CpuBinding::BufferRW(&tiles_bytes),
        CpuBinding::BufferRW(&bump_bytes),
        CpuBinding::BufferRW(&ptcl_bytes),
    ];

    coarse(0, &resources);

    let ptcl_u32: Vec<u32> = {
        let bytes = ptcl_bytes.borrow();
        cast_slice::<u8, u32>(&bytes).to_vec()
    };

    // Only one set-composite is needed for both draws.
    assert_eq!(count_cmd(&ptcl_u32, 14), 1, "CMD_SET_COMPOSITE count");

    // Expect: SOLID, SET_COMPOSITE, COLOR, SOLID, COLOR, END.
    assert_eq!(ptcl_u32[1], 3, "CMD_SOLID");
    assert_eq!(ptcl_u32[2], 14, "CMD_SET_COMPOSITE");
    assert_eq!(ptcl_u32[4], 5, "CMD_COLOR");
    assert_eq!(ptcl_u32[6], 3, "CMD_SOLID");
    assert_eq!(ptcl_u32[7], 5, "CMD_COLOR");
    assert_eq!(ptcl_u32[9], 0, "CMD_END");
}

#[test]
fn coarse_emits_set_composite_when_it_changes() {
    let multiply = vello_encoding::pack_style_composite((1_u32 << 8) | 3_u32, 1.0);
    let screen = vello_encoding::pack_style_composite((2_u32 << 8) | 3_u32, 1.0);

    let layout = Layout {
        n_draw_objects: 2,
        n_paths: 2,
        n_clips: 0,
        bin_data_start: 2,
        path_tag_base: 0,
        path_data_base: 0,
        draw_tag_base: 0,
        draw_data_base: 2,
        transform_base: 0,
        style_base: 0,
    };
    let config = ConfigUniform {
        width_in_tiles: 1,
        height_in_tiles: 1,
        target_width: 16,
        target_height: 16,
        base_color: 0,
        layout,
        lines_size: 0,
        binning_size: 0,
        tiles_size: 1,
        seg_counts_size: 0,
        segments_size: 0,
        blend_size: 0,
        ptcl_size: 64,
    };

    let scene: Vec<u32> = vec![
        DrawTag::COLOR.0, // draw_tag_base + 0
        DrawTag::COLOR.0, // draw_tag_base + 1
        0xFF00_00FF,      // draw_data_base + 0
        0xFFFF_00FF,      // draw_data_base + 1
    ];
    let draw_monoids = vec![
        DrawMonoid {
            path_ix: 0,
            clip_ix: 0,
            scene_offset: 0,
            info_offset: 0,
        },
        DrawMonoid {
            path_ix: 1,
            clip_ix: 0,
            scene_offset: 1,
            info_offset: 1,
        },
    ];
    let bin_headers = vec![BinHeader {
        element_count: 2,
        chunk_offset: 0,
    }];
    let info_bin_data: Vec<u32> = vec![multiply, screen, 0, 1];
    let mut path0 = Path::default();
    path0.bbox = [0, 0, 1, 1];
    path0.tiles = 0;
    let mut path1 = path0;
    path1.tiles = 0;
    let paths = vec![path0, path1];

    let tiles_bytes = RefCell::new(
        cast_slice::<Tile, u8>(&[Tile {
            backdrop: 1,
            segment_count_or_ix: 0,
        }])
        .to_vec(),
    );
    let bump_bytes = RefCell::new(bytes_of(&BumpAllocators::default()).to_vec());
    let ptcl_bytes = RefCell::new(cast_slice::<u32, u8>(&vec![0_u32; 64]).to_vec());

    let resources: [CpuBinding<'_>; 9] = [
        CpuBinding::Buffer(bytes_of(&config)),
        CpuBinding::Buffer(cast_slice(&scene)),
        CpuBinding::Buffer(cast_slice(&draw_monoids)),
        CpuBinding::Buffer(cast_slice(&bin_headers)),
        CpuBinding::Buffer(cast_slice(&info_bin_data)),
        CpuBinding::Buffer(cast_slice(&paths)),
        CpuBinding::BufferRW(&tiles_bytes),
        CpuBinding::BufferRW(&bump_bytes),
        CpuBinding::BufferRW(&ptcl_bytes),
    ];

    coarse(0, &resources);

    let ptcl_u32: Vec<u32> = {
        let bytes = ptcl_bytes.borrow();
        cast_slice::<u8, u32>(&bytes).to_vec()
    };

    assert_eq!(count_cmd(&ptcl_u32, 14), 2, "CMD_SET_COMPOSITE count");

    // Expect: SOLID, SET(multiply), COLOR, SOLID, SET(screen), COLOR, END.
    assert_eq!(ptcl_u32[1], 3, "CMD_SOLID");
    assert_eq!(ptcl_u32[2], 14, "CMD_SET_COMPOSITE");
    assert_eq!(ptcl_u32[3], multiply, "multiply payload");
    assert_eq!(ptcl_u32[4], 5, "CMD_COLOR");
    assert_eq!(ptcl_u32[6], 3, "CMD_SOLID");
    assert_eq!(ptcl_u32[7], 14, "CMD_SET_COMPOSITE");
    assert_eq!(ptcl_u32[8], screen, "screen payload");
    assert_eq!(ptcl_u32[9], 5, "CMD_COLOR");
    assert_eq!(ptcl_u32[11], 0, "CMD_END");
}
