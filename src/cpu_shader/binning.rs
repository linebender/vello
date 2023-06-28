// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{BinHeader, BumpAllocators, DrawMonoid, PathBbox, RenderConfig};

const WG_SIZE: usize = 256;
const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;
const N_TILE_X: usize = 16;
const N_TILE_Y: usize = 16;
const SX: f32 = 1.0 / ((N_TILE_X * TILE_WIDTH) as f32);
const SY: f32 = 1.0 / ((N_TILE_Y * TILE_HEIGHT) as f32);

fn bbox_intersect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

pub fn binning(
    config: &RenderConfig,
    draw_monoids: &[DrawMonoid],
    path_bbox_buf: &[PathBbox],
    clip_bbox_buf: &[[f32; 4]],
    intersected_bbox: &mut [[f32; 4]],
    info: &mut [u32],
    bump: &mut BumpAllocators,
    bin_data: &mut [u32],
    bin_header: &mut [BinHeader],
) {
    for wg in 0..config.workgroup_counts.binning.0 as usize {
        let mut counts = [0; WG_SIZE];
        let mut bboxes = [[0, 0, 0, 0]; WG_SIZE];
        let width_in_bins =
            ((config.gpu.width_in_tiles + N_TILE_X as u32 - 1) / N_TILE_X as u32) as i32;
        let height_in_bins =
            ((config.gpu.height_in_tiles + N_TILE_Y as u32 - 1) / N_TILE_Y as u32) as i32;
        for local_ix in 0..WG_SIZE {
            let element_ix = wg * WG_SIZE + local_ix;
            let mut x0 = 0;
            let mut y0 = 0;
            let mut x1 = 0;
            let mut y1 = 0;
            if element_ix < config.gpu.layout.n_draw_objects as usize {
                let draw_monoid = draw_monoids[element_ix];
                let mut clip_bbox = [-1e9, -1e9, 1e9, 1e9];
                if draw_monoid.clip_ix > 0 {
                    clip_bbox = clip_bbox_buf[draw_monoid.clip_ix as usize - 1];
                }
                let path_bbox = path_bbox_buf[draw_monoid.path_ix as usize];
                let pb = [
                    path_bbox.x0 as f32,
                    path_bbox.y0 as f32,
                    path_bbox.x1 as f32,
                    path_bbox.y1 as f32,
                ];
                let bbox_raw = bbox_intersect(clip_bbox, pb);
                let bbox = [
                    bbox_raw[0],
                    bbox_raw[1],
                    bbox_raw[0].max(bbox_raw[2]),
                    bbox_raw[1].max(bbox_raw[3]),
                ];
                intersected_bbox[element_ix] = bbox;
                x0 = (bbox[0] * SX).floor() as i32;
                x1 = (bbox[1] * SY).floor() as i32;
                y0 = (bbox[2] * SX).ceil() as i32;
                y1 = (bbox[3] * SY).ceil() as i32;
            }
            x0 = x0.clamp(0, width_in_bins);
            y0 = y0.clamp(0, height_in_bins);
            x1 = x1.clamp(0, width_in_bins);
            y1 = y1.clamp(0, height_in_bins);
            for y in y0..y1 {
                for x in x0..x1 {
                    counts[(y * width_in_bins + x) as usize] += 1;
                }
            }
            bboxes[local_ix] = [x0, y0, x1, y1];
        }
        let mut chunk_offset = [0; WG_SIZE];
        for local_ix in 0..WG_SIZE {
            let global_ix = wg * WG_SIZE + local_ix;
            chunk_offset[local_ix] = bump.binning;
            bump.binning += counts[local_ix];
            bin_header[global_ix] = BinHeader {
                element_count: counts[local_ix],
                chunk_offset: chunk_offset[local_ix],
            };
        }
        for local_ix in 0..WG_SIZE {
            let element_ix = wg * WG_SIZE + local_ix;
            let bbox = bboxes[local_ix];
            for y in bbox[2]..bbox[3] {
                for x in bbox[0]..bbox[1] {
                    let bin_ix = (y * width_in_bins + x) as usize;
                    let ix = config.gpu.layout.bin_data_start + chunk_offset[bin_ix];
                    bin_data[ix as usize] = element_ix as u32;
                    chunk_offset[bin_ix] += 1;
                }
            }
        }
    }
}
