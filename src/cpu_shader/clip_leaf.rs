// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, PathBbox};

use crate::cpu_dispatch::CpuBinding;

struct ClipStackElement {
    // index of draw object
    parent_ix: u32,
    path_ix: u32,
    bbox: [f32; 4],
}

const BIG_BBOX: [f32; 4] = [-1e9, -1e9, 1e9, 1e9];

// Note: this implementation doesn't rigorously follow the
// WGSL original. In particular, it just computes the clips
// sequentially rather than using the partition reductions.
fn clip_leaf_main(
    config: &ConfigUniform,
    clip_inp: &[Clip],
    path_bboxes: &[PathBbox],
    draw_monoids: &mut [DrawMonoid],
    clip_bboxes: &mut [[f32; 4]],
) {
    let mut stack: Vec<ClipStackElement> = Vec::new();
    for global_ix in 0..config.layout.n_clips {
        let clip_el = clip_inp[global_ix as usize];
        if clip_el.path_ix >= 0 {
            // begin clip
            let path_ix = clip_el.path_ix as u32;
            let path_bbox = path_bboxes[path_ix as usize];
            let p_bbox = [
                path_bbox.x0 as f32,
                path_bbox.y0 as f32,
                path_bbox.x1 as f32,
                path_bbox.y1 as f32,
            ];
            let bbox = if let Some(last) = stack.last() {
                [
                    p_bbox[0].max(last.bbox[0]),
                    p_bbox[1].max(last.bbox[1]),
                    p_bbox[2].min(last.bbox[2]),
                    p_bbox[3].min(last.bbox[3]),
                ]
            } else {
                p_bbox
            };
            clip_bboxes[global_ix as usize] = bbox;
            let parent_ix = clip_el.ix;
            stack.push(ClipStackElement {
                parent_ix,
                path_ix,
                bbox,
            });
        } else {
            // end clip
            let tos = stack.pop().unwrap();
            let bbox = if let Some(nos) = stack.last() {
                nos.bbox
            } else {
                BIG_BBOX
            };
            clip_bboxes[global_ix as usize] = bbox;
            draw_monoids[clip_el.ix as usize].path_ix = tos.path_ix;
            draw_monoids[clip_el.ix as usize].scene_offset =
                draw_monoids[tos.parent_ix as usize].scene_offset;
        }
    }
}

pub fn clip_leaf(_n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let r1 = resources[1].as_buf();
    let r2 = resources[2].as_buf();
    let mut r5 = resources[5].as_buf();
    let mut r6 = resources[6].as_buf();
    let config = bytemuck::from_bytes(&r0);
    let clip_inp = bytemuck::cast_slice(&r1);
    let path_bboxes = bytemuck::cast_slice(&r2);
    let draw_monoids = bytemuck::cast_slice_mut(r5.as_mut());
    let clip_bboxes = bytemuck::cast_slice_mut(r6.as_mut());
    clip_leaf_main(config, clip_inp, path_bboxes, draw_monoids, clip_bboxes);
}
