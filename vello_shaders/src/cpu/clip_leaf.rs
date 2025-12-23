// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, PathBbox};

use super::CpuBinding;

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
            draw_monoids[clip_el.ix as usize].info_offset =
                draw_monoids[tos.parent_ix as usize].info_offset;
        }
    }
}

pub fn clip_leaf(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let clip_inp = resources[1].as_slice();
    let path_bboxes = resources[2].as_slice();
    let mut draw_monoids = resources[5].as_slice_mut();
    let mut clip_bboxes = resources[6].as_slice_mut();
    clip_leaf_main(
        &config,
        &clip_inp,
        &path_bboxes,
        &mut draw_monoids,
        &mut clip_bboxes,
    );
}
