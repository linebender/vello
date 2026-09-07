// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{Clip, ClipBic, ClipElement, PathBbox};

use super::CpuBinding;

const WG_SIZE: usize = 256;

fn clip_reduce_main(
    n_wg: u32,
    clip_inp: &[Clip],
    path_bboxes: &[PathBbox],
    reduced: &mut [ClipBic],
    clip_out: &mut [ClipElement],
) {
    let mut scratch = Vec::with_capacity(WG_SIZE);
    for wg_ix in 0..n_wg {
        scratch.clear();
        let mut bic_reduced = ClipBic::default();
        // reverse scan
        for local_ix in (0..WG_SIZE).rev() {
            let global_ix = wg_ix as usize * WG_SIZE + local_ix;
            let inp = clip_inp[global_ix].path_ix;
            let is_push = inp >= 0;
            let bic = ClipBic::new(1 - is_push as u32, is_push as u32);
            if is_push && bic_reduced.a == 0 {
                scratch.push(global_ix as u32);
            }
            bic_reduced = bic.combine(bic_reduced);
        }
        reduced[wg_ix as usize] = bic_reduced;
        for (i, parent_ix) in scratch.iter().rev().enumerate() {
            let mut clip_el = ClipElement::default();
            clip_el.parent_ix = *parent_ix;
            let path_ix = clip_inp[*parent_ix as usize].path_ix;
            let path_bbox = path_bboxes[path_ix as usize];
            clip_el.bbox = [
                path_bbox.x0 as f32,
                path_bbox.y0 as f32,
                path_bbox.x1 as f32,
                path_bbox.y1 as f32,
            ];
            let global_ix = wg_ix as usize * WG_SIZE + i;
            clip_out[global_ix] = clip_el;
        }
    }
}

pub fn clip_reduce(n_wg: u32, resources: &[CpuBinding<'_>]) {
    let clip_inp = resources[0].as_slice();
    let path_bboxes = resources[1].as_slice();
    let mut reduced = resources[2].as_slice_mut();
    let mut clip_out = resources[3].as_slice_mut();
    clip_reduce_main(n_wg, &clip_inp, &path_bboxes, &mut reduced, &mut clip_out);
}
