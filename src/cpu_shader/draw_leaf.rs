// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, DrawTag, Monoid, PathBbox, RenderConfig};

use crate::cpu_dispatch::CpuBinding;

use super::util::Transform;

const WG_SIZE: usize = 256;

fn draw_leaf_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    reduced: &[DrawMonoid],
    path_bbox: &[PathBbox],
    draw_monoid: &mut [DrawMonoid],
    info: &mut [u32],
    clip_inp: &mut [Clip],
) {
    let drawtag_base = config.layout.draw_tag_base;
    let mut prefix = DrawMonoid::default();
    for i in 0..n_wg {
        let mut m = prefix;
        for j in 0..WG_SIZE {
            let ix = i * WG_SIZE as u32 + j as u32;
            let tag_raw = if ix < config.layout.n_draw_objects {
                scene[(drawtag_base + ix) as usize]
            } else {
                0
            };
            let tag_word = DrawTag(tag_raw);
            // store exclusive prefix sum
            if ix < config.layout.n_draw_objects {
                draw_monoid[ix as usize] = m;
            }
            let dd = config.layout.draw_data_base + m.scene_offset;
            let di = m.info_offset as usize;
            if tag_word == DrawTag::COLOR
                || tag_word == DrawTag::LINEAR_GRADIENT
                || tag_word == DrawTag::RADIAL_GRADIENT
                || tag_word == DrawTag::IMAGE
                || tag_word == DrawTag::BEGIN_CLIP
            {
                let bbox = path_bbox[m.path_ix as usize];
                let transform = Transform::read(config.layout.transform_base, bbox.trans_ix, scene);
                let fill_mode = (bbox.linewidth >= 0.0) as u32;
                let mut linewidth = bbox.linewidth;
                match tag_word {
                    DrawTag::COLOR => {
                        info[di] = f32::to_bits(linewidth);
                    }
                    DrawTag::IMAGE => {
                        let z = transform.0;
                        info[di] = f32::to_bits(linewidth);
                        let inv_det = (z[0] * z[3] - z[1] * z[2]).recip();
                        let inv_mat = [
                            z[3] * inv_det,
                            -z[1] * inv_det,
                            -z[2] * inv_det,
                            z[0] * inv_det,
                        ];
                        let inv_tr = [
                            -(inv_mat[0] * z[4] + inv_mat[2] * z[5]),
                            -(inv_mat[1] * z[4] + inv_mat[3] * z[5]),
                        ];
                        info[di + 1] = f32::to_bits(inv_mat[0]);
                        info[di + 2] = f32::to_bits(inv_mat[1]);
                        info[di + 3] = f32::to_bits(inv_mat[2]);
                        info[di + 4] = f32::to_bits(inv_mat[3]);
                        info[di + 5] = f32::to_bits(inv_tr[0]);
                        info[di + 6] = f32::to_bits(inv_tr[1]);
                        info[di + 7] = scene[dd as usize];
                        info[di + 8] = scene[dd as usize + 1];
                    }
                    DrawTag::BEGIN_CLIP => (),
                    _ => todo!("unhandled draw tag {:x}", tag_word.0),
                }
            }
            if tag_word == DrawTag::BEGIN_CLIP {
                let path_ix = m.path_ix as i32;
                clip_inp[m.clip_ix as usize] = Clip { ix, path_ix };
            } else if tag_word == DrawTag::END_CLIP {
                let path_ix = !ix as i32;
                clip_inp[m.clip_ix as usize] = Clip { ix, path_ix };
            }
            m = m.combine(&DrawMonoid::new(tag_word));
        }
        prefix = prefix.combine(&reduced[i as usize]);
    }
}

pub fn draw_leaf(n_wg: u32, resources: &[CpuBinding]) {
    let r0 = resources[0].as_buf();
    let r1 = resources[1].as_buf();
    let r2 = resources[2].as_buf();
    let r3 = resources[3].as_buf();
    let mut r4 = resources[4].as_buf();
    let mut r5 = resources[5].as_buf();
    let mut r6 = resources[6].as_buf();
    let config = bytemuck::from_bytes(&r0);
    let scene = bytemuck::cast_slice(&r1);
    let reduced = bytemuck::cast_slice(&r2);
    let path_bbox = bytemuck::cast_slice(&r3);
    let draw_monoid = bytemuck::cast_slice_mut(r4.as_mut());
    let info = bytemuck::cast_slice_mut(r5.as_mut());
    let clip_inp = bytemuck::cast_slice_mut(r6.as_mut());
    draw_leaf_main(
        n_wg,
        config,
        scene,
        reduced,
        path_bbox,
        draw_monoid,
        info,
        clip_inp,
    );
}
