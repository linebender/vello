// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, DrawTag, Monoid, PathBbox};

use crate::cpu_dispatch::CpuBinding;

use super::util::{read_draw_tag_from_scene, Transform, Vec2};

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
    let mut prefix = DrawMonoid::default();
    for i in 0..n_wg {
        let mut m = prefix;
        for j in 0..WG_SIZE {
            let ix = i * WG_SIZE as u32 + j as u32;
            let tag_raw = read_draw_tag_from_scene(config, scene, ix);
            let tag_word = DrawTag(tag_raw);
            // store exclusive prefix sum
            if ix < config.layout.n_draw_objects {
                draw_monoid[ix as usize] = m;
            }
            let m_next = m.combine(&DrawMonoid::new(tag_word));
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
                let draw_flags = bbox.draw_flags;
                match tag_word {
                    DrawTag::COLOR => {
                        info[di] = draw_flags;
                    }
                    DrawTag::LINEAR_GRADIENT => {
                        info[di] = draw_flags;
                        let p0 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 1]),
                            f32::from_bits(scene[dd as usize + 2]),
                        );
                        let p1 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 3]),
                            f32::from_bits(scene[dd as usize + 4]),
                        );
                        let p0 = transform.apply(p0);
                        let p1 = transform.apply(p1);
                        let dxy = p1 - p0;
                        let scale = 1.0 / dxy.dot(dxy);
                        let line_xy = dxy * scale;
                        let line_c = -p0.dot(line_xy);
                        info[di + 1] = f32::to_bits(line_xy.x);
                        info[di + 2] = f32::to_bits(line_xy.y);
                        info[di + 3] = f32::to_bits(line_c);
                    }
                    DrawTag::RADIAL_GRADIENT => {
                        info[di] = draw_flags;
                        let p0 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 1]),
                            f32::from_bits(scene[dd as usize + 2]),
                        );
                        let p1 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 3]),
                            f32::from_bits(scene[dd as usize + 4]),
                        );
                        let r0 = f32::from_bits(scene[dd as usize + 5]);
                        let r1 = f32::from_bits(scene[dd as usize + 6]);
                        let z = transform.0;
                        let inv_det = (z[0] * z[3] - z[1] * z[2]).recip();
                        let inv_mat = [
                            z[3] * inv_det,
                            -z[1] * inv_det,
                            -z[2] * inv_det,
                            z[0] * inv_det,
                        ];
                        let inv_tr = [
                            -(inv_mat[0] * z[4] + inv_mat[2] * z[5]) - p0.x,
                            -(inv_mat[1] * z[4] + inv_mat[3] * z[5]) - p0.y,
                        ];
                        let center1 = p1 - p0;
                        let rr = r1 / (r1 - r0);
                        let ra_inv = rr / (r1 * r1 - center1.dot(center1));
                        let c1 = center1 * ra_inv;
                        let ra = rr * ra_inv;
                        let roff = rr - 1.0;
                        info[di + 1] = f32::to_bits(inv_mat[0]);
                        info[di + 2] = f32::to_bits(inv_mat[1]);
                        info[di + 3] = f32::to_bits(inv_mat[2]);
                        info[di + 4] = f32::to_bits(inv_mat[3]);
                        info[di + 5] = f32::to_bits(inv_tr[0]);
                        info[di + 6] = f32::to_bits(inv_tr[1]);
                        info[di + 7] = f32::to_bits(c1.x);
                        info[di + 8] = f32::to_bits(c1.y);
                        info[di + 9] = f32::to_bits(ra);
                        info[di + 19] = f32::to_bits(roff);
                    }
                    DrawTag::IMAGE => {
                        info[di] = draw_flags;
                        let z = transform.0;
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
            m = m_next;
        }
        prefix = prefix.combine(&reduced[i as usize]);
    }
}

pub fn draw_leaf(n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let reduced = resources[2].as_slice();
    let path_bbox = resources[3].as_slice();
    let mut draw_monoid = resources[4].as_slice_mut();
    let mut info = resources[5].as_slice_mut();
    let mut clip_inp = resources[6].as_slice_mut();
    draw_leaf_main(
        n_wg,
        &config,
        &scene,
        &reduced,
        &path_bbox,
        &mut draw_monoid,
        &mut info,
        &mut clip_inp,
    );
}
