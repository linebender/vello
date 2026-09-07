// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{Clip, ConfigUniform, DrawMonoid, DrawTag, Monoid, PathBbox};

use super::{
    CpuBinding, RAD_GRAD_KIND_CIRCULAR, RAD_GRAD_KIND_CONE, RAD_GRAD_KIND_FOCAL_ON_CIRCLE,
    RAD_GRAD_KIND_STRIP, RAD_GRAD_SWAPPED,
    util::{Transform, Vec2, read_draw_tag_from_scene},
};

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
    let num_blocks_total = (config.layout.n_draw_objects as usize).div_ceil(WG_SIZE);
    let n_blocks_base = num_blocks_total / WG_SIZE;
    let remainder = num_blocks_total % WG_SIZE;
    let mut prefix = DrawMonoid::default();
    for i in 0..n_wg as usize {
        let first_block = n_blocks_base * i + i.min(remainder);
        let n_blocks = n_blocks_base + (i < remainder) as usize;
        let mut m = prefix;
        for j in 0..WG_SIZE * n_blocks {
            let ix = (first_block * WG_SIZE) as u32 + j as u32;
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
                || tag_word == DrawTag::SWEEP_GRADIENT
                || tag_word == DrawTag::IMAGE
                || tag_word == DrawTag::BEGIN_CLIP
                || tag_word == DrawTag::BLUR_RECT
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
                        const GRADIENT_EPSILON: f32 = 1.0_f32 / (1 << 12) as f32;
                        info[di] = draw_flags;
                        let mut p0 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 1]),
                            f32::from_bits(scene[dd as usize + 2]),
                        );
                        let mut p1 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 3]),
                            f32::from_bits(scene[dd as usize + 4]),
                        );
                        let mut r0 = f32::from_bits(scene[dd as usize + 5]);
                        let mut r1 = f32::from_bits(scene[dd as usize + 6]);
                        let user_to_gradient = transform.inverse();
                        let xform;
                        let mut focal_x = 0.0;
                        let radius;
                        let mut kind;
                        let mut flags = 0;
                        if (r0 - r1).abs() < GRADIENT_EPSILON {
                            // When the radii are the same, emit a strip gradient
                            kind = RAD_GRAD_KIND_STRIP;
                            let scaled = r0 / p0.distance(p1);
                            xform = two_point_to_unit_line(p0, p1) * user_to_gradient;
                            radius = scaled * scaled;
                        } else {
                            // Assume a two point conical gradient unless the centers
                            // are equal.
                            kind = RAD_GRAD_KIND_CONE;
                            if p0 == p1 {
                                kind = RAD_GRAD_KIND_CIRCULAR;
                                // Nudge p0 a bit to avoid denormals.
                                p0.x += GRADIENT_EPSILON;
                            }
                            if r1 == 0.0 {
                                // If r1 == 0.0, swap the points and radii
                                flags |= RAD_GRAD_SWAPPED;
                                core::mem::swap(&mut p0, &mut p1);
                                core::mem::swap(&mut r0, &mut r1);
                            }
                            focal_x = r0 / (r0 - r1);
                            let cf = (1.0 - focal_x) * p0 + focal_x * p1;
                            radius = r1 / cf.distance(p1);
                            let user_to_unit_line =
                                two_point_to_unit_line(cf, p1) * user_to_gradient;
                            let user_to_scaled;
                            // When r == 1.0, focal point is on circle
                            if (radius - 1.0).abs() <= GRADIENT_EPSILON {
                                kind = RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                                let scale = 0.5 * (1.0 - focal_x).abs();
                                user_to_scaled = Transform([scale, 0.0, 0.0, scale, 0.0, 0.0])
                                    * user_to_unit_line;
                            } else {
                                let a = radius * radius - 1.0;
                                let scale_ratio = (1.0 - focal_x).abs() / a;
                                let scale_x = radius * scale_ratio;
                                let scale_y = a.abs().sqrt() * scale_ratio;
                                user_to_scaled = Transform([scale_x, 0.0, 0.0, scale_y, 0.0, 0.0])
                                    * user_to_unit_line;
                            }
                            xform = user_to_scaled;
                        }
                        info[di + 1] = f32::to_bits(xform.0[0]);
                        info[di + 2] = f32::to_bits(xform.0[1]);
                        info[di + 3] = f32::to_bits(xform.0[2]);
                        info[di + 4] = f32::to_bits(xform.0[3]);
                        info[di + 5] = f32::to_bits(xform.0[4]);
                        info[di + 6] = f32::to_bits(xform.0[5]);
                        info[di + 7] = f32::to_bits(focal_x);
                        info[di + 8] = f32::to_bits(radius);
                        info[di + 9] = (flags << 3) | kind;
                    }
                    DrawTag::SWEEP_GRADIENT => {
                        info[di] = draw_flags;
                        let p0 = Vec2::new(
                            f32::from_bits(scene[dd as usize + 1]),
                            f32::from_bits(scene[dd as usize + 2]),
                        );
                        let xform =
                            (transform * Transform([1.0, 0.0, 0.0, 1.0, p0.x, p0.y])).inverse();
                        info[di + 1] = f32::to_bits(xform.0[0]);
                        info[di + 2] = f32::to_bits(xform.0[1]);
                        info[di + 3] = f32::to_bits(xform.0[2]);
                        info[di + 4] = f32::to_bits(xform.0[3]);
                        info[di + 5] = f32::to_bits(xform.0[4]);
                        info[di + 6] = f32::to_bits(xform.0[5]);
                        info[di + 7] = scene[dd as usize + 3];
                        info[di + 8] = scene[dd as usize + 4];
                    }
                    DrawTag::IMAGE => {
                        info[di] = draw_flags;
                        let xform = transform.inverse();
                        info[di + 1] = f32::to_bits(xform.0[0]);
                        info[di + 2] = f32::to_bits(xform.0[1]);
                        info[di + 3] = f32::to_bits(xform.0[2]);
                        info[di + 4] = f32::to_bits(xform.0[3]);
                        info[di + 5] = f32::to_bits(xform.0[4]);
                        info[di + 6] = f32::to_bits(xform.0[5]);
                        info[di + 7] = scene[dd as usize];
                        info[di + 8] = scene[dd as usize + 1];
                        info[di + 9] = scene[dd as usize + 2];
                    }
                    DrawTag::BLUR_RECT => {
                        info[di] = draw_flags;
                        let xform = transform.inverse();
                        info[di + 1] = f32::to_bits(xform.0[0]);
                        info[di + 2] = f32::to_bits(xform.0[1]);
                        info[di + 3] = f32::to_bits(xform.0[2]);
                        info[di + 4] = f32::to_bits(xform.0[3]);
                        info[di + 5] = f32::to_bits(xform.0[4]);
                        info[di + 6] = f32::to_bits(xform.0[5]);
                        info[di + 7] = scene[dd as usize + 1];
                        info[di + 8] = scene[dd as usize + 2];
                        info[di + 9] = scene[dd as usize + 3];
                        info[di + 10] = scene[dd as usize + 4];
                    }
                    DrawTag::BEGIN_CLIP => {
                        info[di] = draw_flags;
                    }
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
        prefix = prefix.combine(&reduced[i]);
    }
}

pub fn draw_leaf(n_wg: u32, resources: &[CpuBinding<'_>]) {
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

fn two_point_to_unit_line(p0: Vec2, p1: Vec2) -> Transform {
    let tmp1 = from_poly2(p0, p1);
    let inv = tmp1.inverse();
    let tmp2 = from_poly2(Vec2::default(), Vec2::new(1.0, 0.0));
    tmp2 * inv
}

fn from_poly2(p0: Vec2, p1: Vec2) -> Transform {
    Transform([
        p1.y - p0.y,
        p0.x - p1.x,
        p1.x - p0.x,
        p1.y - p0.y,
        p0.x,
        p0.y,
    ])
}
