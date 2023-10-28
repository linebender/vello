// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use crate::cpu_dispatch::CpuBinding;

use super::util::{Transform, Vec2};
use vello_encoding::{
    BumpAllocators, ConfigUniform, LineSoup, Monoid, PathBbox, PathMonoid, Style,
};

fn to_minus_one_quarter(x: f32) -> f32 {
    // could also be written x.powf(-0.25)
    x.sqrt().sqrt().recip()
}

const D: f32 = 0.67;
fn approx_parabola_integral(x: f32) -> f32 {
    x * to_minus_one_quarter(1.0 - D + (D * D * D * D + 0.25 * x * x))
}

const B: f32 = 0.39;
fn approx_parabola_inv_integral(x: f32) -> f32 {
    x * (1.0 - B + (B * B + 0.5 * x * x)).sqrt()
}

#[derive(Clone, Copy, Default)]
struct SubdivResult {
    val: f32,
    a0: f32,
    a2: f32,
}

fn estimate_subdiv(p0: Vec2, p1: Vec2, p2: Vec2, sqrt_tol: f32) -> SubdivResult {
    let d01 = p1 - p0;
    let d12 = p2 - p1;
    let dd = d01 - d12;
    let cross = (p2.x - p0.x) * dd.y - (p2.y - p0.y) * dd.x;
    let cross_inv = if cross.abs() < 1.0e-9 {
        1.0e9
    } else {
        cross.recip()
    };
    let x0 = d01.dot(dd) * cross_inv;
    let x2 = d12.dot(dd) * cross_inv;
    let scale = (cross / (dd.length() * (x2 - x0))).abs();
    let a0 = approx_parabola_integral(x0);
    let a2 = approx_parabola_integral(x2);
    let mut val = 0.0;
    if scale < 1e9 {
        let da = (a2 - a0).abs();
        let sqrt_scale = scale.sqrt();
        if x0.signum() == x2.signum() {
            val = sqrt_scale;
        } else {
            let xmin = sqrt_tol / sqrt_scale;
            val = sqrt_tol / approx_parabola_integral(xmin);
        }
        val *= da;
    }
    SubdivResult { val, a0, a2 }
}

fn eval_quad(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    p0 * (mt * mt) + (p1 * (mt * 2.0) + p2 * t) * t
}

fn eval_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    p0 * (mt * mt * mt) + (p1 * (mt * mt * 3.0) + (p2 * (mt * 3.0) + p3 * t) * t) * t
}

const MAX_QUADS: u32 = 16;

struct Cubic {
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    path_ix: u32,
}

fn flatten_cubic(cubic: Cubic, line_ix: &mut usize, lines: &mut [LineSoup]) {
    let p0 = cubic.p0;
    let p1 = cubic.p1;
    let p2 = cubic.p2;
    let p3 = cubic.p3;
    let err_v = (p2 - p1) * 3.0 + p0 - p3;
    let err = err_v.dot(err_v);
    const ACCURACY: f32 = 0.25;
    const Q_ACCURACY: f32 = ACCURACY * 0.1;
    const REM_ACCURACY: f32 = ACCURACY - Q_ACCURACY;
    const MAX_HYPOT2: f32 = 432.0 * Q_ACCURACY * Q_ACCURACY;
    let mut n_quads = ((err * (1.0 / MAX_HYPOT2)).powf(1.0 / 6.0).ceil() as u32).max(1);
    n_quads = n_quads.min(MAX_QUADS);
    let mut keep_params = [SubdivResult::default(); MAX_QUADS as usize];
    let mut val = 0.0;
    let mut qp0 = p0;
    let step = (n_quads as f32).recip();
    for i in 0..n_quads {
        let t = (i + 1) as f32 * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        let mut qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = qp1 * 2.0 - (qp0 + qp2) * 0.5;
        let params = estimate_subdiv(qp0, qp1, qp2, REM_ACCURACY.sqrt());
        keep_params[i as usize] = params;
        val += params.val;
        qp0 = qp2;
    }
    let n = ((val * (0.5 / REM_ACCURACY.sqrt())).ceil() as u32).max(1);
    let mut lp0 = p0;
    qp0 = p0;
    let v_step = val / (n as f32);
    let mut n_out = 1;
    let mut val_sum = 0.0;
    for i in 0..n_quads {
        let t = (i + 1) as f32 * step;
        let qp2 = eval_cubic(p0, p1, p2, p3, t);
        let mut qp1 = eval_cubic(p0, p1, p2, p3, t - 0.5 * step);
        qp1 = qp1 * 2.0 - (qp0 + qp2) * 0.5;
        let params = keep_params[i as usize];
        let u0 = approx_parabola_inv_integral(params.a0);
        let u2 = approx_parabola_inv_integral(params.a2);
        let uscale = (u2 - u0).recip();
        let mut val_target = (n_out as f32) * v_step;
        while n_out == n || val_target < val_sum + params.val {
            let lp1 = if n_out == n {
                p3
            } else {
                let u = (val_target - val_sum) / params.val;
                let a = params.a0 + (params.a2 - params.a0) * u;
                let au = approx_parabola_inv_integral(a);
                let t = (au - u0) * uscale;
                eval_quad(qp0, qp1, qp2, t)
            };
            let ls = LineSoup {
                path_ix: cubic.path_ix,
                _padding: Default::default(),
                p0: lp0.to_array(),
                p1: lp1.to_array(),
            };
            lines[*line_ix] = ls;
            *line_ix += 1;
            n_out += 1;
            val_target += v_step;
            lp0 = lp1;
        }
        val_sum += params.val;
        qp0 = qp2;
    }
}

fn read_f32_point(ix: u32, pathdata: &[u32]) -> Vec2 {
    let x = f32::from_bits(pathdata[ix as usize]);
    let y = f32::from_bits(pathdata[ix as usize + 1]);
    Vec2 { x, y }
}

struct IntBbox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Default for IntBbox {
    fn default() -> Self {
        IntBbox {
            x0: 0x7fff_ffff,
            y0: 0x7fff_ffff,
            x1: -0x8000_0000,
            y1: -0x8000_0000,
        }
    }
}

impl IntBbox {
    fn add_pt(&mut self, pt: Vec2) {
        self.x0 = self.x0.min(pt.x.floor() as i32);
        self.y0 = self.y0.min(pt.y.floor() as i32);
        self.x1 = self.x1.max(pt.x.ceil() as i32);
        self.y1 = self.y1.max(pt.y.ceil() as i32);
    }
}

// TODO: we're skipping i16 point reading as it's not present in our scenes

const WG_SIZE: usize = 256;

const PATH_TAG_SEG_TYPE: u8 = 3;
const PATH_TAG_PATH: u8 = 0x10;
const PATH_TAG_LINETO: u8 = 1;
const PATH_TAG_QUADTO: u8 = 2;
const PATH_TAG_CUBICTO: u8 = 3;
const PATH_TAG_F32: u8 = 8;

fn flatten_main(
    n_wg: u32,
    config: &ConfigUniform,
    scene: &[u32],
    tag_monoids: &[PathMonoid],
    path_bboxes: &mut [PathBbox],
    bump: &mut BumpAllocators,
    lines: &mut [LineSoup],
) {
    let mut line_ix = 0;
    let mut bbox = IntBbox::default();
    for ix in 0..n_wg as usize * WG_SIZE {
        let tag_word = scene[config.layout.path_tag_base as usize + (ix >> 2)];
        let shift = (ix & 3) * 8;
        let mut tm = PathMonoid::new(tag_word & ((1 << shift) - 1));
        let tag_byte = (tag_word >> shift) as u8;
        if tag_byte != 0 {
            tm = tag_monoids[ix >> 2].combine(&tm);
        }
        let style_flags = scene[(config.layout.style_base + tm.style_ix) as usize];
        if (tag_byte & PATH_TAG_PATH) != 0 {
            let out = &mut path_bboxes[tm.path_ix as usize];
            // TODO: We assume all paths are fills at the moment. This is where we will extract the
            // stroke vs fill state using STYLE_FLAGS_STYLE_BIT.
            // TODO: The downstream pipelines still use the old floating-point linewidth/fill
            // encoding scheme. This will change to represent the fill rule as a single bit inside
            // the bounding box and draw info data structures.
            out.linewidth = if (style_flags & Style::FLAGS_FILL_BIT) == 0 {
                -1.0
            } else {
                -2.0
            };
            out.trans_ix = tm.trans_ix;
        }
        let seg_type = tag_byte & PATH_TAG_SEG_TYPE;
        let pathdata = &scene[config.layout.path_data_base as usize..];
        if seg_type != 0 {
            let mut p0;
            let mut p1;
            let mut p2 = Vec2::default();
            let mut p3 = Vec2::default();
            if (tag_byte & PATH_TAG_F32) != 0 {
                p0 = read_f32_point(tm.pathseg_offset, pathdata);
                p1 = read_f32_point(tm.pathseg_offset + 2, pathdata);
                if seg_type >= PATH_TAG_QUADTO {
                    p2 = read_f32_point(tm.pathseg_offset + 4, pathdata);
                    if seg_type == PATH_TAG_CUBICTO {
                        p3 = read_f32_point(tm.pathseg_offset + 6, pathdata);
                    }
                }
            } else {
                todo!("i16 path data not supported yet");
            }
            let transform = Transform::read(config.layout.transform_base, tm.trans_ix, scene);
            p0 = transform.apply(p0);
            bbox.add_pt(p0);
            p1 = transform.apply(p1);
            bbox.add_pt(p1);
            if seg_type == PATH_TAG_LINETO {
                p3 = p1;
                p2 = p3.mix(p0, 1.0 / 3.0);
                p1 = p0.mix(p3, 1.0 / 3.0);
            } else if seg_type >= PATH_TAG_QUADTO {
                p2 = transform.apply(p2);
                bbox.add_pt(p2);
                if seg_type == PATH_TAG_CUBICTO {
                    p3 = transform.apply(p3);
                    bbox.add_pt(p3);
                } else {
                    p3 = p2;
                    p2 = p1.mix(p2, 1.0 / 3.0);
                    p1 = p1.mix(p0, 1.0 / 3.0);
                }
            }
            let path_ix = tm.path_ix;
            let cubic = Cubic {
                p0,
                p1,
                p2,
                p3,
                path_ix,
            };
            flatten_cubic(cubic, &mut line_ix, lines);
        }
        if (tag_byte & PATH_TAG_PATH) != 0 {
            let out = &mut path_bboxes[tm.path_ix as usize];
            out.x0 = bbox.x0;
            out.y0 = bbox.y0;
            out.x1 = bbox.x1;
            out.y1 = bbox.y1;
            bbox = IntBbox::default();
        }
    }
    bump.lines = line_ix as u32;
}

pub fn flatten(n_wg: u32, resources: &[CpuBinding]) {
    let config = resources[0].as_typed();
    let scene = resources[1].as_slice();
    let tag_monoids = resources[2].as_slice();
    let mut path_bboxes = resources[3].as_slice_mut();
    let mut bump = resources[4].as_typed_mut();
    let mut lines = resources[5].as_slice_mut();
    flatten_main(
        n_wg,
        &config,
        &scene,
        &tag_monoids,
        &mut path_bboxes,
        &mut bump,
        &mut lines,
    );
}
