// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::FineKernel;
use crate::fine::{COLOR_COMPONENTS, Painter};
use crate::peniko::BlendMode;
use crate::region::Region;
use vello_common::fearless_simd::*;
use vello_common::mask::Mask;
use vello_common::paint::PremulColor;
use vello_common::tile::Tile;

pub(crate) mod blend;
pub(crate) mod compose;

/// The kernel for doing rendering using f32.
#[derive(Clone, Copy, Debug)]
pub struct F32Kernel;

impl<S: Simd> FineKernel<S> for F32Kernel {
    type Numeric = f32;
    type Composite = f32x16<S>;
    type NumericVec = f32x16<S>;

    #[inline(always)]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_f32().components
    }

    #[inline(always)]
    fn pack(simd: S, region: &mut Region<'_>, blend_buf: &[Self::Numeric]) {
        simd.vectorize(
            #[inline(always)]
            || {
                for y in 0..Tile::HEIGHT {
                    for (x, pixel) in region
                        .row_mut(y)
                        .chunks_exact_mut(COLOR_COMPONENTS)
                        .enumerate()
                    {
                        let idx =
                            COLOR_COMPONENTS * (usize::from(Tile::HEIGHT) * x + usize::from(y));
                        let start = &blend_buf[idx..];
                        // TODO: Use explicit SIMD
                        let converted = [
                            (start[0] * 255.0 + 0.5) as u8,
                            (start[1] * 255.0 + 0.5) as u8,
                            (start[2] * 255.0 + 0.5) as u8,
                            (start[3] * 255.0 + 0.5) as u8,
                        ];
                        pixel.copy_from_slice(&converted);
                    }
                }
            },
        );
    }

    // Not having this tanks performance for some reason.
    #[inline(never)]
    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let color = f32x16::block_splat(src.simd_into(simd));

                for el in dest.chunks_exact_mut(16) {
                    el.copy_from_slice(&color.val);
                }
            },
        );
    }

    fn apply_mask(
        simd: S,
        dest: &mut [Self::Numeric],
        mut src: impl Iterator<Item = Self::NumericVec>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for el in dest.chunks_exact_mut(16) {
                    let loaded = f32x16::from_slice(simd, el);
                    let mulled = loaded * src.next().unwrap();
                    el.copy_from_slice(&mulled.val);
                }
            },
        );
    }

    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], mut painter: impl Painter + 'a) {
        painter.paint_f32(dest);
    }

    #[inline(always)]
    fn alpha_composite_solid(
        simd: S,
        dest: &mut [Self::Numeric],
        src: [Self::Numeric; 4],
        alphas: Option<&[u8]>,
    ) {
        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite_solid(
                simd,
                dest,
                src,
                alphas.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]),
            );
        } else {
            fill::alpha_composite_solid(simd, dest, src);
        }
    }

    fn alpha_composite_buffer(
        simd: S,
        dest: &mut [Self::Numeric],
        src: &[Self::Numeric],
        alphas: Option<&[u8]>,
    ) {
        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
                alphas.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]),
            );
        } else {
            fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
            );
        }
    }

    fn blend(
        simd: S,
        dest: &mut [Self::Numeric],
        mut start_x: u16,
        start_y: u16,
        src: impl Iterator<Item = Self::Composite>,
        blend_mode: BlendMode,
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    ) {
        let alpha_iter = alphas.map(|a| a.chunks_exact(4).map(|d| [d[0], d[1], d[2], d[3]]));

        let mask_iter = mask.map(|m| {
            core::iter::from_fn(|| {
                let sample = |x: u16, y: u16| {
                    if x < m.width() && y < m.height() {
                        m.sample(x, y)
                    } else {
                        255
                    }
                };

                let samples = [
                    sample(start_x, start_y),
                    sample(start_x, start_y + 1),
                    sample(start_x, start_y + 2),
                    sample(start_x, start_y + 3),
                ];

                start_x += 1;

                Some(samples)
            })
        });

        match (alpha_iter, mask_iter) {
            (Some(alpha_iter), Some(mut mask_iter)) => {
                let iter = alpha_iter.map(|a1| {
                    let a2 = mask_iter.next().unwrap();
                    [
                        ((a1[0] as u16 * a2[0] as u16) / 255) as u8,
                        ((a1[1] as u16 * a2[1] as u16) / 255) as u8,
                        ((a1[2] as u16 * a2[2] as u16) / 255) as u8,
                        ((a1[3] as u16 * a2[3] as u16) / 255) as u8,
                    ]
                });
                alpha_fill::blend(simd, dest, src, iter, blend_mode);
            }
            (None, Some(mask_iter)) => alpha_fill::blend(simd, dest, src, mask_iter, blend_mode),
            (Some(alpha_iter), None) => alpha_fill::blend(simd, dest, src, alpha_iter, blend_mode),
            (None, None) => {
                fill::blend(simd, dest, src, blend_mode);
            }
        }
    }
}

mod fill {
    use crate::fine::Splat4thExt;
    use crate::fine::highp::blend;
    use crate::fine::highp::compose::ComposeExt;
    use crate::peniko::BlendMode;

    use vello_common::fearless_simd::*;
    // Careful: From my experiments, inlining / not inlining these functions can have drastic (negative)
    // consequences on performance.

    #[inline(always)]
    pub(super) fn alpha_composite_solid<S: Simd>(s: S, dest: &mut [f32], src: [f32; 4]) {
        s.vectorize(
            #[inline(always)]
            || {
                let one_minus_alpha = 1.0 - f32x16::block_splat(f32x4::splat(s, src[3]));
                let src_c = f32x16::block_splat(f32x4::simd_from(src, s));

                for next_dest in dest.chunks_exact_mut(16) {
                    alpha_composite_inner(s, next_dest, src_c, one_minus_alpha);
                }
            },
        );
    }

    #[inline(always)]
    pub(super) fn alpha_composite_arbitrary<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for (next_dest, next_src) in dest.chunks_exact_mut(16).zip(src) {
                    let one_minus_alpha = 1.0 - next_src.splat_4th();
                    alpha_composite_inner(simd, next_dest, next_src, one_minus_alpha);
                }
            },
        );
    }

    pub(super) fn blend<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        blend_mode: BlendMode,
    ) {
        let mask = f32x16::splat(simd, 1.0);

        for (next_dest, next_src) in dest.chunks_exact_mut(16).zip(src) {
            let bg_v = f32x16::from_slice(simd, next_dest);
            let src_c = blend::mix(next_src, bg_v, blend_mode);
            let res = blend_mode.compose(simd, src_c, bg_v, mask);
            next_dest.copy_from_slice(&res.val);
        }
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [f32],
        src: f32x16<S>,
        one_minus_alpha: f32x16<S>,
    ) {
        let mut bg_c = f32x16::from_slice(s, dest);
        bg_c = src.madd(one_minus_alpha, bg_c);
        dest.copy_from_slice(&bg_c.val);
    }
}

mod alpha_fill {
    use crate::fine::Splat4thExt;
    use crate::fine::highp::compose::ComposeExt;
    use crate::fine::highp::{blend, extract_masks};
    use crate::peniko::BlendMode;
    use vello_common::fearless_simd::*;

    #[inline(always)]
    pub(super) fn alpha_composite_solid<S: Simd>(
        s: S,
        dest: &mut [f32],
        src: [f32; 4],
        alphas: impl Iterator<Item = [u8; 4]>,
    ) {
        s.vectorize(
            #[inline(always)]
            || {
                let src_a = f32x16::splat(s, src[3]);
                let src_c = f32x16::block_splat(src.simd_into(s));
                let one = f32x16::splat(s, 1.0);

                for (next_dest, next_mask) in dest.chunks_exact_mut(16).zip(alphas) {
                    alpha_composite_inner(s, next_dest, &next_mask, src_c, src_a, one);
                }
            },
        );
    }

    pub(super) fn alpha_composite_arbitrary<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        alphas: impl Iterator<Item = [u8; 4]>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                let one = f32x16::splat(simd, 1.0);

                for ((next_dest, next_mask), next_src) in
                    dest.chunks_exact_mut(16).zip(alphas).zip(src)
                {
                    let src_a = next_src.splat_4th();
                    alpha_composite_inner(simd, next_dest, &next_mask, next_src, src_a, one);
                }
            },
        );
    }

    pub(super) fn blend<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        alphas: impl Iterator<Item = [u8; 4]>,
        blend_mode: BlendMode,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for ((next_dest, next_mask), next_src) in
                    dest.chunks_exact_mut(16).zip(alphas).zip(src)
                {
                    let masks = extract_masks(simd, &next_mask);
                    let bg = f32x16::from_slice(simd, next_dest);
                    let src_c = blend::mix(next_src, bg, blend_mode);
                    let res = blend_mode.compose(simd, src_c, bg, masks);
                    next_dest.copy_from_slice(&res.val);
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [f32],
        masks: &[u8; 4],
        src_c: f32x16<S>,
        src_a: f32x16<S>,
        one: f32x16<S>,
    ) {
        let bg_c = f32x16::from_slice(s, dest);
        let mask_a = extract_masks(s, masks);
        let inv_src_a_mask_a = one.msub(src_a, mask_a);

        let res = (src_c * mask_a).madd(bg_c, inv_src_a_mask_a);
        dest.copy_from_slice(&res.val);
    }
}

#[inline(always)]
fn extract_masks<S: Simd>(simd: S, masks: &[u8; 4]) -> f32x16<S> {
    let mut base_mask = [
        masks[0] as f32,
        masks[1] as f32,
        masks[2] as f32,
        masks[3] as f32,
    ]
    .simd_into(simd);

    base_mask = base_mask * f32x4::splat(simd, 1.0 / 255.0);

    let res = f32x16::block_splat(base_mask);
    let zip_low = res.zip_low(res);

    zip_low.zip_low(zip_low)
}
