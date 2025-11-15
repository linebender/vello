// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod compose;
mod gradient;
mod image;

use crate::fine::lowp::image::BilinearImagePainter;
use crate::fine::{COLOR_COMPONENTS, Painter, SCRATCH_BUF_SIZE};
use crate::fine::{FineKernel, highp, u8_to_f32};
use crate::peniko::BlendMode;
use crate::region::Region;
use crate::util::scalar::div_255;
use bytemuck::cast_slice;
use core::iter;
use vello_common::coarse::WideTile;
use vello_common::encode::{EncodedGradient, EncodedImage};
use vello_common::fearless_simd::*;
use vello_common::mask::Mask;
use vello_common::paint::PremulColor;
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;
use vello_common::util::{Div255Ext, f32_to_u8};

/// The kernel for doing rendering using u8/u16.
#[derive(Clone, Copy, Debug)]
pub struct U8Kernel;

impl<S: Simd> FineKernel<S> for U8Kernel {
    type Numeric = u8;
    type Composite = u8x32<S>;
    type NumericVec = u8x16<S>;

    #[inline]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_rgba8().to_u8_array()
    }

    #[inline(always)]
    fn pack(simd: S, region: &mut Region<'_>, blend_buf: &[Self::Numeric]) {
        if region.width != WideTile::WIDTH || region.height != Tile::HEIGHT {
            // For some reason putting this into `vectorize` as well makes it much slower on
            // SSE4.2
            pack(region, blend_buf);
        } else {
            simd.vectorize(
                #[inline(always)]
                || {
                    pack_block(simd, region, blend_buf);
                },
            );
        }
    }

    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let color = u8x64::block_splat(
                    u32x4::splat(simd, u32::from_ne_bytes(src)).reinterpret_u8(),
                );

                for el in dest.chunks_exact_mut(64) {
                    el.copy_from_slice(&color.val);
                }
            },
        );
    }

    fn gradient_painter<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || gradient::GradientPainter::new(simd, gradient, t_vals),
        )
    }

    fn medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || BilinearImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
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
                    let loaded = u8x16::from_slice(simd, el);
                    let mulled = simd.narrow_u16x16(
                        (simd.widen_u8x16(loaded) * simd.widen_u8x16(src.next().unwrap()))
                            .div_255(),
                    );
                    el.copy_from_slice(&mulled.val);
                }
            },
        );
    }

    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], mut painter: impl Painter + 'a) {
        painter.paint_u8(dest);
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
                alphas
                    .chunks_exact(8)
                    .map(|d| [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]),
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
        let src_iter = src.chunks_exact(32).map(|el| u8x32::from_slice(simd, el));

        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite(
                simd,
                dest,
                src_iter,
                alphas
                    .chunks_exact(8)
                    .map(|d| [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]),
            );
        } else {
            fill::alpha_composite(simd, dest, src_iter);
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
        let alpha_iter = alphas.map(|a| {
            a.chunks_exact(8)
                .map(|d| [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
        });

        let mask_iter = mask.map(|m| {
            iter::from_fn(|| {
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
                    sample(start_x + 1, start_y),
                    sample(start_x + 1, start_y + 1),
                    sample(start_x + 1, start_y + 2),
                    sample(start_x + 1, start_y + 3),
                ];

                start_x += 2;

                Some(samples)
            })
        });

        match (alpha_iter, mask_iter) {
            (Some(alpha_iter), Some(mut mask_iter)) => {
                let iter = alpha_iter.map(|a1| {
                    let a2 = mask_iter.next().unwrap();
                    [
                        div_255(a1[0] as u16 * a2[0] as u16) as u8,
                        div_255(a1[1] as u16 * a2[1] as u16) as u8,
                        div_255(a1[2] as u16 * a2[2] as u16) as u8,
                        div_255(a1[3] as u16 * a2[3] as u16) as u8,
                        div_255(a1[4] as u16 * a2[4] as u16) as u8,
                        div_255(a1[5] as u16 * a2[5] as u16) as u8,
                        div_255(a1[6] as u16 * a2[6] as u16) as u8,
                        div_255(a1[7] as u16 * a2[7] as u16) as u8,
                    ]
                });
                alpha_fill::blend(simd, dest, src, blend_mode, iter);
            }
            (None, Some(mask_iter)) => alpha_fill::blend(simd, dest, src, blend_mode, mask_iter),
            (Some(alpha_iter), None) => alpha_fill::blend(simd, dest, src, blend_mode, alpha_iter),
            (None, None) => {
                fill::blend(simd, dest, src, blend_mode);
            }
        }
    }
}

mod fill {
    use crate::fine::Splat4thExt;
    use crate::fine::lowp::compose::ComposeExt;
    use crate::fine::lowp::mix;
    use crate::peniko::{BlendMode, Mix};
    use vello_common::fearless_simd::*;
    use vello_common::util::normalized_mul_u8x32;

    pub(super) fn blend<S: Simd, T: Iterator<Item = u8x32<S>>>(
        simd: S,
        dest: &mut [u8],
        src: T,
        blend_mode: BlendMode,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                #[expect(deprecated, reason = "Provided by the user, need to handle correctly.")]
                let default_mix = matches!(blend_mode.mix, Mix::Normal | Mix::Clip);
                for (next_dest, next_src) in dest.chunks_exact_mut(32).zip(src) {
                    let bg_v = u8x32::from_slice(simd, next_dest);
                    let src_v = if default_mix {
                        next_src
                    } else {
                        mix(next_src, bg_v, blend_mode)
                    };
                    let res = blend_mode.compose(simd, src_v, bg_v, None);
                    next_dest.copy_from_slice(&res.val);
                }
            },
        );
    }

    pub(super) fn alpha_composite_solid<S: Simd>(s: S, dest: &mut [u8], src: [u8; 4]) {
        s.vectorize(
            #[inline(always)]
            || {
                let one_minus_alpha = 255 - u8x32::splat(s, src[3]);
                let src_c = u32x8::splat(s, u32::from_ne_bytes(src)).reinterpret_u8();

                for next_dest in dest.chunks_exact_mut(64) {
                    // We process in batches of 64 because loading/storing is much faster this way (at least on NEON),
                    // but since we widen to u16, we can only work with 256 bits, so we split it up.
                    let bg_v = u8x64::from_slice(s, next_dest);
                    let (bg_1, bg_2) = s.split_u8x64(bg_v);
                    let res_1 = alpha_composite_inner(s, bg_1, src_c, one_minus_alpha);
                    let res_2 = alpha_composite_inner(s, bg_2, src_c, one_minus_alpha);
                    let combined = s.combine_u8x32(res_1, res_2);
                    next_dest.copy_from_slice(&combined.val);
                }
            },
        );
    }

    pub(super) fn alpha_composite<S: Simd, T: Iterator<Item = u8x32<S>>>(
        simd: S,
        dest: &mut [u8],
        src: T,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for (next_dest, next_src) in dest.chunks_exact_mut(32).zip(src) {
                    let one_minus_alpha = 255 - next_src.splat_4th();
                    let bg_v = u8x32::from_slice(simd, next_dest);
                    let res = alpha_composite_inner(simd, bg_v, next_src, one_minus_alpha);
                    next_dest.copy_from_slice(&res.val);
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        bg: u8x32<S>,
        src: u8x32<S>,
        one_minus_alpha: u8x32<S>,
    ) -> u8x32<S> {
        s.narrow_u16x32(normalized_mul_u8x32(bg, one_minus_alpha)) + src
    }
}

mod alpha_fill {
    use crate::fine::Splat4thExt;
    use crate::fine::lowp::compose::ComposeExt;
    use crate::fine::lowp::{extract_masks, mix};
    use crate::peniko::{BlendMode, Mix};
    use vello_common::fearless_simd::*;
    use vello_common::util::{Div255Ext, normalized_mul_u8x32};

    pub(super) fn blend<S: Simd, T: Iterator<Item = u8x32<S>>>(
        simd: S,
        dest: &mut [u8],
        src: T,
        blend_mode: BlendMode,
        alphas: impl Iterator<Item = [u8; 8]>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                #[expect(deprecated, reason = "Provided by the user, need to handle correctly.")]
                let default_mix = matches!(blend_mode.mix, Mix::Normal | Mix::Clip);

                for ((next_bg, next_mask), next_src) in
                    dest.chunks_exact_mut(32).zip(alphas).zip(src)
                {
                    let bg_v = u8x32::from_slice(simd, next_bg);
                    let src_c = if default_mix {
                        next_src
                    } else {
                        mix(next_src, bg_v, blend_mode)
                    };
                    let masks = extract_masks(simd, &next_mask);
                    let res = blend_mode.compose(simd, src_c, bg_v, Some(masks));

                    next_bg.copy_from_slice(&res.val);
                }
            },
        );
    }

    #[inline(always)]
    pub(super) fn alpha_composite_solid<S: Simd>(
        s: S,
        dest: &mut [u8],
        src: [u8; 4],
        alphas: impl Iterator<Item = [u8; 8]>,
    ) {
        s.vectorize(
            #[inline(always)]
            || {
                let src_a = u8x32::splat(s, src[3]);
                let src_c = u32x8::splat(s, u32::from_ne_bytes(src)).reinterpret_u8();
                let one = u8x32::splat(s, 255);

                for (next_bg, next_mask) in dest.chunks_exact_mut(32).zip(alphas) {
                    alpha_composite_inner(s, next_bg, &next_mask, src_c, src_a, one);
                }
            },
        );
    }

    #[inline(always)]
    pub(super) fn alpha_composite<S: Simd, T: Iterator<Item = u8x32<S>>>(
        simd: S,
        dest: &mut [u8],
        src: T,
        alphas: impl Iterator<Item = [u8; 8]>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                let one = u8x32::splat(simd, 255);

                for ((next_dest, next_mask), next_src) in
                    dest.chunks_exact_mut(32).zip(alphas).zip(src)
                {
                    let src_a = next_src.splat_4th();
                    alpha_composite_inner(simd, next_dest, &next_mask, next_src, src_a, one);
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [u8],
        masks: &[u8; 8],
        src_c: u8x32<S>,
        src_a: u8x32<S>,
        one: u8x32<S>,
    ) {
        s.vectorize(
            #[inline(always)]
            || {
                let bg_v = u8x32::from_slice(s, dest);

                let mask_v = extract_masks(s, masks);
                let inv_src_a_mask_a = one - s.narrow_u16x32(normalized_mul_u8x32(src_a, mask_v));

                let p1 = s.widen_u8x32(bg_v) * s.widen_u8x32(inv_src_a_mask_a);
                let p2 = s.widen_u8x32(src_c) * s.widen_u8x32(mask_v);
                let res = s.narrow_u16x32((p1 + p2).div_255());

                dest.copy_from_slice(&res.val);
            },
        );
    }
}

// TODO: Add a proper lowp mix pipeline
fn mix<S: Simd>(src_c: u8x32<S>, bg_c: u8x32<S>, blend_mode: BlendMode) -> u8x32<S> {
    let to_f32 = |val: u8x32<S>| {
        let (a, b) = src_c.simd.split_u8x32(val);
        let mut a = u8_to_f32(a);
        let mut b = u8_to_f32(b);
        a *= f32x16::splat(src_c.simd, 1.0 / 255.0);
        b *= f32x16::splat(src_c.simd, 1.0 / 255.0);
        (a, b)
    };

    let to_u8 = |val1: f32x16<S>, val2: f32x16<S>| {
        let val1 =
            f32_to_u8(f32x16::splat(val1.simd, 255.0).madd(val1, f32x16::splat(val1.simd, 0.5)));
        let val2 =
            f32_to_u8(f32x16::splat(val2.simd, 255.0).madd(val2, f32x16::splat(val2.simd, 0.5)));

        val1.simd.combine_u8x16(val1, val2)
    };

    let (mut src_1, mut src_2) = to_f32(src_c);
    let (bg_1, bg_2) = to_f32(bg_c);

    src_1 = highp::blend::mix(src_1, bg_1, blend_mode);
    src_2 = highp::blend::mix(src_2, bg_2, blend_mode);

    to_u8(src_1, src_2)
}

#[inline(always)]
fn extract_masks<S: Simd>(simd: S, masks: &[u8; 8]) -> u8x32<S> {
    let m1 =
        u32x4::splat(simd, u32::from_ne_bytes(masks[0..4].try_into().unwrap())).reinterpret_u8();
    let m2 =
        u32x4::splat(simd, u32::from_ne_bytes(masks[4..8].try_into().unwrap())).reinterpret_u8();

    let zipped1 = m1.zip_low(m1);
    let zipped1 = zipped1.zip_low(zipped1);

    let zipped2 = m2.zip_low(m2);
    let zipped2 = zipped2.zip_low(zipped2);

    simd.combine_u8x16(zipped1, zipped2)
}

#[inline(always)]
fn pack(region: &mut Region<'_>, blend_buf: &[u8]) {
    for y in 0..Tile::HEIGHT {
        for (x, pixel) in region
            .row_mut(y)
            .chunks_exact_mut(COLOR_COMPONENTS)
            .enumerate()
        {
            let idx = COLOR_COMPONENTS * (usize::from(Tile::HEIGHT) * x + usize::from(y));
            pixel.copy_from_slice(&blend_buf[idx..][..COLOR_COMPONENTS]);
        }
    }
}

// Note: This method is 3x slower than `pack_regular` when using fallback SIMD, but it's
// 3x faster than `pack_regular` using the NEON level. Perhaps we should add a way of
// always falling back to `regular` when in fallback mode.
#[inline(always)]
fn pack_block<S: Simd>(simd: S, region: &mut Region<'_>, mut buf: &[u8]) {
    buf = &buf[..SCRATCH_BUF_SIZE];

    const CHUNK_LENGTH: usize = 64;
    const SLICE_WIDTH: usize = WideTile::WIDTH as usize * COLOR_COMPONENTS;

    let region_areas = region.areas();
    let [s1, s2, s3, s4] = region_areas;
    let dest_slices: &mut [&mut [u8; SLICE_WIDTH]; 4] = &mut [
        (*s1).try_into().unwrap(),
        (*s2).try_into().unwrap(),
        (*s3).try_into().unwrap(),
        (*s4).try_into().unwrap(),
    ];

    for (idx, col) in buf.chunks_exact(CHUNK_LENGTH).enumerate() {
        let dest_idx = idx * CHUNK_LENGTH / 4;

        let casted: &[u32; 16] = cast_slice::<u8, u32>(col).try_into().unwrap();

        let loaded = simd.load_interleaved_128_u32x16(casted).reinterpret_u8();
        dest_slices[0][dest_idx..][..16].copy_from_slice(&loaded.val[..16]);
        dest_slices[1][dest_idx..][..16].copy_from_slice(&loaded.val[16..32]);
        dest_slices[2][dest_idx..][..16].copy_from_slice(&loaded.val[32..48]);
        dest_slices[3][dest_idx..][..16].copy_from_slice(&loaded.val[48..64]);
    }
}
