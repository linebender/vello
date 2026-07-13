// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Low-precision (u8/u16) rendering kernel implementation.
//!
//! This module implements the fine rasterization stage using 8-bit unsigned integers
//! for color values and 16-bit for intermediate calculations. This provides better
//! performance on many architectures compared to floating-point operations, while
//! maintaining sufficient precision for most rendering tasks.

pub(crate) mod blend;
mod compose;
mod gradient;
mod image;

use crate::filter::context::ScratchBuffer;
use crate::filter::filter_lowp;
use crate::fine::lowp::image::{BilinearImagePainter, PlainBilinearImagePainter};
use crate::fine::{COLOR_COMPONENTS, FineKernel, Painter, Splat4thExt, TILE_HEIGHT_COMPONENTS};
use crate::peniko::BlendMode;
use crate::region::Region;
use crate::util::NormalizedMulExt;
use crate::util::scalar::div_255;
use bytemuck::{cast_slice, cast_slice_mut};
use core::iter;
use vello_common::encode::{EncodedGradient, EncodedImage};
use vello_common::fearless_simd::*;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::mask::Mask;
use vello_common::paint::{PremulColor, Tint, TintMode};
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;
use vello_common::util::Div255Ext;

/// The kernel for doing rendering using u8/u16.
#[derive(Clone, Copy, Debug)]
pub struct U8Kernel;

impl<S: Simd> FineKernel<S> for U8Kernel {
    type Numeric = u8;
    type Composite = u8x32<S>;
    type NumericVec = u8x16<S>;

    /// Extracts RGBA color components from a premultiplied color as u8 values [0, 255].
    #[inline]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_rgba8().to_u8_array()
    }

    /// Applies a filter effect to a rendered layer.
    ///
    /// Delegates to the u8-specific filter implementation.
    #[expect(
        private_interfaces,
        reason = "`FineKernel` is public but this specific method is not needed."
    )]
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        filter_scratch: &mut ScratchBuffer,
        transform: Affine,
    ) {
        filter_lowp(filter, pixmap, filter_scratch, transform);
    }

    /// Fills a buffer with a solid color using SIMD operations.
    ///
    /// Efficiently broadcasts a single RGBA color across all pixels in the destination.
    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let target: &mut [u32] = cast_slice_mut(dest);
                target.fill(u32::from_ne_bytes(src));
            },
        );
    }

    /// Creates a painter for rendering gradients in u8 precision.
    ///
    /// Returns a painter that evaluates the gradient at each pixel position.
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

    /// Creates a painter for rendering images with bilinear filtering in u8 precision.
    ///
    /// Returns a painter that samples the image with bilinear interpolation.
    fn medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || BilinearImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Creates a painter for rendering axis-aligned images with bilinear filtering in u8 precision.
    ///
    /// Returns a painter that samples the image with bilinear interpolation.
    /// This is an optimized version for images without skew transformation.
    fn plain_medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || PlainBilinearImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Applies per-pixel mask values to a buffer by multiplying each component.
    ///
    /// Used for anti-aliasing and clipping effects. Each pixel is multiplied by
    /// its corresponding mask value and normalized.
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
                    mulled.store_slice(el);
                }
            },
        );
    }

    /// Applies a painter's output to the destination buffer.
    ///
    /// Delegates to the painter's u8-specific implementation.
    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], painter: impl Painter + 'a) {
        painter.paint_u8(dest);
    }

    #[inline(always)]
    fn apply_tint(simd: S, dest: &mut [Self::Numeric], tint: &Tint) {
        let premul = tint.color.premultiply();
        let [r, g, b, a] = premul.components;
        let to_u8 = |v: f32| (v * 255.0 + 0.5) as u8;
        let color = u32::from_ne_bytes([to_u8(r), to_u8(g), to_u8(b), to_u8(a)]);

        simd.vectorize(
            #[inline(always)]
            || {
                let tint_v = u32x8::block_splat(u32x4::splat(simd, color)).to_bytes();

                match tint.mode {
                    TintMode::AlphaMask => {
                        for chunk in dest.chunks_exact_mut(32) {
                            let pixel = u8x32::from_slice(simd, chunk);
                            let alphas = pixel.splat_4th();
                            let tinted = tint_v.normalized_mul(alphas);
                            tinted.store_slice(chunk);
                        }
                    }
                    TintMode::Multiply => {
                        for chunk in dest.chunks_exact_mut(32) {
                            let pixel = u8x32::from_slice(simd, chunk);
                            let tinted = pixel.normalized_mul(tint_v);
                            tinted.store_slice(chunk);
                        }
                    }
                }
            },
        );
    }

    /// Composites a solid color onto a buffer using alpha blending.
    ///
    /// Dispatches to either the masked or unmasked implementation based on the
    /// presence of per-pixel alpha masks.
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
                cast_slice::<u8, [u8; 8]>(alphas).iter().copied(),
            );
        } else {
            fill::alpha_composite_solid(simd, dest, src);
        }
    }

    /// Composites a source buffer onto a destination buffer using alpha blending.
    ///
    /// Dispatches to either the masked or unmasked implementation based on the
    /// presence of per-pixel alpha masks. Each source pixel's alpha determines
    /// the blending amount.
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
                cast_slice::<u8, [u8; 8]>(alphas).iter().copied(),
            );
        } else {
            fill::alpha_composite(simd, dest, src_iter);
        }
    }

    /// Applies a blend mode to composite source pixels onto destination.
    ///
    /// Dispatches to either the masked or unmasked blend implementation.
    /// Handles both color mixing (multiply, screen, etc.) and compositing.
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
        let alpha_iter = alphas.map(|a| cast_slice::<u8, [u8; 8]>(a).iter().copied());

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

    fn pack(simd: S, scratch: &[Self::Numeric], width: usize, region: &mut Region<'_>) {
        simd.vectorize(
            #[inline(always)]
            || {
                pack(simd, scratch, width, region);
            },
        );
    }

    fn unpack(simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [Self::Numeric]) {
        simd.vectorize(
            #[inline(always)]
            || {
                unpack(simd, region, width, scratch);
            },
        );
    }
}

#[inline(always)]
fn pack<S: Simd>(simd: S, scratch: &[u8], width: usize, region: &mut Region<'_>) {
    let block_width = if region.height == Tile::HEIGHT {
        (width / Tile::WIDTH as usize) * Tile::WIDTH as usize
    } else {
        0
    };
    if block_width > 0 {
        pack_block(simd, scratch, block_width, region);
    }

    let tail_width = width - block_width;
    if tail_width > 0 {
        pack_tail(
            &scratch[block_width * TILE_HEIGHT_COMPONENTS..],
            block_width,
            tail_width,
            region,
        );
    }
}

#[inline(always)]
fn pack_block<S: Simd>(simd: S, scratch: &[u8], width: usize, region: &mut Region<'_>) {
    const CHUNK_LENGTH: usize = Tile::WIDTH as usize * TILE_HEIGHT_COMPONENTS;

    let [row0, row1, row2, row3] = region.areas();
    let row_len = width * COLOR_COMPONENTS;
    let mut row0 = &mut row0[..row_len];
    let mut row1 = &mut row1[..row_len];
    let mut row2 = &mut row2[..row_len];
    let mut row3 = &mut row3[..row_len];

    for col in scratch[..width * TILE_HEIGHT_COMPONENTS].chunks_exact(CHUNK_LENGTH) {
        let casted: &[u32; 16] = cast_slice::<u8, u32>(col).try_into().unwrap();

        let loaded = simd.load_interleaved_128_u32x16(casted).to_bytes();
        let (loaded_lo, loaded_hi) = simd.split_u8x64(loaded);
        let (loaded_1, loaded_2) = simd.split_u8x32(loaded_lo);
        let (loaded_3, loaded_4) = simd.split_u8x32(loaded_hi);

        let (dest0, rest0) = row0.split_at_mut(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (dest1, rest1) = row1.split_at_mut(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (dest2, rest2) = row2.split_at_mut(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (dest3, rest3) = row3.split_at_mut(Tile::WIDTH as usize * COLOR_COMPONENTS);

        loaded_1.store_slice(dest0);
        loaded_2.store_slice(dest1);
        loaded_3.store_slice(dest2);
        loaded_4.store_slice(dest3);

        row0 = rest0;
        row1 = rest1;
        row2 = rest2;
        row3 = rest3;
    }
}

#[inline(always)]
fn pack_tail(scratch: &[u8], x: usize, width: usize, region: &mut Region<'_>) {
    for y in 0..region.height {
        let row = &mut region.row_mut(y)[x * COLOR_COMPONENTS..(x + width) * COLOR_COMPONENTS];
        for (dx, pixel) in row.chunks_exact_mut(COLOR_COMPONENTS).enumerate() {
            let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * dx + usize::from(y));
            pixel.copy_from_slice(&scratch[idx..idx + COLOR_COMPONENTS]);
        }
    }
}

#[inline(always)]
fn unpack<S: Simd>(simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [u8]) {
    let block_width = if region.height == Tile::HEIGHT {
        width / Tile::WIDTH as usize * Tile::WIDTH as usize
    } else {
        0
    };
    if block_width > 0 {
        unpack_block(simd, region, block_width, scratch);
    }

    let tail_width = width - block_width;
    if tail_width > 0 {
        unpack_tail(
            region,
            block_width,
            tail_width,
            &mut scratch[block_width * TILE_HEIGHT_COMPONENTS..],
        );
    }
}

#[inline(always)]
fn unpack_block<S: Simd>(simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [u8]) {
    let scratch: &mut [f32] = cast_slice_mut(&mut scratch[..width * TILE_HEIGHT_COMPONENTS]);
    const CHUNK_LENGTH: usize = 16;

    let [row0, row1, row2, row3] = region.areas();
    let row_len = width * COLOR_COMPONENTS;
    let mut row0 = &row0[..row_len];
    let mut row1 = &row1[..row_len];
    let mut row2 = &row2[..row_len];
    let mut row3 = &row3[..row_len];

    for col in scratch.as_chunks_mut::<CHUNK_LENGTH>().0.iter_mut() {
        // Note: We experimented with using u32 vs. f32 for this, but it seems like for some reason
        // f32 works better on M1, while on M4 they are the same. Probably worth doing more
        // benchmarks on different systems.
        let (src0, rest0) = row0.split_at(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (src1, rest1) = row1.split_at(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (src2, rest2) = row2.split_at(Tile::WIDTH as usize * COLOR_COMPONENTS);
        let (src3, rest3) = row3.split_at(Tile::WIDTH as usize * COLOR_COMPONENTS);

        let r0 = f32x4::from_bytes(u8x16::from_slice(simd, src0));
        let r1 = f32x4::from_bytes(u8x16::from_slice(simd, src1));
        let r2 = f32x4::from_bytes(u8x16::from_slice(simd, src2));
        let r3 = f32x4::from_bytes(u8x16::from_slice(simd, src3));
        let combined = simd.combine_f32x8(simd.combine_f32x4(r0, r1), simd.combine_f32x4(r2, r3));

        simd.store_interleaved_128_f32x16(combined, col);

        row0 = rest0;
        row1 = rest1;
        row2 = rest2;
        row3 = rest3;
    }
}

#[inline(always)]
fn unpack_tail(region: &mut Region<'_>, x: usize, width: usize, scratch: &mut [u8]) {
    for y in 0..region.height {
        let row = &region.row_mut(y)[x * COLOR_COMPONENTS..(x + width) * COLOR_COMPONENTS];
        for (dx, pixel) in row.chunks_exact(COLOR_COMPONENTS).enumerate() {
            let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * dx + usize::from(y));
            scratch[idx..idx + COLOR_COMPONENTS].copy_from_slice(pixel);
        }
    }
}

mod fill {
    //! Alpha compositing and blending operations without per-pixel alpha masks.
    //!
    //! This module handles the case where we're compositing full opacity pixels,
    //! using only the source alpha channel for compositing.

    use crate::fine::Splat4thExt;
    use crate::fine::lowp::blend;
    use crate::fine::lowp::compose::ComposeExt;
    use crate::peniko::{BlendMode, Mix};
    use vello_common::fearless_simd::*;
    use vello_common::util::normalized_mul_u8x32;

    /// Applies blend mode compositing to a buffer without per-pixel masks.
    pub(super) fn blend<S: Simd, T: Iterator<Item = u8x32<S>>>(
        simd: S,
        dest: &mut [u8],
        src: T,
        blend_mode: BlendMode,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                let default_mix = matches!(blend_mode.mix, Mix::Normal);
                for (next_dest, next_src) in dest.chunks_exact_mut(32).zip(src) {
                    let bg_v = u8x32::from_slice(simd, next_dest);
                    let src_v = if default_mix {
                        next_src
                    } else {
                        blend::mix(next_src, bg_v, blend_mode)
                    };
                    let res = blend_mode.compose(simd, src_v, bg_v, None);
                    res.store_slice(next_dest);
                }
            },
        );
    }

    /// Composites a solid color onto a buffer using alpha blending.
    ///
    /// Uses the "over" operator: `result = src + bg * (1 - src_alpha)`
    pub(super) fn alpha_composite_solid<S: Simd>(s: S, dest: &mut [u8], src: [u8; 4]) {
        s.vectorize(
            #[inline(always)]
            || {
                let one_minus_alpha = 255 - u8x32::splat(s, src[3]);
                let src_c = u32x8::splat(s, u32::from_ne_bytes(src)).to_bytes();

                for next_dest in dest.chunks_exact_mut(64) {
                    // We process in batches of 64 because loading/storing is much faster this way (at least on NEON),
                    // but since we widen to u16, we can only work with 256 bits, so we split it up.
                    let bg_v = u8x64::from_slice(s, next_dest);
                    let (bg_1, bg_2) = s.split_u8x64(bg_v);
                    let res_1 = alpha_composite_inner(s, bg_1, src_c, one_minus_alpha);
                    let res_2 = alpha_composite_inner(s, bg_2, src_c, one_minus_alpha);
                    let combined = s.combine_u8x32(res_1, res_2);
                    combined.store_slice(next_dest);
                }
            },
        );
    }

    /// Composites a buffer of colors onto another buffer using alpha blending.
    ///
    /// Each source pixel is composited individually based on its alpha channel.
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
                    res.store_slice(next_dest);
                }
            },
        );
    }

    /// Performs the core alpha compositing calculation.
    ///
    /// Formula: `result = src + bg * (1 - src_alpha)`
    /// This implements the Porter-Duff "source over" operator.
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
    //! Alpha compositing and blending operations with per-pixel alpha masks.
    //!
    //! This module handles compositing when each pixel has an additional mask value
    //! (e.g., from anti-aliasing or clip masks) that modulates the source alpha.

    use crate::fine::Splat4thExt;
    use crate::fine::lowp::compose::ComposeExt;
    use crate::fine::lowp::{blend, extract_masks};
    use crate::peniko::{BlendMode, Mix};
    use vello_common::fearless_simd::*;
    use vello_common::util::{Div255Ext, normalized_mul_u8x32};

    /// Applies blend mode compositing with per-pixel alpha masks.
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
                let default_mix = matches!(blend_mode.mix, Mix::Normal);

                for ((next_bg, next_mask), next_src) in
                    dest.chunks_exact_mut(32).zip(alphas).zip(src)
                {
                    let bg_v = u8x32::from_slice(simd, next_bg);
                    let src_c = if default_mix {
                        next_src
                    } else {
                        blend::mix(next_src, bg_v, blend_mode)
                    };
                    let masks = extract_masks(simd, &next_mask);
                    let res = blend_mode.compose(simd, src_c, bg_v, Some(masks));

                    res.store_slice(next_bg);
                }
            },
        );
    }

    /// Composites a solid color with per-pixel alpha masks.
    ///
    /// Combines source alpha with mask values: `effective_alpha = src_alpha * mask / 255`
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
                let src_c = u32x8::splat(s, u32::from_ne_bytes(src)).to_bytes();
                let one = u8x32::splat(s, 255);

                for (next_bg, next_mask) in dest.chunks_exact_mut(32).zip(alphas) {
                    alpha_composite_inner(s, next_bg, &next_mask, src_c, src_a, one);
                }
            },
        );
    }

    /// Composites a buffer of colors with per-pixel alpha masks.
    ///
    /// Each pixel's source alpha is modulated by its corresponding mask value.
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

    /// Performs alpha compositing with mask modulation.
    ///
    /// Formula: `result = src * mask + bg * (1 - src_alpha * mask)`
    /// The mask value modulates both the source contribution and the inverse alpha.
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

                res.store_slice(dest);
            },
        );
    }
}

/// Expands 8 mask bytes into a 32-byte SIMD vector where each pixel's 4 components
/// share the same mask value (each of 8 mask values is repeated 4 times).
///
/// Input: [m0, m1, m2, m3, m4, m5, m6, m7]
/// Output: [m0, m0, m0, m0, m1, m1, m1, m1, ..., m7, m7, m7, m7]
#[inline(always)]
fn extract_masks<S: Simd>(simd: S, masks: &[u8; 8]) -> u8x32<S> {
    let m1 = u32x4::splat(simd, u32::from_ne_bytes(masks[0..4].try_into().unwrap())).to_bytes();
    let m2 = u32x4::splat(simd, u32::from_ne_bytes(masks[4..8].try_into().unwrap())).to_bytes();

    let zipped1 = m1.zip_low(m1);
    let zipped1 = zipped1.zip_low(zipped1);

    let zipped2 = m2.zip_low(m2);
    let zipped2 = zipped2.zip_low(zipped2);

    simd.combine_u8x16(zipped1, zipped2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::fearless_simd::{Level, dispatch};
    use vello_common::geometry::RectU16;

    fn test_pack_unpack_roundtrip(
        width: u16,
        pack_fn: impl FnOnce(&mut Region<'_>, &[u8]),
        unpack_fn: impl FnOnce(&mut Region<'_>, &mut [u8]),
    ) {
        let height = Tile::HEIGHT;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;

        // Just some pseudo-random numbers.
        let scratch = (0..scratch_len)
            .map(|n| ((n * 7 + 13) % 256) as u8)
            .collect::<Vec<_>>();

        let mut pixmap = Pixmap::new(width, height);
        let mut pixmap = pixmap.as_mut();
        let mut region = Region::new(&mut pixmap, RectU16::new(0, 0, width, height));

        pack_fn(&mut region, &scratch);

        let mut unpacked = vec![0_u8; scratch_len];
        unpack_fn(&mut region, &mut unpacked);

        assert_eq!(scratch, unpacked);
    }

    #[test]
    fn pack_block_unpack_block_roundtrip() {
        let width = Tile::WIDTH * 2;
        dispatch!(Level::try_detect().unwrap_or(Level::baseline()), simd => {
            test_pack_unpack_roundtrip(
                width,
                |region, scratch| {
                    simd.vectorize(|| pack_block(simd, scratch, usize::from(width), region));
                },
                |region, scratch| {
                    simd.vectorize(|| unpack_block(simd, region, usize::from(width), scratch));
                },
            );
        });
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let width = Tile::WIDTH * 2 + 1;
        dispatch!(Level::try_detect().unwrap_or(Level::baseline()), simd => {
            test_pack_unpack_roundtrip(
                width,
                |region, scratch| {
                    simd.vectorize(|| pack(simd, scratch, usize::from(width), region));
                },
                |region, scratch| {
                    simd.vectorize(|| unpack(simd, region, usize::from(width), scratch));
                },
            );
        });
    }
}
