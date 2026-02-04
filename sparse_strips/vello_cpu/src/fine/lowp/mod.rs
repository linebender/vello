// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Low-precision (u8/u16) rendering kernel implementation.
//!
//! This module implements the fine rasterization stage using 8-bit unsigned integers
//! for color values and 16-bit for intermediate calculations. This provides better
//! performance on many architectures compared to floating-point operations, while
//! maintaining sufficient precision for most rendering tasks.

mod compose;
mod gradient;
mod image;

use crate::filter::filter_lowp;
use crate::fine::lowp::image::{BilinearImagePainter, PlainBilinearImagePainter};
use crate::fine::{COLOR_COMPONENTS, Painter, SCRATCH_BUF_SIZE};
use crate::fine::{FineKernel, highp, u8_to_f32};
use crate::layer_manager::LayerManager;
use crate::peniko::BlendMode;
use crate::region::Region;
use crate::util::scalar::div_255;
use bytemuck::cast_slice;
use core::iter;
use vello_common::coarse::WideTile;
use vello_common::encode::{EncodedGradient, EncodedImage};
use vello_common::fearless_simd::*;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
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

    /// Extracts RGBA color components from a premultiplied color as u8 values [0, 255].
    #[inline]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_rgba8().to_u8_array()
    }

    /// Copies rendered pixels from the scratch buffer to the output region.
    ///
    /// Converts from column-major scratch buffer layout to row-major region layout,
    /// using either a SIMD-optimized path for full tiles or a scalar fallback.
    #[inline(always)]
    fn pack(simd: S, region: &mut Region<'_>, blend_buf: &[Self::Numeric]) {
        if region.width != WideTile::WIDTH || region.height != Tile::HEIGHT {
            // Use scalar path for non-standard tile sizes. Wrapping this in `vectorize`
            // degrades performance significantly on SSE4.2.
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

    /// Copies pixels from the output region to the scratch buffer.
    ///
    /// Converts from row-major region layout to column-major scratch buffer layout.
    /// This is the inverse operation of `pack`.
    #[inline(always)]
    fn unpack(simd: S, region: &mut Region<'_>, blend_buf: &mut [Self::Numeric]) {
        if region.width != WideTile::WIDTH || region.height != Tile::HEIGHT {
            // Use scalar path for non-standard tile sizes.
            // Note that right now, this path is unused (only for benchmarking), because when
            // using filters we always allocate pixmaps of the same size as a wide tile.
            // Nevertheless, we still keep this function here for reference, or in case that
            // changes in the future.
            unpack(region, blend_buf);
        } else {
            simd.vectorize(
                #[inline(always)]
                || {
                    unpack_block(simd, region, blend_buf);
                },
            );
        }
    }

    /// Applies a filter effect to a rendered layer.
    ///
    /// Delegates to the u8-specific filter implementation.
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        layer_manager: &mut LayerManager,
        transform: Affine,
    ) {
        filter_lowp(filter, pixmap, layer_manager, transform);
    }

    /// Fills a buffer with a solid color using SIMD operations.
    ///
    /// Efficiently broadcasts a single RGBA color across all pixels in the destination.
    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let target: &mut [u32] = bytemuck::cast_slice_mut(dest);
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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
                    el.copy_from_slice(mulled.as_slice());
                }
            },
        );
    }

    /// Applies a painter's output to the destination buffer.
    ///
    /// Delegates to the painter's u8-specific implementation.
    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], mut painter: impl Painter + 'a) {
        painter.paint_u8(dest);
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
}

mod fill {
    //! Alpha compositing and blending operations without per-pixel alpha masks.
    //!
    //! This module handles the case where we're compositing full opacity pixels,
    //! using only the source alpha channel for compositing.

    use crate::fine::Splat4thExt;
    use crate::fine::lowp::compose::ComposeExt;
    use crate::fine::lowp::mix;
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
                        mix(next_src, bg_v, blend_mode)
                    };
                    let res = blend_mode.compose(simd, src_v, bg_v, None);
                    next_dest.copy_from_slice(res.as_slice());
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
                    next_dest.copy_from_slice(combined.as_slice());
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
                    next_dest.copy_from_slice(res.as_slice());
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
    use crate::fine::lowp::{extract_masks, mix};
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
                        mix(next_src, bg_v, blend_mode)
                    };
                    let masks = extract_masks(simd, &next_mask);
                    let res = blend_mode.compose(simd, src_c, bg_v, Some(masks));

                    next_bg.copy_from_slice(res.as_slice());
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

                dest.copy_from_slice(res.as_slice());
            },
        );
    }
}

/// Applies blend mode mixing by converting to f32, mixing, then converting back to u8.
///
/// TODO: Add a proper lowp mix pipeline that operates entirely in integer space
/// for better performance (currently converts through f32 which is slower).
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

/// Copies color data from the scratch buffer to the output region (scalar fallback).
///
/// The scratch buffer stores pixels in column-major order for SIMD efficiency,
/// while the region uses row-major order for output.
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

/// Copies color data from the output region to the scratch buffer.
///
/// Converts from row-major (region) to column-major (scratch buffer) layout.
/// This is the inverse operation of `pack`.
#[inline(always)]
fn unpack(region: &mut Region<'_>, blend_buf: &mut [u8]) {
    for y in 0..Tile::HEIGHT {
        for (x, pixel) in region.row_mut(y).chunks_exact(COLOR_COMPONENTS).enumerate() {
            let idx = COLOR_COMPONENTS * (usize::from(Tile::HEIGHT) * x + usize::from(y));
            blend_buf[idx..][..COLOR_COMPONENTS].copy_from_slice(pixel);
        }
    }
}

/// SIMD-optimized version of `pack` for full-size tiles using interleaved loads.
///
/// Uses `load_interleaved_128` to efficiently transpose and copy data from column-major
/// scratch buffer to row-major output region. Performance characteristics are highly
/// architecture-dependent:
/// - On NEON: ~3x faster than scalar `pack`
/// - On fallback SIMD: ~3x slower than scalar `pack`
///
/// TODO: Consider runtime detection to fall back to scalar on non-NEON architectures.
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

        let loaded = simd.load_interleaved_128_u32x16(casted).to_bytes();
        dest_slices[0][dest_idx..][..16].copy_from_slice(&loaded.as_slice()[..16]);
        dest_slices[1][dest_idx..][..16].copy_from_slice(&loaded.as_slice()[16..32]);
        dest_slices[2][dest_idx..][..16].copy_from_slice(&loaded.as_slice()[32..48]);
        dest_slices[3][dest_idx..][..16].copy_from_slice(&loaded.as_slice()[48..64]);
    }
}

/// The pendant to `pack_block`, but for unpacking.
///
/// See the [`unpack`] method for more information.
#[inline(always)]
fn unpack_block<S: Simd>(simd: S, region: &mut Region<'_>, buf: &mut [u8]) {
    let buf: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[..SCRATCH_BUF_SIZE]);
    const CHUNK_LENGTH: usize = 16;

    let region_areas = region.areas();
    let [s1, s2, s3, s4] = region_areas;

    for (idx, col) in buf.as_chunks_mut::<CHUNK_LENGTH>().0.iter_mut().enumerate() {
        let src_idx = idx * CHUNK_LENGTH;

        // Note: We experimented with using u32 vs. f32 for this, but it seems like for some reason
        // f32 works better on M1, while on M4 they are the same. Probably worth doing more
        // benchmarks on different systems.
        let r0 = f32x4::from_bytes(u8x16::from_slice(simd, &s1[src_idx..][..16]));
        let r1 = f32x4::from_bytes(u8x16::from_slice(simd, &s2[src_idx..][..16]));
        let r2 = f32x4::from_bytes(u8x16::from_slice(simd, &s3[src_idx..][..16]));
        let r3 = f32x4::from_bytes(u8x16::from_slice(simd, &s4[src_idx..][..16]));

        let combined = simd.combine_f32x8(simd.combine_f32x4(r0, r1), simd.combine_f32x4(r2, r3));

        simd.store_interleaved_128_f32x16(combined, col);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::fearless_simd::dispatch;

    fn test_pack_unpack_roundtrip(
        pack_fn: impl FnOnce(&mut Region<'_>, &[u8]),
        unpack_fn: impl FnOnce(&mut Region<'_>, &mut [u8]),
    ) {
        let width = WideTile::WIDTH;
        let height = Tile::HEIGHT;

        // Just some pseudo-random numbers.
        let blend_buf = (0..SCRATCH_BUF_SIZE)
            .map(|n| ((n * 7 + 13) % 256) as u8)
            .collect::<Vec<_>>();

        let mut region_data = vec![0_u8; width as usize * height as usize * COLOR_COMPONENTS];
        let row_len = width as usize * COLOR_COMPONENTS;
        let (r0, rest) = region_data.split_at_mut(row_len);
        let (r1, rest) = rest.split_at_mut(row_len);
        let (r2, r3) = rest.split_at_mut(row_len);
        let mut region = Region::new([r0, r1, r2, r3], 0, 0, width, height);

        // First pack.
        pack_fn(&mut region, &blend_buf);

        // Now reverse the process, unpacking into a new buffer.
        let mut unpacked_buf = vec![0_u8; SCRATCH_BUF_SIZE];
        unpack_fn(&mut region, &mut unpacked_buf);

        assert_eq!(&blend_buf, &unpacked_buf);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        test_pack_unpack_roundtrip(pack, unpack);
    }

    #[test]
    fn pack_block_unpack_block_roundtrip() {
        dispatch!(Level::try_detect().unwrap_or(Level::fallback()), simd => {
            test_pack_unpack_roundtrip(
                |region, buf| simd.vectorize(|| pack_block(simd, region, buf)),
                |region, buf| simd.vectorize(|| unpack_block(simd, region, buf)),
            );
        });
    }
}
