// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! High-precision (f32) rendering kernel implementation.
//!
//! This module implements the fine rasterization stage using 32-bit floating-point
//! values for color components. This provides maximum precision and color accuracy,
//! at the cost of higher memory bandwidth and potentially slower performance compared
//! to the low-precision u8 kernel.
//!
//! The f32 kernel is particularly useful for:
//! - Scenes requiring high precision (e.g., gradients with subtle color transitions)
//! - Debugging and reference implementations
//! - Platforms where SIMD f32 operations are well-optimized

use crate::filter::filter_highp;
use crate::fine::FineKernel;
use crate::fine::{COLOR_COMPONENTS, Painter};
use crate::layer_manager::LayerManager;
use crate::peniko::BlendMode;
use crate::region::Region;
use vello_common::fearless_simd::*;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::mask::Mask;
use vello_common::paint::PremulColor;
use vello_common::pixmap::Pixmap;
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

    /// Extracts RGBA color components from a premultiplied color as f32 values [0.0, 1.0].
    #[inline(always)]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_f32().components
    }

    /// Copies rendered pixels from the scratch buffer to the output region.
    ///
    /// Converts f32 color values in [0.0, 1.0] range to u8 [0, 255] with rounding,
    /// transforming from column-major scratch buffer to row-major region layout.
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
                        // Convert f32 [0.0, 1.0] to u8 [0, 255] with rounding.
                        // TODO: Use explicit SIMD for better performance
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

    /// Copies pixels from the output region to the scratch buffer.
    ///
    /// Converts u8 color values [0, 255] to normalized f32 [0.0, 1.0],
    /// transforming from row-major region layout to column-major scratch buffer.
    /// This is the inverse operation of `pack`.
    #[inline(always)]
    fn unpack(simd: S, region: &mut Region<'_>, blend_buf: &mut [Self::Numeric]) {
        simd.vectorize(
            #[inline(always)]
            || {
                for y in 0..Tile::HEIGHT {
                    for (x, pixel) in region.row_mut(y).chunks_exact(COLOR_COMPONENTS).enumerate() {
                        let idx =
                            COLOR_COMPONENTS * (usize::from(Tile::HEIGHT) * x + usize::from(y));
                        let start = &mut blend_buf[idx..];
                        // Convert u8 [0, 255] to normalized f32 [0.0, 1.0] (reverse of pack)
                        start[0] = pixel[0] as f32 / 255.0;
                        start[1] = pixel[1] as f32 / 255.0;
                        start[2] = pixel[2] as f32 / 255.0;
                        start[3] = pixel[3] as f32 / 255.0;
                    }
                }
            },
        );
    }

    /// Applies a filter effect to a rendered layer.
    ///
    /// Delegates to the f32-specific filter implementation.
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        layer_manager: &mut LayerManager,
        transform: Affine,
    ) {
        filter_highp(filter, pixmap, layer_manager, transform);
    }

    /// Fills a buffer with a solid color using SIMD operations.
    ///
    /// Efficiently broadcasts a single RGBA color across all pixels in the destination.
    #[inline(never)]
    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let color = f32x16::block_splat(src.simd_into(simd));

                for el in dest.chunks_exact_mut(16) {
                    el.copy_from_slice(color.as_slice());
                }
            },
        );
    }

    /// Applies per-pixel mask values to a buffer by multiplying each component.
    ///
    /// Used for anti-aliasing and clipping effects. Each pixel is multiplied by
    /// its corresponding mask value (already normalized to [0.0, 1.0]).
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
                    el.copy_from_slice(mulled.as_slice());
                }
            },
        );
    }

    /// Applies a painter's output to the destination buffer.
    ///
    /// Delegates to the painter's f32-specific implementation.
    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], mut painter: impl Painter + 'a) {
        painter.paint_f32(dest);
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
                bytemuck::cast_slice::<u8, [u8; 4]>(alphas).iter().copied(),
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
        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
                bytemuck::cast_slice::<u8, [u8; 4]>(alphas).iter().copied(),
            );
        } else {
            fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
            );
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
        let alpha_iter = alphas.map(|a| bytemuck::cast_slice::<u8, [u8; 4]>(a).iter().copied());

        let mask_iter = mask.map(|m| {
            let width = m.width();
            let height = m.height();

            core::iter::from_fn(move || {
                let samples = if start_x < width && start_y + 3 < height {
                    // All in bounds, sample directly
                    [
                        m.sample(start_x, start_y),
                        m.sample(start_x, start_y + 1),
                        m.sample(start_x, start_y + 2),
                        m.sample(start_x, start_y + 3),
                    ]
                } else {
                    // Fallback: check each individually
                    [
                        if start_x < width && start_y < height {
                            m.sample(start_x, start_y)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 1 < height {
                            m.sample(start_x, start_y + 1)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 2 < height {
                            m.sample(start_x, start_y + 2)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 3 < height {
                            m.sample(start_x, start_y + 3)
                        } else {
                            255
                        },
                    ]
                };

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
    //! Alpha compositing and blending operations without per-pixel alpha masks.
    //!
    //! This module handles the case where we're compositing full opacity pixels,
    //! using only the source alpha channel for compositing.

    use crate::fine::Splat4thExt;
    use crate::fine::highp::blend;
    use crate::fine::highp::compose::ComposeExt;
    use crate::peniko::BlendMode;

    use vello_common::fearless_simd::*;

    // IMPORTANT: The inlining attributes (#[inline(always)], #[inline(never)]) in this
    // module have been carefully tuned through benchmarking. Changing them can cause
    // significant performance regressions.

    /// Composites a solid color onto a buffer using alpha blending.
    ///
    /// Uses the "over" operator: `result = src + bg * (1 - src_alpha)`
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

    /// Composites a buffer of colors onto another buffer using alpha blending.
    ///
    /// Each source pixel is composited individually based on its alpha channel.
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

    /// Applies blend mode compositing to a buffer without per-pixel masks.
    pub(super) fn blend<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        blend_mode: BlendMode,
    ) {
        for (next_dest, next_src) in dest.chunks_exact_mut(16).zip(src) {
            let bg_v = f32x16::from_slice(simd, next_dest);
            let src_c = blend::mix(next_src, bg_v, blend_mode);
            let res = blend_mode.compose(simd, src_c, bg_v, None);
            next_dest.copy_from_slice(res.as_slice());
        }
    }

    /// Performs the core alpha compositing calculation.
    ///
    /// Formula: `result = src + bg * (1 - src_alpha)`
    /// This implements the Porter-Duff "source over" operator using FMA for efficiency.
    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [f32],
        src: f32x16<S>,
        one_minus_alpha: f32x16<S>,
    ) {
        let mut bg_c = f32x16::from_slice(s, dest);
        bg_c = one_minus_alpha.madd(bg_c, src);
        dest.copy_from_slice(bg_c.as_slice());
    }
}

mod alpha_fill {
    //! Alpha compositing and blending operations with per-pixel alpha masks.
    //!
    //! This module handles compositing when each pixel has an additional mask value
    //! (e.g., from anti-aliasing or clip masks) that modulates the source alpha.

    use crate::fine::Splat4thExt;
    use crate::fine::highp::compose::ComposeExt;
    use crate::fine::highp::{blend, extract_masks};
    use crate::peniko::BlendMode;
    use vello_common::fearless_simd::*;

    /// Composites a solid color with per-pixel alpha masks.
    ///
    /// Combines source alpha with mask values: `effective_alpha = src_alpha * mask`
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

    /// Composites a buffer of colors with per-pixel alpha masks.
    ///
    /// Each pixel's source alpha is modulated by its corresponding mask value.
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

    /// Applies blend mode compositing with per-pixel alpha masks.
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
                    let res = blend_mode.compose(simd, src_c, bg, Some(masks));
                    next_dest.copy_from_slice(res.as_slice());
                }
            },
        );
    }

    /// Performs alpha compositing with mask modulation.
    ///
    /// Formula: `result = src * mask + bg * (1 - src_alpha * mask)`
    /// The mask value modulates both the source contribution and the inverse alpha.
    /// Uses FMA instructions for optimal performance.
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
        // 1 - src_a * mask_a
        let inv_src_a_mask_a = src_a.madd(-mask_a, one);

        let res = bg_c.madd(inv_src_a_mask_a, src_c * mask_a);
        dest.copy_from_slice(res.as_slice());
    }
}

/// Expands 4 mask bytes into a 16-element f32 SIMD vector with normalized values.
///
/// Converts u8 mask values to f32 in range [0.0, 1.0], then duplicates each mask
/// value across 4 consecutive elements (one per color component).
///
/// Input: [m0, m1, m2, m3] (as u8, 0-255)
/// Output: [m0/255, m0/255, m0/255, m0/255, m1/255, ..., m3/255] (as f32, 16 elements)
#[inline(always)]
fn extract_masks<S: Simd>(simd: S, masks: &[u8; 4]) -> f32x16<S> {
    let mut base_mask = [
        masks[0] as f32,
        masks[1] as f32,
        masks[2] as f32,
        masks[3] as f32,
    ]
    .simd_into(simd);

    base_mask *= f32x4::splat(simd, 1.0 / 255.0);

    let res = f32x16::block_splat(base_mask);
    let zip_low = res.zip_low(res);

    zip_low.zip_low(zip_low)
}
