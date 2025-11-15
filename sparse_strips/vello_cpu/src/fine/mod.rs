// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod common;
mod highp;
mod lowp;

use crate::peniko::{BlendMode, ImageQuality};
use crate::region::Region;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use vello_common::coarse::{Cmd, WideTile};
use vello_common::encode::{
    EncodedBlurredRoundedRectangle, EncodedGradient, EncodedImage, EncodedKind, EncodedPaint,
};
use vello_common::paint::{ImageSource, Paint, PremulColor};
use vello_common::tile::Tile;

pub(crate) const COLOR_COMPONENTS: usize = 4;
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;
pub const SCRATCH_BUF_SIZE: usize =
    WideTile::WIDTH as usize * Tile::HEIGHT as usize * COLOR_COMPONENTS;

use crate::fine::common::gradient::linear::SimdLinearKind;
use crate::fine::common::gradient::radial::SimdRadialKind;
use crate::fine::common::gradient::sweep::SimdSweepKind;
use crate::fine::common::gradient::{GradientPainter, calculate_t_vals};
use crate::fine::common::image::{FilteredImagePainter, NNImagePainter, PlainNNImagePainter};
use crate::fine::common::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::util::{BlendModeExt, EncodedImageExt};
pub use highp::F32Kernel;
pub use lowp::U8Kernel;
use vello_common::fearless_simd::{
    Simd, SimdBase, SimdFloat, SimdInto, f32x4, f32x8, f32x16, u8x16, u8x32, u32x4, u32x8,
};
use vello_common::mask::Mask;
use vello_common::pixmap::Pixmap;
use vello_common::simd::Splat4thExt;
use vello_common::util::f32_to_u8;

pub type ScratchBuf<F> = [F; SCRATCH_BUF_SIZE];

pub trait Numeric: Copy + Default + Clone + Debug + PartialEq + Send + Sync + 'static {
    const ZERO: Self;
    const ONE: Self;
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Numeric for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 255;
}

pub trait NumericVec<S: Simd>: Copy + Clone + Send + Sync {
    fn from_f32(simd: S, val: f32x16<S>) -> Self;
    fn from_u8(simd: S, val: u8x16<S>) -> Self;
}

impl<S: Simd> NumericVec<S> for f32x16<S> {
    #[inline(always)]
    fn from_f32(_: S, val: Self) -> Self {
        val
    }

    #[inline(always)]
    fn from_u8(simd: S, val: u8x16<S>) -> Self {
        let converted = u8_to_f32(val);
        converted * Self::splat(simd, 1.0 / 255.0)
    }
}

impl<S: Simd> NumericVec<S> for u8x16<S> {
    #[inline(always)]
    fn from_f32(simd: S, val: f32x16<S>) -> Self {
        let v1 = f32x16::splat(simd, 255.0);
        let v2 = f32x16::splat(simd, 0.5);
        let mulled = val.madd(v1, v2);

        f32_to_u8(mulled)
    }

    #[inline(always)]
    fn from_u8(_: S, val: Self) -> Self {
        val
    }
}

#[inline(always)]
pub(crate) fn u8_to_f32<S: Simd>(val: u8x16<S>) -> f32x16<S> {
    let simd = val.simd;
    let zeroes = u8x16::splat(simd, 0);

    let zip1 = simd.zip_high_u8x16(val, zeroes);
    let zip2 = simd.zip_low_u8x16(val, zeroes);

    let p1 = simd.zip_low_u8x16(zip2, zeroes).reinterpret_u32().cvt_f32();
    let p2 = simd
        .zip_high_u8x16(zip2, zeroes)
        .reinterpret_u32()
        .cvt_f32();
    let p3 = simd.zip_low_u8x16(zip1, zeroes).reinterpret_u32().cvt_f32();
    let p4 = simd
        .zip_high_u8x16(zip1, zeroes)
        .reinterpret_u32()
        .cvt_f32();

    simd.combine_f32x8(simd.combine_f32x4(p1, p2), simd.combine_f32x4(p3, p4))
}

pub trait CompositeType<N: Numeric, S: Simd>: Copy + Clone + Send + Sync {
    const LENGTH: usize;

    fn from_slice(simd: S, slice: &[N]) -> Self;
    fn from_color(simd: S, color: [N; 4]) -> Self;
}

impl<S: Simd> CompositeType<f32, S> for f32x16<S> {
    const LENGTH: usize = 16;

    #[inline(always)]
    fn from_slice(simd: S, slice: &[f32]) -> Self {
        <Self as SimdBase<_, _>>::from_slice(simd, slice)
    }

    #[inline(always)]
    fn from_color(simd: S, color: [f32; 4]) -> Self {
        Self::block_splat(f32x4::from_slice(simd, &color[..]))
    }
}

impl<S: Simd> CompositeType<u8, S> for u8x32<S> {
    const LENGTH: usize = 32;

    #[inline(always)]
    fn from_slice(simd: S, slice: &[u8]) -> Self {
        <Self as SimdBase<_, _>>::from_slice(simd, slice)
    }

    #[inline(always)]
    fn from_color(simd: S, color: [u8; 4]) -> Self {
        u32x8::block_splat(u32x4::splat(simd, u32::from_ne_bytes(color))).reinterpret_u8()
    }
}

/// A kernel for performing fine rasterization.
pub trait FineKernel<S: Simd>: Send + Sync + 'static {
    /// The basic underlying numerical type of the kernel.
    type Numeric: Numeric;
    /// The type that is used for blending and compositing.
    type Composite: CompositeType<Self::Numeric, S>;
    /// The base SIMD vector type for converting between u8 and f32.
    type NumericVec: NumericVec<S>;

    /// Extract the color from a premultiplied color.
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4];
    /// Pack the blend buf into the given region.
    fn pack(simd: S, region: &mut Region<'_>, blend_buf: &[Self::Numeric]);
    /// Repeatedly copy the solid color into the target buffer.
    fn copy_solid(simd: S, target: &mut [Self::Numeric], color: [Self::Numeric; 4]);
    /// Return the painter used for painting gradients.
    fn gradient_painter<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, false, t_vals),
        )
    }
    /// Return the painter used for painting gradients, with support for masking undefined locations.
    fn gradient_painter_with_undefined<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, true, t_vals),
        )
    }
    /// Return the painter used for painting plain nearest-neighbor images.
    ///
    /// Plain nearest-neighbor images are images with the quality 'Low' and no skewing component in their
    /// transform.
    fn plain_nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || PlainNNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }
    /// Return the painter used for painting plain nearest-neighbor images.
    ///
    /// Same as `plain_nn`, but must also support skewing transforms.
    fn nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || NNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }
    /// Return the painter used for painting image with `Medium` quality.
    fn medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }
    /// Return the painter used for painting image with `High` quality.
    fn high_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }
    /// Return the painter used for painting blurred rounded rectangles.
    fn blurred_rounded_rectangle_painter(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: u16,
        start_y: u16,
    ) -> impl Painter {
        simd.vectorize(
            #[inline(always)]
            || BlurredRoundedRectFiller::new(simd, rect, start_x, start_y),
        )
    }
    /// Apply the mask to the destination buffer.
    fn apply_mask(simd: S, dest: &mut [Self::Numeric], src: impl Iterator<Item = Self::NumericVec>);
    /// Apply the painter to the destination buffer.
    fn apply_painter<'a>(simd: S, dest: &mut [Self::Numeric], painter: impl Painter + 'a);
    /// Do basic alpha compositing with a solid color.
    fn alpha_composite_solid(
        simd: S,
        target: &mut [Self::Numeric],
        src: [Self::Numeric; 4],
        alphas: Option<&[u8]>,
    );
    /// Do basic alpha compositing with the given buffer.
    fn alpha_composite_buffer(
        simd: S,
        dest: &mut [Self::Numeric],
        src: &[Self::Numeric],
        alphas: Option<&[u8]>,
    );
    /// Blend the source into the destination with the given blend mode.
    fn blend(
        simd: S,
        dest: &mut [Self::Numeric],
        start_x: u16,
        start_y: u16,
        src: impl Iterator<Item = Self::Composite>,
        blend_mode: BlendMode,
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    );
}

/// An object for performing fine rasterization
#[derive(Debug)]
pub struct Fine<S: Simd, T: FineKernel<S>> {
    /// The coordinates of the currently covered wide tile.
    pub(crate) wide_coords: (u16, u16),
    /// The stack of blend buffers.
    pub(crate) blend_buf: Vec<ScratchBuf<T::Numeric>>,
    /// An intermediate buffer used by shaders to store their contents.
    pub(crate) paint_buf: ScratchBuf<T::Numeric>,
    /// An intermediate buffer used by gradients to store the t values.
    pub(crate) f32_buf: Vec<f32>,
    pub(crate) simd: S,
}

impl<S: Simd, T: FineKernel<S>> Fine<S, T> {
    pub fn new(simd: S) -> Self {
        Self {
            simd,
            wide_coords: (0, 0),
            blend_buf: vec![[T::Numeric::ZERO; SCRATCH_BUF_SIZE]],
            f32_buf: vec![0.0; SCRATCH_BUF_SIZE / 4],
            paint_buf: [T::Numeric::ZERO; SCRATCH_BUF_SIZE],
        }
    }

    pub fn set_coords(&mut self, x: u16, y: u16) {
        self.wide_coords = (x, y);
    }

    pub fn clear(&mut self, premul_color: PremulColor) {
        let converted_color = T::extract_color(premul_color);
        let blend_buf = self.blend_buf.last_mut().unwrap();

        T::copy_solid(self.simd, blend_buf, converted_color);
    }

    pub fn pack(&self, region: &mut Region<'_>) {
        let blend_buf = self.blend_buf.last().unwrap();

        T::pack(self.simd, region, blend_buf);
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u8], paints: &[EncodedPaint]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(
                    usize::from(f.x),
                    usize::from(f.width),
                    &f.paint,
                    f.blend_mode,
                    paints,
                    None,
                    f.mask.as_ref(),
                );
            }
            Cmd::AlphaFill(s) => {
                self.fill(
                    usize::from(s.x),
                    usize::from(s.width),
                    &s.paint,
                    s.blend_mode,
                    paints,
                    Some(&alphas[s.alpha_idx..]),
                    s.mask.as_ref(),
                );
            }
            Cmd::PushBuf => {
                self.blend_buf.push([T::Numeric::ZERO; SCRATCH_BUF_SIZE]);
            }
            Cmd::PopBuf => {
                self.blend_buf.pop();
            }
            Cmd::ClipFill(cf) => {
                self.clip(cf.x as usize, cf.width as usize, None);
            }
            Cmd::ClipStrip(cs) => {
                self.clip(
                    cs.x as usize,
                    cs.width as usize,
                    Some(&alphas[cs.alpha_idx..]),
                );
            }
            Cmd::Blend(b) => self.blend(*b),
            Cmd::Mask(m) => {
                let start_x = self.wide_coords.0 * WideTile::WIDTH;
                let start_y = self.wide_coords.1 * Tile::HEIGHT;

                let blend_buf = self.blend_buf.last_mut().unwrap();

                let width = (blend_buf.len() / (Tile::HEIGHT as usize * COLOR_COMPONENTS)) as u16;
                let y = start_y as u32 + u32x4::from_slice(self.simd, &[0, 1, 2, 3]);

                let iter = (start_x..(start_x + width)).map(|x| {
                    let x_in_range = x < m.width();

                    macro_rules! sample {
                        ($idx:expr) => {
                            if x_in_range && (y[$idx] as u16) < m.height() {
                                m.sample(x, y[$idx] as u16)
                            } else {
                                0
                            }
                        };
                    }

                    let s1 = sample!(0);
                    let s2 = sample!(1);
                    let s3 = sample!(2);
                    let s4 = sample!(3);

                    let samples = u8x16::from_slice(
                        self.simd,
                        &[
                            s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3, s4, s4, s4, s4,
                        ],
                    );
                    T::NumericVec::from_u8(self.simd, samples)
                });

                T::apply_mask(self.simd, blend_buf, iter);
            }
            Cmd::Opacity(o) => {
                if *o != 1.0 {
                    let blend_buf = self.blend_buf.last_mut().unwrap();

                    T::apply_mask(
                        self.simd,
                        blend_buf,
                        iter::repeat(T::NumericVec::from_f32(
                            self.simd,
                            f32x16::splat(self.simd, *o),
                        )),
                    );
                }
            }
        }
    }

    /// Fill at a given x and with a width using the given paint.
    // For short strip segments, benchmarks showed that not inlining leads to significantly
    // worse performance.
    pub fn fill(
        &mut self,
        x: usize,
        width: usize,
        fill: &Paint,
        blend_mode: BlendMode,
        encoded_paints: &[EncodedPaint],
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    ) {
        let blend_buf = &mut self.blend_buf.last_mut().unwrap()[x * TILE_HEIGHT_COMPONENTS..]
            [..TILE_HEIGHT_COMPONENTS * width];
        let default_blend = blend_mode.is_default();

        match fill {
            Paint::Solid(color) => {
                let color = T::extract_color(*color);

                // If color is completely opaque, we can just directly override
                // the blend buffer.
                if color[3] == T::Numeric::ONE
                    && default_blend
                    && alphas.is_none()
                    && mask.is_none()
                {
                    T::copy_solid(self.simd, blend_buf, color);

                    return;
                }

                if default_blend && mask.is_none() {
                    T::alpha_composite_solid(self.simd, blend_buf, color, alphas);
                } else {
                    let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
                    let start_y = self.wide_coords.1 * Tile::HEIGHT;

                    T::blend(
                        self.simd,
                        blend_buf,
                        start_x,
                        start_y,
                        iter::repeat(T::Composite::from_color(self.simd, color)),
                        blend_mode,
                        alphas,
                        mask,
                    );
                }
            }
            Paint::Indexed(paint) => {
                let color_buf = &mut self.paint_buf[x * TILE_HEIGHT_COMPONENTS..]
                    [..TILE_HEIGHT_COMPONENTS * width];

                let encoded_paint = &encoded_paints[paint.index()];

                let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
                let start_y = self.wide_coords.1 * Tile::HEIGHT;

                // We need to have this as a macro because closures cannot take generic arguments, and
                // we would have to repeatedly provide all arguments if we made it a function.
                macro_rules! fill_complex_paint {
                    ($has_opacities:expr, $filler:expr) => {
                        if $has_opacities || alphas.is_some() {
                            T::apply_painter(self.simd, color_buf, $filler);

                            if default_blend && mask.is_none() {
                                T::alpha_composite_buffer(self.simd, blend_buf, color_buf, alphas);
                            } else {
                                T::blend(
                                    self.simd,
                                    blend_buf,
                                    start_x,
                                    start_y,
                                    color_buf
                                        .chunks_exact(T::Composite::LENGTH)
                                        .map(|s| T::Composite::from_slice(self.simd, s)),
                                    blend_mode,
                                    alphas,
                                    mask,
                                );
                            }
                        } else {
                            // Similarly to solid colors we can just override the previous values
                            // if all colors in the gradient are fully opaque.
                            T::apply_painter(self.simd, blend_buf, $filler);
                        }
                    };
                }

                match encoded_paint {
                    EncodedPaint::BlurredRoundedRect(b) => {
                        fill_complex_paint!(
                            true,
                            T::blurred_rounded_rectangle_painter(self.simd, b, start_x, start_y)
                        );
                    }
                    EncodedPaint::Gradient(g) => {
                        // Note that we are calculating the t values first, store them in a separate
                        // buffer and then pass that buffer to the iterator instead of calculating
                        // the t values on the fly in the iterator. The latter would be faster, but
                        // it would probably increase code size a lot, because the functions for
                        // position calculation need to be inlined for good performance.
                        let f32_buf = &mut self.f32_buf[..width * Tile::HEIGHT as usize];

                        match &g.kind {
                            EncodedKind::Linear(l) => {
                                calculate_t_vals(
                                    self.simd,
                                    SimdLinearKind::new(self.simd, *l),
                                    f32_buf,
                                    g,
                                    start_x,
                                    start_y,
                                );

                                fill_complex_paint!(
                                    g.has_opacities,
                                    T::gradient_painter(self.simd, g, f32_buf)
                                );
                            }
                            EncodedKind::Sweep(s) => {
                                calculate_t_vals(
                                    self.simd,
                                    SimdSweepKind::new(self.simd, s),
                                    f32_buf,
                                    g,
                                    start_x,
                                    start_y,
                                );

                                fill_complex_paint!(
                                    g.has_opacities,
                                    T::gradient_painter(self.simd, g, f32_buf)
                                );
                            }
                            EncodedKind::Radial(r) => {
                                calculate_t_vals(
                                    self.simd,
                                    SimdRadialKind::new(self.simd, r),
                                    f32_buf,
                                    g,
                                    start_x,
                                    start_y,
                                );

                                if r.has_undefined() {
                                    fill_complex_paint!(
                                        g.has_opacities,
                                        T::gradient_painter_with_undefined(self.simd, g, f32_buf)
                                    );
                                } else {
                                    fill_complex_paint!(
                                        g.has_opacities,
                                        T::gradient_painter(self.simd, g, f32_buf)
                                    );
                                }
                            }
                        }
                    }
                    EncodedPaint::Image(i) => {
                        let ImageSource::Pixmap(pixmap) = &i.source else {
                            panic!("vello_cpu doesn't support the opaque image source.");
                        };

                        match (i.has_skew(), i.nearest_neighbor()) {
                            (_, false) => {
                                if i.sampler.quality == ImageQuality::Medium {
                                    fill_complex_paint!(
                                        i.has_opacities,
                                        T::medium_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                } else {
                                    fill_complex_paint!(
                                        i.has_opacities,
                                        T::high_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                }
                            }
                            (false, true) => {
                                fill_complex_paint!(
                                    i.has_opacities,
                                    T::plain_nn_image_painter(
                                        self.simd, i, pixmap, start_x, start_y
                                    )
                                );
                            }
                            (true, true) => {
                                fill_complex_paint!(
                                    i.has_opacities,
                                    T::nn_image_painter(self.simd, i, pixmap, start_x, start_y)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn blend(&mut self, blend_mode: BlendMode) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        if blend_mode.is_default() {
            T::alpha_composite_buffer(self.simd, target_buffer, source_buffer, None);
        } else {
            T::blend(
                self.simd,
                target_buffer,
                // `start_x` and `start_y` are only needed to sample the correct position
                // of a mask, so we can just pass dummy values here.
                0,
                0,
                source_buffer
                    .chunks_exact(T::Composite::LENGTH)
                    .map(|s| T::Composite::from_slice(self.simd, s)),
                blend_mode,
                None,
                None,
            );
        }
    }

    fn clip(&mut self, x: usize, width: usize, alphas: Option<&[u8]>) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        let source_buffer =
            &mut source_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];
        let target_buffer =
            &mut target_buffer[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        T::alpha_composite_buffer(self.simd, target_buffer, source_buffer, alphas);
    }
}

/// A trait for shaders that can render their contents into a u8/f32 buffer. Note that while
/// the trait has a method for both, f32 and u8, some shaders might only support 1 of them, so
/// care is needed when using them.
pub trait Painter {
    fn paint_u8(&mut self, buf: &mut [u8]);
    fn paint_f32(&mut self, buf: &mut [f32]);
}

/// Calculate the x/y position using the x/y advances for each pixel, assuming a tile height of 4.
pub trait PosExt<S: Simd> {
    fn splat_pos(simd: S, pos: f32, x_advance: f32, y_advance: f32) -> Self;
}

impl<S: Simd> PosExt<S> for f32x4<S> {
    #[inline(always)]
    fn splat_pos(simd: S, pos: f32, _: f32, y_advance: f32) -> Self {
        let columns: [f32; Tile::HEIGHT as usize] = [0.0, 1.0, 2.0, 3.0];
        let column_mask: Self = columns.simd_into(simd);

        column_mask.madd(Self::splat(simd, y_advance), Self::splat(simd, pos))
    }
}

impl<S: Simd> PosExt<S> for f32x8<S> {
    #[inline(always)]
    fn splat_pos(simd: S, pos: f32, x_advance: f32, y_advance: f32) -> Self {
        simd.combine_f32x4(
            f32x4::splat_pos(simd, pos, x_advance, y_advance),
            f32x4::splat_pos(simd, pos + x_advance, x_advance, y_advance),
        )
    }
}

/// The results of an f32 shader, where each channel stored separately.
pub(crate) struct ShaderResultF32<S: Simd> {
    pub(crate) r: f32x8<S>,
    pub(crate) g: f32x8<S>,
    pub(crate) b: f32x8<S>,
    pub(crate) a: f32x8<S>,
}

impl<S: Simd> ShaderResultF32<S> {
    /// Convert the result into two f32x16 elements, interleaved as RGBA.
    #[inline(always)]
    pub(crate) fn get(&self) -> (f32x16<S>, f32x16<S>) {
        let (r_1, r_2) = self.r.simd.split_f32x8(self.r);
        let (g_1, g_2) = self.g.simd.split_f32x8(self.g);
        let (b_1, b_2) = self.b.simd.split_f32x8(self.b);
        let (a_1, a_2) = self.a.simd.split_f32x8(self.a);

        let first = self.r.simd.combine_f32x8(
            self.r.simd.combine_f32x4(r_1, g_1),
            self.r.simd.combine_f32x4(b_1, a_1),
        );

        let second = self.r.simd.combine_f32x8(
            self.r.simd.combine_f32x4(r_2, g_2),
            self.r.simd.combine_f32x4(b_2, a_2),
        );

        (first, second)
    }
}

mod macros {
    /// The default `Painter` implementation for an iterator
    /// that returns its results as f32x16.
    macro_rules! f32x16_painter {
        ($($type_path:tt)+) => {
            impl<S: Simd> crate::fine::Painter for $($type_path)+ {
                fn paint_u8(&mut self, buf: &mut [u8]) {
                    use vello_common::fearless_simd::*;
                    use crate::fine::NumericVec;

                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            let converted = u8x16::<S>::from_f32(next.simd, next);
                            chunk.copy_from_slice(&converted.val);
                        }
                    })
                }

                fn paint_f32(&mut self, buf: &mut [f32]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            chunk.copy_from_slice(&next.val);
                        }
                    })
                }
            }
        };
    }

    /// The default `Painter` implementation for an iterator
    /// that returns its results as u8x16.
    macro_rules! u8x16_painter {
        ($($type_path:tt)+) => {
            impl<S: Simd> crate::fine::Painter for $($type_path)+ {
                fn paint_u8(&mut self, buf: &mut [u8]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            chunk.copy_from_slice(&next.val);
                        }
                    })
                }

                fn paint_f32(&mut self, buf: &mut [f32]) {
                    use vello_common::fearless_simd::*;
                    use crate::fine::NumericVec;

                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            let converted = f32x16::<S>::from_u8(next.simd, next);
                            chunk.copy_from_slice(&converted.val);
                        }
                    })
                }
            }
        };
    }

    pub(crate) use f32x16_painter;
    pub(crate) use u8x16_painter;
}
