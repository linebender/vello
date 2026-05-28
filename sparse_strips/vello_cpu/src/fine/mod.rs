// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization stage of the rendering pipeline.
//!
//! This module implements the fine rasterization phase, which processes tiles at the pixel level.
//! It supports both high-precision (f32) and low-precision (u8) rendering paths, along with
//! various paint types including solid colors, gradients, images, and blurred rounded rectangles.

mod common;
mod highp;
mod lowp;

use crate::fine::common::gradient::GradientPainter;
pub(crate) use crate::fine::common::gradient::calculate_t_vals;
pub(crate) use crate::fine::common::gradient::linear::SimdLinearKind;
pub(crate) use crate::fine::common::gradient::radial::SimdRadialKind;
pub(crate) use crate::fine::common::gradient::sweep::SimdSweepKind;
use crate::fine::common::image::{FilteredImagePainter, NNImagePainter, PlainNNImagePainter};
use crate::fine::common::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::layer_manager::LayerManager;
use crate::peniko::{BlendMode, ImageQuality};
use crate::row::{Cmd, CommandBucketer, FillAttrs, FillCmd};
use crate::util::EncodedImageExt;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use vello_common::encode::{
    EncodedBlurredRoundedRectangle, EncodedGradient, EncodedImage, EncodedKind, EncodedPaint,
};
use vello_common::fearless_simd::{
    Bytes, Simd, SimdBase, SimdFloat, SimdInt, SimdInto, f32x4, f32x8, f32x16, u8x16, u8x32, u32x4,
    u32x8,
};
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, ImageSource, Paint, PremulColor, Tint};
use vello_common::pixmap::Pixmap;
use vello_common::simd::Splat4thExt;
use vello_common::tile::Tile;
use vello_common::util::f32_to_u8;

pub use highp::F32Kernel;
pub use lowp::U8Kernel;

/// Offset to shift from pixel corner to pixel center for sampling.
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const DEPTH_BUCKET_WIDTH: u16 = 32;

/// Number of color components per pixel (RGBA).
pub(crate) const COLOR_COMPONENTS: usize = 4;

/// Number of color components in a single column of a tile (height * components).
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;

/// Trait for numeric types used in fine rasterization.
///
/// This trait abstracts over `f32` and `u8` to allow the same rendering logic
/// to work with both high-precision (floating-point) and low-precision (integer)
/// representations. This enables performance optimizations while maintaining accuracy
/// where needed.
pub trait Numeric: Copy + Default + Clone + Debug + PartialEq + Send + Sync + 'static {
    /// The zero value for this numeric type (0.0 for f32, 0 for u8).
    const ZERO: Self;

    /// The maximum opacity value for this numeric type (1.0 for f32, 255 for u8).
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

/// Trait for SIMD vector types that can convert between f32 and u8 representations.
///
/// This trait enables efficient batch conversions between different numeric representations
/// during rendering operations, supporting both high-precision and low-precision rendering paths.
pub trait NumericVec<S: Simd>: Copy + Clone + Send + Sync {
    /// Convert from a SIMD vector of f32 values to this type.
    fn from_f32(simd: S, val: f32x16<S>) -> Self;

    /// Convert from a SIMD vector of u8 values to this type.
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
        let mulled = val.mul_add(v1, v2);

        f32_to_u8(mulled)
    }

    #[inline(always)]
    fn from_u8(_: S, val: Self) -> Self {
        val
    }
}

/// Convert a SIMD vector of u8 values to f32 values.
///
/// This function efficiently converts 16 u8 values to their f32 equivalents using SIMD operations,
/// preserving the values without normalization (i.e., 255 becomes 255.0, not 1.0).
#[inline(always)]
pub(crate) fn u8_to_f32<S: Simd>(val: u8x16<S>) -> f32x16<S> {
    let simd = val.simd;
    let zeroes = u8x16::splat(simd, 0);

    let zip1 = simd.zip_high_u8x16(val, zeroes);
    let zip2 = simd.zip_low_u8x16(val, zeroes);

    let p1 = simd
        .zip_low_u8x16(zip2, zeroes)
        .bitcast::<u32x4<S>>()
        .to_float::<f32x4<S>>();
    let p2 = simd
        .zip_high_u8x16(zip2, zeroes)
        .bitcast::<u32x4<S>>()
        .to_float::<f32x4<S>>();
    let p3 = simd
        .zip_low_u8x16(zip1, zeroes)
        .bitcast::<u32x4<S>>()
        .to_float::<f32x4<S>>();
    let p4 = simd
        .zip_high_u8x16(zip1, zeroes)
        .bitcast::<u32x4<S>>()
        .to_float::<f32x4<S>>();

    simd.combine_f32x8(simd.combine_f32x4(p1, p2), simd.combine_f32x4(p3, p4))
}

/// Trait for SIMD vector types used in compositing and blending operations.
///
/// This trait abstracts over different SIMD vector widths (f32x16 for high-precision,
/// u8x32 for low-precision) to enable efficient batch processing of pixel data during
/// blending and compositing.
pub trait CompositeType<N: Numeric, S: Simd>: Copy + Clone + Send + Sync {
    /// The number of numeric values this composite type can hold.
    const LENGTH: usize;

    /// Load values from a slice into this composite type.
    fn from_slice(simd: S, slice: &[N]) -> Self;

    /// Create a composite type by repeating a single RGBA color across all elements.
    fn from_color(simd: S, color: [N; 4]) -> Self;
}

impl<S: Simd> CompositeType<f32, S> for f32x16<S> {
    const LENGTH: usize = 16;

    #[inline(always)]
    fn from_slice(simd: S, slice: &[f32]) -> Self {
        <Self as SimdBase<_>>::from_slice(simd, slice)
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
        <Self as SimdBase<_>>::from_slice(simd, slice)
    }

    #[inline(always)]
    fn from_color(simd: S, color: [u8; 4]) -> Self {
        u32x8::block_splat(u32x4::splat(simd, u32::from_ne_bytes(color))).to_bytes()
    }
}

/// A kernel for performing fine rasterization.
///
/// This trait defines the interface for tile-level rendering operations, abstracting over
/// different numeric precisions (f32 vs u8). Implementations provide the low-level pixel
/// manipulation, blending, and painting operations needed to render tiles.
///
/// The two main implementations are:
/// - [`F32Kernel`]: High-precision rendering using 32-bit floating-point values
/// - [`U8Kernel`]: Low-precision rendering using 8-bit integer values
pub trait FineKernel<S: Simd>: Send + Sync + 'static {
    /// The basic underlying numerical type of the kernel (f32 or u8).
    type Numeric: Numeric;

    /// The SIMD composite type used for efficient batch blending and compositing operations.
    type Composite: CompositeType<Self::Numeric, S>;

    /// The SIMD vector type used for conversions between u8 and f32 representations.
    type NumericVec: NumericVec<S>;

    /// Extract and convert a premultiplied color to the kernel's numeric type.
    ///
    /// Converts RGBA components from the standard premultiplied color format to
    /// the kernel's internal representation (e.g., 0.0-1.0 for f32, 0-255 for u8).
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4];

    /// Apply a filter to a layer.
    ///
    /// This is used for applying filters to whole layers, which is necessary for
    /// spatial filters (like blur) that need to access neighboring pixels. The filter
    /// is applied in-place to the provided pixmap.
    ///
    /// The transform parameter is used to scale filter parameters based on the current
    /// transformation matrix (e.g., zoom level), ensuring filters look consistent
    /// regardless of scale.
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        layer_manager: &mut LayerManager,
        transform: Affine,
    );

    /// Fill the target buffer with a solid color.
    ///
    /// Efficiently replicates the given RGBA color across all pixels in the target buffer.
    fn copy_solid(simd: S, target: &mut [Self::Numeric], color: [Self::Numeric; 4]);
    /// Create a painter for rendering gradients.
    ///
    /// Returns a painter that can render linear, radial, or sweep gradients based on
    /// pre-computed t values (gradient interpolation parameters).
    fn gradient_painter<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, t_vals),
        )
    }

    /// Create a painter for rendering gradients with undefined region support.
    ///
    /// Similar to `gradient_painter`, but with support for masking undefined locations
    /// (used for radial gradients that may have mathematically undefined regions).
    ///
    /// This is intentionally a duplicate of the default [`FineKernel::gradient_painter`]
    /// implementation--the `U8Kernel` overrides that method, but not this one.
    fn gradient_painter_with_undefined<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, t_vals),
        )
    }
    /// Create a painter for rendering axis-aligned nearest-neighbor images.
    ///
    /// Optimized painter for images with `Low` quality and no skewing component in their
    /// transform. This is the fastest image rendering path.
    fn plain_nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || PlainNNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Create a painter for rendering nearest-neighbor images with transforms.
    ///
    /// Similar to `plain_nn_image_painter`, but supports arbitrary affine transforms
    /// including skewing and rotation.
    fn nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || NNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Create a painter for rendering images with `Medium` quality filtering.
    ///
    /// Uses bilinear filtering for smoother appearance than nearest-neighbor.
    fn medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 1>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Create a painter for rendering axis-aligned images with `Medium` quality filtering.
    ///
    /// Optimized painter for images with bilinear filtering and no skewing component.
    fn plain_medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 1>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Create a painter for rendering images with `High` quality filtering.
    ///
    /// Uses high-quality filtering for the best visual appearance.
    fn high_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 2>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    /// Create a painter for rendering blurred rounded rectangles.
    ///
    /// Efficiently renders rounded rectangles with gaussian blur applied,
    /// computing the blur analytically rather than as a post-process.
    fn blurred_rounded_rectangle_painter(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter {
        simd.vectorize(
            #[inline(always)]
            || BlurredRoundedRectFiller::new(simd, rect, start_x, start_y),
        )
    }
    /// Apply a mask to the destination buffer.
    ///
    /// Multiplies each pixel in the destination by the corresponding mask value,
    /// effectively masking out or reducing the opacity of pixels.
    fn apply_mask(simd: S, dest: &mut [Self::Numeric], src: impl Iterator<Item = Self::NumericVec>);

    /// Apply a painter to render content into the destination buffer.
    ///
    /// Invokes the painter to generate pixel values and writes them to the destination.
    fn apply_painter<'a>(simd: S, dest: &mut [Self::Numeric], painter: impl Painter + 'a);

    /// Apply an image tint to an already-painted buffer.
    ///
    /// This is called as a post-pass after `apply_painter`, only when a tint is
    /// present. Keeping tint application out of the per-pixel iterator avoids
    /// regressing the non-tinted fast path.
    fn apply_tint(simd: S, dest: &mut [Self::Numeric], tint: &Tint);

    /// Perform alpha compositing with a solid color over the target buffer.
    ///
    /// Blends a solid RGBA color over the existing contents using standard alpha compositing
    /// (Porter-Duff source-over). Optionally applies additional per-pixel alpha values.
    fn alpha_composite_solid(
        simd: S,
        target: &mut [Self::Numeric],
        src: [Self::Numeric; 4],
        alphas: Option<&[u8]>,
    );

    /// Perform alpha compositing with a source buffer over the destination buffer.
    ///
    /// Blends the source buffer contents over the destination using standard alpha compositing.
    /// Optionally applies additional per-pixel alpha values.
    fn alpha_composite_buffer(
        simd: S,
        dest: &mut [Self::Numeric],
        src: &[Self::Numeric],
        alphas: Option<&[u8]>,
    );

    /// Blend the source into the destination with a specified blend mode.
    ///
    /// Applies advanced blending operations (e.g., multiply, screen, overlay) as specified
    /// by the blend mode. Optionally applies additional per-pixel alpha values.
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

    /// Fill a row scratch span with a solid color, optionally modulated by per-pixel alphas.
    #[inline(always)]
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>) {
        let color = Self::extract_color(color);
        if color[3] == Self::Numeric::ONE && alphas.is_none() {
            Self::copy_solid(simd, dest, color);
        } else {
            Self::alpha_composite_solid(simd, dest, color, alphas);
        }
    }

    /// Pack tile-aligned row scratch blocks into a row-major output buffer.
    fn pack_block(
        simd: S,
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        out_width: usize,
        out: &mut [u8],
    );

    /// Pack an arbitrary-width/height row scratch tail into a row-major output buffer.
    fn pack_tail(
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        height: usize,
        out_width: usize,
        out: &mut [u8],
    );
}

#[derive(Debug)]
pub(crate) struct Fine<S: Simd, T: FineKernel<S>> {
    simd: S,
    out_width: u16,
    buffer_width: u16,
    buffers: Vec<Vec<T::Numeric>>,
    buffer_pool: Vec<Vec<T::Numeric>>,
    paint_buf: Vec<T::Numeric>,
    f32_buf: Vec<f32>,
    depth: Vec<u32>,
}

impl<S: Simd, T: FineKernel<S>> Fine<S, T> {
    pub(crate) fn new(simd: S, out_width: u16, buffer_width: u16) -> Self {
        let scratch_len = usize::from(buffer_width) * TILE_HEIGHT_COMPONENTS;
        Self {
            simd,
            out_width,
            buffer_width,
            buffers: vec![vec![T::Numeric::ZERO; scratch_len]],
            buffer_pool: Vec::new(),
            paint_buf: Vec::new(),
            f32_buf: Vec::new(),
            depth: vec![0; usize::from(buffer_width.div_ceil(DEPTH_BUCKET_WIDTH))],
        }
    }

    fn clear_range(&mut self, x: u16, width: u16) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let depth_start = usize::from(x / DEPTH_BUCKET_WIDTH);
        let depth_end = ((usize::from(x) + usize::from(width))
            .div_ceil(usize::from(DEPTH_BUCKET_WIDTH)))
        .min(self.depth.len());

        self.buffers[0][scratch_start..scratch_start + scratch_len].fill(T::Numeric::ZERO);
        self.depth[depth_start..depth_end].fill(0);
    }

    fn push_layer(&mut self, x: u16, width: u16) {
        let mut buf = self
            .buffer_pool
            .pop()
            .unwrap_or_else(|| vec![T::Numeric::ZERO; self.buffers[0].len()]);
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        buf[scratch_start..scratch_start + scratch_len].fill(T::Numeric::ZERO);
        self.buffers.push(buf);
    }

    fn opacity(&mut self, x: u16, width: u16, opacity: f32) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let target = self.buffers.last_mut().unwrap();
        let target = &mut target[scratch_start..scratch_start + scratch_len];

        T::apply_mask(
            self.simd,
            target,
            iter::repeat(T::NumericVec::from_f32(
                self.simd,
                f32x16::splat(self.simd, opacity),
            )),
        );
    }

    fn mask(&mut self, row_y: u16, x: u16, width: u16, mask: &Mask) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let target = self.buffers.last_mut().unwrap();
        let target = &mut target[scratch_start..scratch_start + scratch_len];

        Self::apply_mask(self.simd, target, x, row_y, width, mask);
    }

    fn blend_fill(&mut self, row_y: u16, x: u16, width: u16, blend_mode: BlendMode) {
        self.blend(row_y, x, width, blend_mode, None);
    }

    fn blend_alpha_fill(
        &mut self,
        row_y: u16,
        x: u16,
        width: u16,
        blend_mode: BlendMode,
        alphas: &[u8],
    ) {
        self.blend(row_y, x, width, blend_mode, Some(alphas));
    }

    fn blend(
        &mut self,
        row_y: u16,
        x: u16,
        width: u16,
        blend_mode: BlendMode,
        alphas: Option<&[u8]>,
    ) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let (source, rest) = self.buffers.split_last_mut().unwrap();
        let target = rest.last_mut().unwrap();
        let source = &mut source[scratch_start..scratch_start + scratch_len];
        let target = &mut target[scratch_start..scratch_start + scratch_len];

        if blend_mode == BlendMode::default() {
            T::alpha_composite_buffer(self.simd, target, source, alphas);
        } else {
            T::blend(
                self.simd,
                target,
                x,
                row_y,
                source
                    .chunks_exact(T::Composite::LENGTH)
                    .map(|s| T::Composite::from_slice(self.simd, s)),
                blend_mode,
                alphas,
                None,
            );
        }
    }

    fn pop_buf(&mut self) {
        let popped = self.buffers.pop().unwrap();
        self.buffer_pool.push(popped);
    }

    fn apply_mask(simd: S, target: &mut [T::Numeric], x: u16, y: u16, width: u16, mask: &Mask) {
        let y = u32::from(y) + u32x4::from_slice(simd, &[0, 1, 2, 3]);
        let iter = (x..x.saturating_add(width)).map(|x| {
            let x_in_range = x < mask.width();

            macro_rules! sample {
                ($idx:expr) => {
                    if x_in_range && (y[$idx] as u16) < mask.height() {
                        mask.sample(x, y[$idx] as u16)
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
                simd,
                &[
                    s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3, s4, s4, s4, s4,
                ],
            );
            T::NumericVec::from_u8(simd, samples)
        });

        T::apply_mask(simd, target, iter);
    }

    #[inline(always)]
    fn fill_solid(&mut self, x: u16, width: u16, color: PremulColor, alphas: Option<&[u8]>) {
        if width == 0 {
            return;
        }

        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let simd = self.simd;
        let scratch = self.buffers.last_mut().unwrap();
        T::fill_solid(simd, &mut scratch[start..start + len], color, alphas);
    }

    #[inline(always)]
    fn fill_solid_with_attrs(
        &mut self,
        x: u16,
        y: u16,
        width: u16,
        color: PremulColor,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        alphas: Option<&[u8]>,
    ) {
        if blend_mode == BlendMode::default() && mask.is_none() {
            self.fill_solid(x, width, color, alphas);
            return;
        }

        if width == 0 {
            return;
        }

        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let color = T::extract_color(color);
        let simd = self.simd;
        let color = T::Composite::from_color(simd, color);
        let scratch = self.buffers.last_mut().unwrap();
        T::blend(
            simd,
            &mut scratch[start..start + len],
            x,
            y,
            iter::repeat(color),
            blend_mode,
            alphas,
            mask,
        );
    }

    fn fill_indexed(
        &mut self,
        x: u16,
        y: u16,
        width: u16,
        paint_index: usize,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        alphas: Option<&[u8]>,
    ) {
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        if self.paint_buf.len() < len {
            self.paint_buf.resize(len, T::Numeric::ZERO);
        }

        let t_len = usize::from(width) * Tile::HEIGHT as usize;
        if self.f32_buf.len() < t_len {
            self.f32_buf.resize(t_len, 0.0);
        }

        let scratch = self.buffers.last_mut().unwrap();
        fill_indexed_paint::<S, T>(
            self.simd,
            scratch,
            &mut self.paint_buf,
            &mut self.f32_buf,
            x,
            y,
            width,
            paint_index,
            blend_mode,
            mask,
            encoded_paints,
            image_resolver,
            alphas,
        );
    }

    #[inline(always)]
    fn render_opaque(
        &mut self,
        cmd: FillCmd,
        row_y: u16,
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let start = usize::from(cmd.x / DEPTH_BUCKET_WIDTH);
        let end = usize::from((cmd.x + cmd.width) / DEPTH_BUCKET_WIDTH);

        if start + 1 == end {
            if self.depth[start] == 0 {
                match &attrs.paint {
                    Paint::Solid(color) => self.fill_solid(cmd.x, cmd.width, *color, None),
                    Paint::Indexed(index) => self.fill_indexed(
                        cmd.x,
                        row_y,
                        cmd.width,
                        index.index(),
                        attrs.blend_mode,
                        attrs.mask.as_ref(),
                        encoded_paints,
                        image_resolver,
                        None,
                    ),
                }
                self.depth[start] = attrs.path_id;
            }
            return;
        }

        let mut idx = start;
        while idx < end {
            while idx < end && self.depth[idx] != 0 {
                idx += 1;
            }

            let run_start = idx;
            while idx < end && self.depth[idx] == 0 {
                idx += 1;
            }

            if run_start == idx {
                continue;
            }

            let x = (run_start as u16) * DEPTH_BUCKET_WIDTH;
            let width = (idx - run_start) as u16 * DEPTH_BUCKET_WIDTH;
            match &attrs.paint {
                Paint::Solid(color) => self.fill_solid(x, width, *color, None),
                Paint::Indexed(index) => self.fill_indexed(
                    x,
                    row_y,
                    width,
                    index.index(),
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    encoded_paints,
                    image_resolver,
                    None,
                ),
            }
            for depth in &mut self.depth[run_start..idx] {
                *depth = attrs.path_id;
            }
        }
    }

    #[inline(always)]
    fn render_cmd(
        &mut self,
        cmd: &Cmd,
        row_y: u16,
        alphas: &[u8],
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        use_depth: bool,
    ) {
        let cmd_x = cmd.fill_x();
        let cmd_end = (cmd_x + cmd.fill_width()).min(self.buffer_width);
        if cmd_x >= cmd_end {
            return;
        }

        if !use_depth {
            self.render_cmd_span(
                cmd,
                cmd_x,
                cmd_end,
                row_y,
                alphas,
                attrs,
                encoded_paints,
                image_resolver,
            );
            return;
        }

        let start = usize::from(cmd_x / DEPTH_BUCKET_WIDTH);
        let end = usize::from(cmd_end.div_ceil(DEPTH_BUCKET_WIDTH));

        if start + 1 == end {
            if self.depth[start] <= attrs.path_id {
                self.render_cmd_span(
                    cmd,
                    cmd_x,
                    cmd_end,
                    row_y,
                    alphas,
                    attrs,
                    encoded_paints,
                    image_resolver,
                );
            }
            return;
        }

        let mut idx = start;
        while idx < end {
            while idx < end && self.depth[idx] > attrs.path_id {
                idx += 1;
            }

            let run_start = idx;
            while idx < end && self.depth[idx] <= attrs.path_id {
                idx += 1;
            }

            if run_start == idx {
                continue;
            }

            let x = cmd_x.max(run_start as u16 * DEPTH_BUCKET_WIDTH);
            let end = cmd_end.min(idx as u16 * DEPTH_BUCKET_WIDTH);
            self.render_cmd_span(
                cmd,
                x,
                end,
                row_y,
                alphas,
                attrs,
                encoded_paints,
                image_resolver,
            );
        }
    }

    #[inline(always)]
    fn render_cmd_span(
        &mut self,
        cmd: &Cmd,
        x: u16,
        end: u16,
        row_y: u16,
        alphas: &[u8],
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        match cmd {
            Cmd::Fill(_) => match &attrs.paint {
                Paint::Solid(color) => self.fill_solid_with_attrs(
                    x,
                    row_y,
                    end - x,
                    *color,
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    None,
                ),
                Paint::Indexed(index) => self.fill_indexed(
                    x,
                    row_y,
                    end - x,
                    index.index(),
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    encoded_paints,
                    image_resolver,
                    None,
                ),
            },
            Cmd::AlphaFill(fill) => {
                let alpha_offset =
                    fill.alpha_idx as usize + usize::from(x - fill.x) * Tile::HEIGHT as usize;
                match &attrs.paint {
                    Paint::Solid(color) => self.fill_solid_with_attrs(
                        x,
                        row_y,
                        end - x,
                        *color,
                        attrs.blend_mode,
                        attrs.mask.as_ref(),
                        Some(&alphas[alpha_offset..]),
                    ),
                    Paint::Indexed(index) => {
                        self.fill_indexed(
                            x,
                            row_y,
                            end - x,
                            index.index(),
                            attrs.blend_mode,
                            attrs.mask.as_ref(),
                            encoded_paints,
                            image_resolver,
                            Some(&alphas[alpha_offset..]),
                        );
                    }
                }
            }
            Cmd::PushLayer
            | Cmd::PopBuf
            | Cmd::Opacity(_)
            | Cmd::Mask(_)
            | Cmd::BlendFill(_)
            | Cmd::BlendAlphaFill(_) => unreachable!(),
        }
    }

    fn pack(&self, row_idx: usize, row_height: usize, x: u16, width: u16, buffer: &mut [u8]) {
        let out_width = usize::from(self.out_width);
        let offset = row_idx * Tile::HEIGHT as usize * out_width * COLOR_COMPONENTS;
        let len = row_height * out_width * COLOR_COMPONENTS;
        let out = &mut buffer[offset..offset + len];
        let x = usize::from(x);
        let width = usize::from(width);
        let end = x + width;
        let scratch = self.buffers.last().unwrap();

        let block_start = if row_height == Tile::HEIGHT as usize {
            x.next_multiple_of(Tile::WIDTH as usize).min(end)
        } else {
            end
        };

        if x < block_start {
            T::pack_tail(scratch, x, block_start - x, row_height, out_width, out);
        }

        let block_width = if row_height == Tile::HEIGHT as usize {
            (end - block_start) / Tile::WIDTH as usize * Tile::WIDTH as usize
        } else {
            0
        };

        if block_width > 0 {
            T::pack_block(self.simd, scratch, block_start, block_width, out_width, out);
        }

        let tail_start = block_start + block_width;
        if tail_start < end {
            T::pack_tail(
                scratch,
                tail_start,
                end - tail_start,
                row_height,
                out_width,
                out,
            );
        }
    }

}

fn fill_indexed_paint<S: Simd, T: FineKernel<S>>(
    simd: S,
    scratch: &mut [T::Numeric],
    paint_buf: &mut [T::Numeric],
    f32_buf: &mut [f32],
    x: u16,
    y: u16,
    width: u16,
    paint_index: usize,
    blend_mode: BlendMode,
    mask: Option<&Mask>,
    encoded_paints: &[EncodedPaint],
    image_resolver: &dyn ImageResolver,
    alphas: Option<&[u8]>,
) {
    if width == 0 {
        return;
    }

    let width = usize::from(width);
    let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
    let len = width * TILE_HEIGHT_COMPONENTS;
    let dest = &mut scratch[start..start + len];
    let color_buf = &mut paint_buf[..len];
    let encoded_paint = &encoded_paints[paint_index];

    let sampler_x = f64::from(x) + PIXEL_CENTER_OFFSET;
    let sampler_y = f64::from(y) + PIXEL_CENTER_OFFSET;
    let default_blend = blend_mode == BlendMode::default();

    macro_rules! fill_complex_paint {
        ($may_have_transparency:expr, $filler:expr) => {
            fill_complex_paint!($may_have_transparency, $filler, None::<&Tint>)
        };
        ($may_have_transparency:expr, $filler:expr, $tint:expr) => {
            if $may_have_transparency || alphas.is_some() || !default_blend || mask.is_some() {
                T::apply_painter(simd, color_buf, $filler);
                if let Some(tint) = $tint {
                    T::apply_tint(simd, color_buf, tint);
                }

                if default_blend && mask.is_none() {
                    T::alpha_composite_buffer(simd, dest, color_buf, alphas);
                } else {
                    T::blend(
                        simd,
                        dest,
                        x,
                        y,
                        color_buf
                            .chunks_exact(T::Composite::LENGTH)
                            .map(|s| T::Composite::from_slice(simd, s)),
                        blend_mode,
                        alphas,
                        mask,
                    );
                }
            } else {
                T::apply_painter(simd, dest, $filler);
                if let Some(tint) = $tint {
                    T::apply_tint(simd, dest, tint);
                }
            }
        };
    }

    match encoded_paint {
        EncodedPaint::BlurredRoundedRect(rect) => {
            fill_complex_paint!(
                true,
                T::blurred_rounded_rectangle_painter(simd, rect, sampler_x, sampler_y)
            );
        }
        EncodedPaint::Gradient(gradient) => {
            let t_vals = &mut f32_buf[..width * Tile::HEIGHT as usize];

            match &gradient.kind {
                EncodedKind::Linear(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdLinearKind::new(simd, *kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );
                    fill_complex_paint!(
                        gradient.may_have_transparency,
                        T::gradient_painter(simd, gradient, t_vals)
                    );
                }
                EncodedKind::Sweep(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdSweepKind::new(simd, kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );
                    fill_complex_paint!(
                        gradient.may_have_transparency,
                        T::gradient_painter(simd, gradient, t_vals)
                    );
                }
                EncodedKind::Radial(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdRadialKind::new(simd, kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );

                    if kind.has_undefined() {
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter_with_undefined(simd, gradient, t_vals)
                        );
                    } else {
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter(simd, gradient, t_vals)
                        );
                    }
                }
            }
        }
        EncodedPaint::Image(image) => {
            let pixmap = match &image.source {
                ImageSource::Pixmap(pixmap) => pixmap.clone(),
                ImageSource::OpaqueId { id, .. } => image_resolver
                    .resolve(*id)
                    .unwrap_or_else(|| panic!("Image {:?} not found in registry", id)),
            };
            let tint = image.tint.as_ref();

            match (image.has_skew(), image.nearest_neighbor()) {
                (false, false) => {
                    if image.sampler.quality == ImageQuality::Medium {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::plain_medium_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    } else {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::high_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    }
                }
                (true, false) => {
                    if image.sampler.quality == ImageQuality::Medium {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::medium_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    } else {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::high_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    }
                }
                (false, true) => {
                    fill_complex_paint!(
                        image.may_have_transparency,
                        T::plain_nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                        tint
                    );
                }
                (true, true) => {
                    fill_complex_paint!(
                        image.may_have_transparency,
                        T::nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                        tint
                    );
                }
            }
        }
        EncodedPaint::ExternalTexture(_) => {
            unimplemented!("External textures are not supported by `vello_cpu`")
        }
    }
}

pub(crate) fn rasterize<S: Simd, T: FineKernel<S>>(
    simd: S,
    bucketer: &CommandBucketer,
    alpha_buffers: &[&[u8]],
    buffer: &mut [u8],
    width: u16,
    height: u16,
    encoded_paints: &[EncodedPaint],
    image_resolver: &dyn ImageResolver,
) {
    let mut fine = Fine::<S, T>::new(simd, width, bucketer.width());
    buffer.fill(0);

    for (row_idx, row) in bucketer.rows().iter().enumerate() {
        let row_y = row_idx as u16 * Tile::HEIGHT;
        if row_y >= height {
            break;
        }

        let row_height = usize::from((height - row_y).min(Tile::HEIGHT));
        let Some((row_start, row_end)) = row.bounds() else {
            continue;
        };

        fine.clear_range(row_start, row_end - row_start);

        for &cmd in row.opaque.iter().rev() {
            let attrs = &bucketer.attrs()[cmd.attrs_idx as usize];
            fine.render_opaque(cmd, row_y, attrs, encoded_paints, image_resolver);
        }
        for cmd in &row.cmds {
            match cmd {
                Cmd::Fill(_) | Cmd::AlphaFill(_) => {
                    let attrs = &bucketer.attrs()[cmd.fill_attrs_idx() as usize];
                    let alphas = alpha_buffers[attrs.thread_idx as usize];
                    let use_depth =
                        row.depth_affects(cmd.fill_x(), cmd.fill_width(), attrs.path_id);
                    fine.render_cmd(
                        cmd,
                        row_y,
                        alphas,
                        attrs,
                        encoded_paints,
                        image_resolver,
                        use_depth,
                    );
                }
                Cmd::PushLayer => {
                    fine.push_layer(row_start, row_end - row_start);
                }
                Cmd::PopBuf => {
                    fine.pop_buf();
                }
                Cmd::Opacity(opacity) => {
                    fine.opacity(row_start, row_end - row_start, *opacity);
                }
                Cmd::Mask(mask) => {
                    fine.mask(row_y, row_start, row_end - row_start, mask);
                }
                Cmd::BlendFill(cmd) => {
                    fine.blend_fill(row_y, cmd.x, cmd.width, cmd.blend_mode);
                }
                Cmd::BlendAlphaFill(cmd) => {
                    let alphas = alpha_buffers[cmd.thread_idx as usize];
                    fine.blend_alpha_fill(
                        row_y,
                        cmd.x,
                        cmd.width,
                        cmd.blend_mode,
                        &alphas[cmd.alpha_idx as usize..],
                    );
                }
            }
        }

        let pack_start = row_start.min(width);
        let pack_end = row_end.min(width);
        if pack_start < pack_end {
            fine.pack(
                row_idx,
                row_height,
                pack_start,
                pack_end - pack_start,
                buffer,
            );
        }
    }
}

/// A trait for objects that can render pixel data into buffers.
///
/// Painters abstract over different content sources (gradients, images, etc.) and can
/// generate pixel data in either u8 or f32 format. Implementations should provide at least
/// one of these methods; the other can delegate through conversion.
///
/// Note: Some painters may only efficiently support one numeric type. The implementation
/// may convert between types as needed.
pub trait Painter {
    /// Paint pixel data into a u8 buffer (values in 0-255 range).
    fn paint_u8(&mut self, buf: &mut [u8]);

    /// Paint pixel data into an f32 buffer (values in 0.0-1.0 range).
    fn paint_f32(&mut self, buf: &mut [f32]);
}

/// Extension trait for creating position vectors for gradient and image sampling.
///
/// This trait provides a method to generate SIMD vectors of positions that advance
/// correctly across a tile. It's used by painters to compute per-pixel coordinates
/// for sampling operations.
pub trait PosExt<S: Simd> {
    /// Create a position vector that advances appropriately across a tile.
    ///
    /// Given a starting position and per-pixel advances in x and y directions,
    /// generates a SIMD vector with the correct position for each element.
    fn splat_pos(simd: S, pos: f32, x_advance: f32, y_advance: f32) -> Self;
}

impl<S: Simd> PosExt<S> for f32x4<S> {
    #[inline(always)]
    fn splat_pos(simd: S, pos: f32, _: f32, y_advance: f32) -> Self {
        let columns: [f32; Tile::HEIGHT as usize] = [0.0, 1.0, 2.0, 3.0];
        let column_mask: Self = columns.simd_into(simd);

        column_mask.mul_add(Self::splat(simd, y_advance), Self::splat(simd, pos))
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

/// Intermediate shader result with color channels stored separately for efficient processing.
///
/// This structure holds 8 pixels worth of data in planar format (separate R, G, B, A vectors).
/// The planar layout is more efficient for certain SIMD operations before final interleaving.
pub(crate) struct ShaderResultF32<S: Simd> {
    /// Red channel values for 8 pixels.
    pub(crate) r: f32x8<S>,
    /// Green channel values for 8 pixels.
    pub(crate) g: f32x8<S>,
    /// Blue channel values for 8 pixels.
    pub(crate) b: f32x8<S>,
    /// Alpha channel values for 8 pixels.
    pub(crate) a: f32x8<S>,
}

impl<S: Simd> ShaderResultF32<S> {
    /// Convert from planar format to interleaved RGBA format.
    ///
    /// Returns two f32x16 vectors containing 8 pixels (4 RGBA components each)
    /// with channels interleaved in the standard RGBA order.
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
    /// Implements the `Painter` trait for an iterator that produces f32x16 SIMD vectors.
    ///
    /// This macro generates both `paint_u8` and `paint_f32` methods, converting between
    /// formats as needed. Used for painters that work natively with high-precision f32 data.
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
                            converted.store_slice(chunk);
                        }
                    })
                }

                fn paint_f32(&mut self, buf: &mut [f32]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            next.store_slice(chunk);
                        }
                    })
                }
            }
        };
    }

    /// Implements the `Painter` trait for an iterator that produces u8x16 SIMD vectors.
    ///
    /// This macro generates both `paint_u8` and `paint_f32` methods, converting between
    /// formats as needed. Used for painters that work natively with low-precision u8 data.
    macro_rules! u8x16_painter {
        ($($type_path:tt)+) => {
            impl<S: Simd> crate::fine::Painter for $($type_path)+ {
                fn paint_u8(&mut self, buf: &mut [u8]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            next.store_slice(chunk);
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
                            converted.store_slice(chunk);
                        }
                    })
                }
            }
        };
    }

    pub(crate) use f32x16_painter;
    pub(crate) use u8x16_painter;
}
