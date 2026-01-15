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

use crate::fine::common::gradient::linear::SimdLinearKind;
use crate::fine::common::gradient::radial::SimdRadialKind;
use crate::fine::common::gradient::sweep::SimdSweepKind;
use crate::fine::common::gradient::{GradientPainter, calculate_t_vals};
use crate::fine::common::image::{FilteredImagePainter, NNImagePainter, PlainNNImagePainter};
use crate::fine::common::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::layer_manager::LayerManager;
use crate::peniko::{BlendMode, ImageQuality};
use crate::region::Region;
use crate::util::EncodedImageExt;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use vello_common::coarse::{Cmd, CommandAttrs, WideTile};
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
use vello_common::paint::{ImageSource, Paint, PremulColor};
use vello_common::pixmap::Pixmap;
use vello_common::simd::Splat4thExt;
use vello_common::tile::Tile;
use vello_common::util::f32_to_u8;

pub use highp::F32Kernel;
pub use lowp::U8Kernel;

/// Number of color components per pixel (RGBA).
pub(crate) const COLOR_COMPONENTS: usize = 4;

/// Number of color components in a single column of a tile (height * components).
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;

/// Size of the scratch buffer used for intermediate rendering operations.
/// Sized to hold a full wide tile with all color components.
pub const SCRATCH_BUF_SIZE: usize =
    WideTile::WIDTH as usize * Tile::HEIGHT as usize * COLOR_COMPONENTS;

/// Type alias for a scratch buffer that can hold a full wide tile's worth of data.
pub type ScratchBuf<F> = [F; SCRATCH_BUF_SIZE];

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
        let mulled = val.madd(v1, v2);

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

    /// Pack the blend buffer contents into the output region.
    ///
    /// Converts from the internal scratch buffer format to the output tile format,
    /// writing the results to the provided region.
    fn pack(simd: S, region: &mut Region<'_>, blend_buf: &[Self::Numeric]);

    /// Unpack the region contents back into the blend buffer.
    ///
    /// Performs the reverse of `pack`, reading pixel data from the tile region
    /// and loading it into the scratch buffer for further processing.
    fn unpack(simd: S, region: &mut Region<'_>, blend_buf: &mut [Self::Numeric]);

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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
        start_x: u16,
        start_y: u16,
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
}

/// Fine rasterizer for processing tiles at the pixel level.
///
/// This structure maintains the state and scratch buffers needed for tile-based rendering.
/// It processes rendering commands and manages a stack of blend buffers for layer composition.
#[derive(Debug)]
pub struct Fine<S: Simd, T: FineKernel<S>> {
    /// The (x, y) coordinates of the currently active wide tile being rendered.
    pub(crate) wide_coords: (u16, u16),

    /// Stack of blend buffers for managing layers and composition.
    ///
    /// Each layer pushes a new buffer onto this stack, and layers are composited
    /// by popping and blending with the buffer below.
    pub(crate) blend_buf: Vec<ScratchBuf<T::Numeric>>,

    /// Intermediate buffer used by painters to store generated pixel data before compositing.
    pub(crate) paint_buf: ScratchBuf<T::Numeric>,

    /// Buffer for storing gradient interpolation parameters (t values).
    ///
    /// Gradients pre-compute these values for efficiency before color lookup.
    pub(crate) f32_buf: Vec<f32>,

    /// The SIMD context used for vectorized operations.
    pub(crate) simd: S,
}

impl<S: Simd, T: FineKernel<S>> Fine<S, T> {
    /// Create a new fine rasterizer with the given SIMD context.
    ///
    /// Initializes all scratch buffers and sets up the initial blend buffer.
    pub fn new(simd: S) -> Self {
        Self {
            simd,
            wide_coords: (0, 0),
            blend_buf: vec![[T::Numeric::ZERO; SCRATCH_BUF_SIZE]],
            f32_buf: vec![0.0; SCRATCH_BUF_SIZE / 4],
            paint_buf: [T::Numeric::ZERO; SCRATCH_BUF_SIZE],
        }
    }

    /// Set the coordinates of the wide tile currently being rendered.
    ///
    /// This is used by painters and other operations to compute absolute pixel positions.
    pub fn set_coords(&mut self, x: u16, y: u16) {
        self.wide_coords = (x, y);
    }

    /// Clear the current blend buffer to a solid color.
    ///
    /// This efficiently fills the entire buffer with the given premultiplied color.
    pub fn clear(&mut self, premul_color: PremulColor) {
        let converted_color = T::extract_color(premul_color);
        let blend_buf = self.blend_buf.last_mut().unwrap();

        T::copy_solid(self.simd, blend_buf, converted_color);
    }

    /// Writes the current blend buffer contents to the output region.
    ///
    /// This copies pixel data from the internal scratch buffer to the tile region,
    /// converting the layout from the internal representation to the output format.
    pub fn pack(&self, region: &mut Region<'_>) {
        let blend_buf = self.blend_buf.last().unwrap();

        T::pack(self.simd, region, blend_buf);
    }

    /// Reads the region contents back into the blend buffer.
    ///
    /// This copies pixel data from the tile region to the internal scratch buffer,
    /// performing the reverse operation of `pack`. This is typically used when a layer
    /// needs to be read back for further processing.
    pub fn unpack(&mut self, region: &mut Region<'_>) {
        let blend_buf = self.blend_buf.last_mut().unwrap();

        T::unpack(self.simd, region, blend_buf);
    }

    /// Apply a filter to a layer.
    ///
    /// This applies the filter using the kernel's implementation, mutating the layer.
    pub fn filter_layer(
        &self,
        pixmap: &mut Pixmap,
        filter: &Filter,
        layer_manager: &mut LayerManager,
        transform: Affine,
    ) {
        T::filter_layer(pixmap, filter, layer_manager, transform);
    }

    /// Execute a rendering command on the current tile.
    ///
    /// This is the main dispatch method that processes different command types including
    /// fills, clips, blends, filters, masks, and buffer operations.
    pub(crate) fn run_cmd(
        &mut self,
        cmd: &Cmd,
        alphas: &[u8],
        paints: &[EncodedPaint],
        attrs: &CommandAttrs,
    ) {
        match cmd {
            Cmd::Fill(f) => {
                let fill_attrs = &attrs.fill[f.attrs_idx as usize];
                self.fill(
                    usize::from(f.x),
                    usize::from(f.width),
                    &fill_attrs.paint,
                    fill_attrs.blend_mode,
                    paints,
                    None,
                    fill_attrs.mask.as_ref(),
                );
            }
            Cmd::AlphaFill(s) => {
                let fill_attrs = &attrs.fill[s.attrs_idx as usize];
                let alpha_idx = fill_attrs.alpha_idx(s.alpha_offset) as usize;
                self.fill(
                    usize::from(s.x),
                    usize::from(s.width),
                    &fill_attrs.paint,
                    fill_attrs.blend_mode,
                    paints,
                    Some(&alphas[alpha_idx..]),
                    fill_attrs.mask.as_ref(),
                );
            }
            Cmd::Filter(_filter, _) => {
                // TODO: Apply non-spatial filters here; spatial filters need layer-level processing
                //
                // Spatial filters (e.g., Gaussian blur) need neighboring pixels and must be
                // rendered to a pixmap for layer-level processing. Non-spatial effects (e.g.,
                // color matrix, component transfer) can be processed here directly on the
                // blend buffer per-pixel as wide commands.
            }
            Cmd::PushBuf(_layer_kind) => {
                self.blend_buf.push([T::Numeric::ZERO; SCRATCH_BUF_SIZE]);
            }
            Cmd::PopBuf => {
                self.blend_buf.pop();
            }
            Cmd::ClipFill(cf) => {
                self.clip(cf.x as usize, cf.width as usize, None);
            }
            Cmd::ClipStrip(cs) => {
                let clip_attrs = &attrs.clip[cs.attrs_idx as usize];
                let alpha_idx = clip_attrs.alpha_idx(cs.alpha_offset) as usize;
                self.clip(cs.x as usize, cs.width as usize, Some(&alphas[alpha_idx..]));
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
            Cmd::PushZeroClip(_) | Cmd::PopZeroClip => {
                // These commands are handled by the dispatcher and should not reach fine rasterization
                unreachable!();
            }
        }
    }

    /// Fill a horizontal strip within the current tile using the given paint.
    ///
    /// This is the core painting method that handles solid colors, gradients, images,
    /// and blurred rounded rectangles. It applies the paint starting at the given x
    /// coordinate with the specified width, using the provided blend mode.
    ///
    /// Note: For short strip segments, benchmarks showed that not inlining this method
    /// leads to significantly worse performance.
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
        let default_blend = blend_mode == BlendMode::default();

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
                    ($may_have_opacities:expr, $filler:expr) => {
                        if $may_have_opacities || alphas.is_some() {
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
                                    g.may_have_opacities,
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
                                    g.may_have_opacities,
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
                                        g.may_have_opacities,
                                        T::gradient_painter_with_undefined(self.simd, g, f32_buf)
                                    );
                                } else {
                                    fill_complex_paint!(
                                        g.may_have_opacities,
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
                            (false, false) => {
                                // Axis-aligned with filtering - use optimized plain painters
                                if i.sampler.quality == ImageQuality::Medium {
                                    fill_complex_paint!(
                                        i.may_have_opacities,
                                        T::plain_medium_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                } else {
                                    fill_complex_paint!(
                                        i.may_have_opacities,
                                        T::high_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                }
                            }
                            (true, false) => {
                                // Skewed with filtering - use generic filtered painters
                                if i.sampler.quality == ImageQuality::Medium {
                                    fill_complex_paint!(
                                        i.may_have_opacities,
                                        T::medium_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                } else {
                                    fill_complex_paint!(
                                        i.may_have_opacities,
                                        T::high_quality_image_painter(
                                            self.simd, i, pixmap, start_x, start_y
                                        )
                                    );
                                }
                            }
                            (false, true) => {
                                fill_complex_paint!(
                                    i.may_have_opacities,
                                    T::plain_nn_image_painter(
                                        self.simd, i, pixmap, start_x, start_y
                                    )
                                );
                            }
                            (true, true) => {
                                fill_complex_paint!(
                                    i.may_have_opacities,
                                    T::nn_image_painter(self.simd, i, pixmap, start_x, start_y)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Blend the top blend buffer into the buffer below it.
    ///
    /// This pops the top buffer from the blend stack and composites it onto the
    /// buffer below using the specified blend mode. This is the core operation for
    /// layer composition.
    pub(crate) fn blend(&mut self, blend_mode: BlendMode) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        if blend_mode == BlendMode::default() {
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

    /// Apply a clipping mask from the top buffer to the buffer below.
    ///
    /// Uses the top buffer's alpha channel as a mask, multiplying it with the buffer
    /// below. This implements clipping by masking out pixels outside the clip region.
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
                            chunk.copy_from_slice(converted.as_slice());
                        }
                    })
                }

                fn paint_f32(&mut self, buf: &mut [f32]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            chunk.copy_from_slice(next.as_slice());
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
                            chunk.copy_from_slice(next.as_slice());
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
                            chunk.copy_from_slice(converted.as_slice());
                        }
                    })
                }
            }
        };
    }

    pub(crate) use f32x16_painter;
    pub(crate) use u8x16_painter;
}
