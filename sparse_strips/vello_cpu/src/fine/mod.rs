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

use crate::coarse::depth::DepthBuffer;
use crate::coarse::{CommandBucketer, DepthFill, PaintFill, PaintFillAttrs, RenderCmd, RowState};
use crate::filter::context::ScratchBuffer;
use crate::fine::common::gradient::GradientPainter;
pub(crate) use crate::fine::common::gradient::calculate_t_vals;
pub(crate) use crate::fine::common::gradient::linear::SimdLinearKind;
pub(crate) use crate::fine::common::gradient::radial::SimdRadialKind;
pub(crate) use crate::fine::common::gradient::sweep::SimdSweepKind;
use crate::fine::common::image::{FilteredImagePainter, NNImagePainter, PlainNNImagePainter};
use crate::fine::common::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::peniko::{BlendMode, ImageQuality};
use crate::region::Region;
use crate::util::{EncodedImageExt, VecPool};
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

#[doc(hidden)]
pub use crate::util::Span;
pub use highp::F32Kernel;
pub use lowp::U8Kernel;

/// Offset to shift from pixel corner to pixel center for sampling.
const PIXEL_CENTER_OFFSET: f64 = 0.5;

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

    /// Pack row scratch data into a row-major output buffer.
    fn pack(simd: S, scratch: &[Self::Numeric], width: usize, region: &mut Region<'_>);

    /// Unpack row-major input data into row scratch.
    fn unpack(simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [Self::Numeric]);

    /// Apply a filter to a layer.
    ///
    /// This is used for applying filters to whole layers, which is necessary for
    /// spatial filters (like blur) that need to access neighboring pixels. The filter
    /// is applied in-place to the provided pixmap.
    ///
    /// The transform parameter is used to scale filter parameters based on the current
    /// transformation matrix (e.g., zoom level), ensuring filters look consistent
    /// regardless of scale.
    #[expect(
        private_interfaces,
        reason = "`FineKernel` is public but this specific method is not needed."
    )]
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        filter_scratch: &mut ScratchBuffer,
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
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>) {
        let color = Self::extract_color(color);

        if color[3] == Self::Numeric::ONE && alphas.is_none() {
            Self::copy_solid(simd, dest, color);
        } else {
            Self::alpha_composite_solid(simd, dest, color, alphas);
        }
    }
}

/// Fine rasterizer for processing strip rows at the pixel level.
#[derive(Debug)]
#[doc(hidden)]
pub struct Fine<S: Simd, T: FineKernel<S>> {
    /// The SIMD context used for vectorized operations.
    simd: S,
    // TODO: If we make sure that strips never exceed the viewport, we can delete this.
    /// Pixel span covered by the blend buffers.
    buffer_span: Span,
    /// Stack of blend buffers for managing layers and composition.
    ///
    /// Each layer pushes a new buffer onto this stack, and layers are composited
    /// by popping and blending with the buffer below.
    blend_buffers: Vec<Vec<T::Numeric>>,
    /// Pool for reusing layer buffer allocations.
    buffer_pool: VecPool<T::Numeric>,
    /// Intermediate buffer used by painters to store generated pixel data before compositing.
    paint_buf: Vec<T::Numeric>,
    /// Buffer for storing gradient interpolation parameters (t values).
    f32_buf: Vec<f32>,
    /// The current strip row y-coordinate in scene/filter coordinates.
    row_y: u16,
    /// The origin of the current target we are rendering into.
    origin: (u16, u16),
}

impl<S: Simd, T: FineKernel<S>> Fine<S, T> {
    /// Create a new fine rasterizer with the given SIMD context.
    ///
    /// Initializes all scratch buffers and sets up the initial blend buffer.
    #[doc(hidden)]
    pub fn new(simd: S, buffer_width: u16) -> Self {
        let scratch_len = usize::from(buffer_width) * TILE_HEIGHT_COMPONENTS;
        Self {
            simd,
            buffer_span: Span::new(0, buffer_width),
            blend_buffers: vec![vec![T::Numeric::ZERO; scratch_len]],
            buffer_pool: VecPool::default(),
            paint_buf: Vec::new(),
            f32_buf: Vec::new(),
            row_y: 0,
            origin: (0, 0),
        }
    }

    fn set_row_y(&mut self, row_y: u16) {
        self.row_y = row_y;
    }

    fn set_paint_offset(&mut self, paint_offset: (u16, u16)) {
        self.origin = paint_offset;
    }

    fn scratch_range(span: Span) -> core::ops::Range<usize> {
        let start = usize::from(span.pixel_x()) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(span.pixel_width()) * TILE_HEIGHT_COMPONENTS;
        start..start + len
    }

    // The reason that we have this optimization is that, as was determined by profiling
    // just always clearing the whole fine buffer can be expensive, especially for larger
    // viewports. This is what a previous version of Vello CPU did.
    // With the current version, we can utilize the fact that we have a depth buffer where
    // a certain range of pixels is already filled with an opaque paint. Therefore, we only
    // need to clear (or unpack) the parts that are not covered by such a paint.
    /// Initialize every range in the buffer that has not been filled yet
    /// with a solid paint.
    ///
    /// In case [`ComositeMode::SrcOver`] was chosen, it will be initialized with
    /// the pixels from the user-supplied pixmap. Otherwise, the range will simply be zeroed.
    fn init_uncovered_range(
        &mut self,
        scratch_span: Span,
        region: &mut Region<'_>,
        use_src_over: bool,
        depth: &DepthBuffer,
    ) {
        depth.for_each_unset_run(scratch_span, |span| {
            let x = span.pixel_x();
            let end = span.pixel_end();

            if use_src_over {
                let mut region = region.sub_span(x, end - x);
                self.unpack(x, &mut region);
            } else {
                self.blend_buffers[0][Self::scratch_range(Span::new(x, end - x))]
                    .fill(T::Numeric::ZERO);
            }
        });
    }

    /// Writes the current buffer contents to the output row.
    #[doc(hidden)]
    pub fn pack(&self, scratch_x_start: u16, region: &mut Region<'_>) {
        let scratch_x = usize::from(scratch_x_start);
        let width = usize::from(region.width());
        let scratch = self.blend_buffers.last().unwrap();

        T::pack(
            self.simd,
            &scratch[scratch_x * TILE_HEIGHT_COMPONENTS..],
            width,
            region,
        );
    }

    /// Reads the pixels of the target back into the buffer.
    ///
    /// This does the opposite of [`Fine::pack`].
    #[doc(hidden)]
    pub fn unpack(&mut self, scratch_x_start: u16, region: &mut Region<'_>) {
        let scratch_x = usize::from(scratch_x_start);
        let width = usize::from(region.width());
        let scratch = self.blend_buffers.last_mut().unwrap();

        T::unpack(
            self.simd,
            region,
            width,
            &mut scratch[scratch_x * TILE_HEIGHT_COMPONENTS..],
        );
    }

    /// Execute a bucketed rendering command on the current strip row.
    ///
    /// This is the main dispatch method for fine rasterization. It processes paint fills,
    /// layer buffers, filter layer composites, masks, opacity, and layer blending.
    fn run_cmd(
        &mut self,
        cmd: RenderCmd,
        bucketer: &CommandBucketer,
        row: &RowState,
        row_y: u16,
        resources: FineResources<'_>,
        depth: &DepthBuffer,
    ) {
        match cmd {
            RenderCmd::PaintFill(cmd) => {
                let attrs = &bucketer.paint_fill_attrs[cmd.attrs_idx as usize];
                let alphas = resources.alpha_buffers[attrs.thread_idx as usize];

                let Some(span) = cmd.span.intersect(self.buffer_span) else {
                    return;
                };

                // Avoid using depth buffer if it's trivially skippable,
                // since it's generally cheaper to not use it at all than consult it just to be
                // returned the same span.
                if !row.can_skip_depth(cmd.span, attrs.draw_id) {
                    depth.for_each_visible_run(span, attrs.draw_id, |span| {
                        self.paint_fill(cmd, span, alphas, attrs, resources);
                    });
                } else {
                    self.paint_fill(cmd, span, alphas, attrs, resources);
                }
            }
            RenderCmd::PushBuf => {
                let mut buf = self.buffer_pool.take();
                buf.resize(self.blend_buffers[0].len(), T::Numeric::ZERO);
                self.blend_buffers.push(buf);
            }
            RenderCmd::PopBuf => {
                let popped = self.blend_buffers.pop().unwrap();
                self.buffer_pool.submit(popped);
            }
            RenderCmd::LayerFill(cmd) => {
                let attrs = &bucketer.layer_fill_attrs[cmd.attrs_idx as usize];

                if attrs.opacity != 1.0 {
                    self.opacity(cmd.span, attrs.opacity);
                }
                if let Some(mask) = attrs.mask.as_ref() {
                    self.mask(row_y, cmd.span, mask);
                }
                let alphas = cmd.alpha_idx().map(|alpha_idx| {
                    &resources.alpha_buffers[attrs.thread_idx as usize][alpha_idx as usize..]
                });

                self.blend(row_y, cmd.span, attrs.blend_mode, alphas);
            }
        }
    }

    fn opacity(&mut self, span: Span, opacity: f32) {
        let target = self.blend_buffers.last_mut().unwrap();
        let target = &mut target[Self::scratch_range(span)];

        T::apply_mask(
            self.simd,
            target,
            iter::repeat(T::NumericVec::from_f32(
                self.simd,
                f32x16::splat(self.simd, opacity),
            )),
        );
    }

    fn mask(&mut self, row_y: u16, span: Span, mask: &Mask) {
        let x = span.pixel_x();
        let width = span.pixel_width();
        let target = self.blend_buffers.last_mut().unwrap();
        let target = &mut target[Self::scratch_range(span)];
        let y = u32::from(row_y) + u32x4::from_slice(self.simd, &[0, 1, 2, 3]);
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
                self.simd,
                &[
                    s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3, s4, s4, s4, s4,
                ],
            );
            T::NumericVec::from_u8(self.simd, samples)
        });

        T::apply_mask(self.simd, target, iter);
    }

    fn blend(&mut self, row_y: u16, span: Span, blend_mode: BlendMode, alphas: Option<&[u8]>) {
        let Some(span) = span.intersect(self.buffer_span) else {
            return;
        };

        let x = span.pixel_x();
        let (source, rest) = self.blend_buffers.split_last_mut().unwrap();
        let target = rest.last_mut().unwrap();
        let range = Self::scratch_range(span);
        let source = &mut source[range.clone()];
        let target = &mut target[range];

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

    #[doc(hidden)]
    pub fn fill(
        &mut self,
        span: Span,
        paint: &Paint,
        blend_mode: BlendMode,
        resources: FineResources<'_>,
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    ) {
        match paint {
            Paint::Solid(color) => {
                self.fill_solid_with_attrs(span, self.row_y, *color, blend_mode, mask, alphas);
            }
            Paint::Indexed(index) => {
                self.fill_indexed(
                    span.pixel_x(),
                    self.row_y,
                    span.pixel_x().saturating_add(self.origin.0),
                    self.row_y.saturating_add(self.origin.1),
                    span.pixel_width(),
                    index.index(),
                    blend_mode,
                    mask,
                    resources,
                    alphas,
                );
            }
        }
    }

    fn fill_solid(&mut self, span: Span, color: PremulColor, alphas: Option<&[u8]>) {
        let simd = self.simd;
        let scratch = self.blend_buffers.last_mut().unwrap();
        T::fill_solid(simd, &mut scratch[Self::scratch_range(span)], color, alphas);
    }

    fn fill_solid_with_attrs(
        &mut self,
        span: Span,
        y: u16,
        color: PremulColor,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        alphas: Option<&[u8]>,
    ) {
        if blend_mode == BlendMode::default() && mask.is_none() {
            self.fill_solid(span, color, alphas);
            return;
        }

        if span.pixel_width() == 0 {
            return;
        }

        let x = span.pixel_x();
        let color = T::extract_color(color);
        let simd = self.simd;
        let color = T::Composite::from_color(simd, color);
        let scratch = self.blend_buffers.last_mut().unwrap();
        T::blend(
            simd,
            &mut scratch[Self::scratch_range(span)],
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
        sample_x: u16,
        sample_y: u16,
        width: u16,
        paint_index: usize,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        resources: FineResources<'_>,
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

        if width == 0 {
            return;
        }

        let simd = self.simd;
        let width = usize::from(width);
        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let dest = &mut self.blend_buffers.last_mut().unwrap()[start..start + len];
        let color_buf = &mut self.paint_buf[..len];
        let encoded_paint = encoded_paint(
            paint_index,
            resources.encoded_paints,
            resources.filter_paints,
        );

        let sampler_x = f64::from(sample_x) + PIXEL_CENTER_OFFSET;
        let sampler_y = f64::from(sample_y) + PIXEL_CENTER_OFFSET;
        let default_blend = blend_mode == BlendMode::default();

        // We need to have this as a macro because closures cannot take generic arguments, and
        // we would have to repeatedly provide all arguments if we made it a function.
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
                    // Similarly to solid colors we can just override the previous values
                    // if all colors in the gradient are fully opaque.
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
                // Note that we are calculating the t values first, store them in a separate
                // buffer and then pass that buffer to the iterator instead of calculating
                // the t values on the fly in the iterator. The latter would be faster, but
                // it would probably increase code size a lot, because the functions for
                // position calculation need to be inlined for good performance.
                let t_vals = &mut self.f32_buf[..width * Tile::HEIGHT as usize];

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
                    ImageSource::OpaqueId { id, .. } => resources
                        .image_resolver
                        .resolve(*id)
                        .unwrap_or_else(|| panic!("Image {:?} not found in registry", id)),
                };
                let tint = image.tint.as_ref();

                match (image.has_skew(), image.nearest_neighbor()) {
                    (false, false) => {
                        // Axis-aligned with filtering - use optimized plain painters
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
                        // Skewed with filtering - use generic filtered painters
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

    fn render_opaque(
        &mut self,
        cmd: DepthFill,
        attrs: &PaintFillAttrs,
        resources: FineResources<'_>,
        depth: &mut DepthBuffer,
    ) {
        self.set_paint_offset(attrs.origin);
        depth.for_each_unset_run_and_write(cmd.bucket_range(), attrs.draw_id, |bucket_range| {
            let span = bucket_range.span();
            self.fill(
                span,
                &attrs.paint,
                attrs.blend_mode,
                resources,
                None,
                attrs.mask.as_ref(),
            );
        });
    }

    fn paint_fill(
        &mut self,
        cmd: PaintFill,
        span: Span,
        alphas: &[u8],
        attrs: &PaintFillAttrs,
        resources: FineResources<'_>,
    ) {
        let x = span.pixel_x();
        self.set_paint_offset(attrs.origin);
        let alphas = cmd.alpha_idx().map(|alpha_idx| {
            let alpha_offset =
                alpha_idx as usize + usize::from(x - cmd.span.pixel_x()) * Tile::HEIGHT as usize;
            &alphas[alpha_offset..]
        });
        self.fill(
            span,
            &attrs.paint,
            attrs.blend_mode,
            resources,
            alphas,
            attrs.mask.as_ref(),
        );
    }
}

fn encoded_paint<'a>(
    index: usize,
    encoded_paints: &'a [EncodedPaint],
    filter_paints: &'a [EncodedPaint],
) -> &'a EncodedPaint {
    encoded_paints
        .get(index)
        .unwrap_or_else(|| &filter_paints[index - encoded_paints.len()])
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct FineResources<'a> {
    pub alpha_buffers: &'a [&'a [u8]],
    pub encoded_paints: &'a [EncodedPaint],
    pub filter_paints: &'a [EncodedPaint],
    pub image_resolver: &'a dyn ImageResolver,
}

impl Debug for FineResources<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FineResources")
            .field("alpha_buffers", &self.alpha_buffers.len())
            .field("encoded_paints", &self.encoded_paints.len())
            .field("filter_paints", &self.filter_paints.len())
            .finish_non_exhaustive()
    }
}

/// Placement and compositing settings for fine rasterization into a target pixmap.
#[derive(Clone, Copy)]
pub(crate) struct FineRenderParams {
    /// Scene/filter dimensions before clipping to the destination pixmap.
    pub(crate) scene_size: (u16, u16),
    /// Destination offset in the target pixmap.
    pub(crate) target_offset: (u16, u16),
}

pub(crate) fn rasterize_region<S: Simd, T: FineKernel<S>>(
    fine: &mut Fine<S, T>,
    depth: &mut DepthBuffer,
    region: &mut Region<'_>,
    bucketer: &CommandBucketer,
    resources: FineResources<'_>,
    unpack_dest: bool,
) {
    let scene_y = region.row_idx as u16 * Tile::HEIGHT;
    let row = &bucketer.rows()[region.row_idx];
    let Some(row_bounds) = row.coarse_span() else {
        return;
    };
    let mut row_start = row_bounds.pixel_x();
    let mut row_end = row_bounds.pixel_end();
    row_start = row_start.min(region.width());
    row_end = row_end.min(region.width());
    if row_start >= row_end {
        return;
    }

    fine.set_row_y(scene_y);
    let row_span = Span::new(row_start, row_end - row_start);
    depth.clear();

    for &cmd in row.depth_cmds.iter().rev() {
        let attrs = &bucketer.paint_fill_attrs[cmd.attrs_idx as usize];
        fine.render_opaque(cmd, attrs, resources, depth);
    }

    fine.init_uncovered_range(row_span, region, unpack_dest, depth);

    for cmd in &row.render_cmds {
        fine.run_cmd(*cmd, bucketer, row, scene_y, resources, depth);
    }

    let pack_start = row_start;
    let pack_end = row_end;
    if pack_start < pack_end {
        let mut region = region.sub_span(pack_start, pack_end - pack_start);
        fine.pack(pack_start, &mut region);
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
