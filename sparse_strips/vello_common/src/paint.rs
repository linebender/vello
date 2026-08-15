// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use crate::pixmap::Pixmap;
use alloc::sync::Arc;
pub use peniko::Color;
use peniko::{
    Gradient,
    color::{AlphaColor, PremulRgba8, Srgb},
};

/// A paint that needs to be resolved via its index.
// In the future, we might add additional flags, that's why we have
// this thin wrapper around u32, so we can change the underlying
// representation without breaking the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPaint(u32);

impl IndexedPaint {
    /// Create a new indexed paint from an index.
    pub fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("exceeded the maximum number of paints"))
    }

    /// Return the index of the paint.
    pub fn index(&self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

/// A paint used internally by a rendering frontend to store how a draw should be painted.
/// There are only two types of paint:
///
/// 1) Simple solid colors, which are stored in premultiplied representation so that
///    the renderer doesn't have to recompute it.
/// 2) Indexed paints, which can represent any arbitrary, more complex paint that is
///    determined by the frontend. The intended way of using this is to store a vector
///    of paints and store its index inside `IndexedPaint`.
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    /// A premultiplied RGBA8 color.
    Solid(PremulColor),
    /// A paint that needs to be resolved via an index.
    Indexed(IndexedPaint),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Self::Solid(PremulColor::from_alpha_color(value))
    }
}

/// Opaque image handle
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ImageId(u32);

impl ImageId {
    // TODO: make this private in future
    /// Create a new image id from a u32.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the image id as a u32.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Bitmap source used by `Image`.
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// Pixmap pixels travel with the scene packet.
    Pixmap(Arc<Pixmap>),
    /// Pixmap pixels were registered earlier; this is just a handle.
    OpaqueId {
        /// The image handle.
        id: ImageId,
        /// Whether the image may contain non-opaque pixels.
        may_have_transparency: bool,
    },
}

impl ImageSource {
    /// Create an [`ImageSource`] from a pre-registered image handle.
    ///
    /// Conservatively assumes the image may have non-opaque pixels.
    /// Use [`Self::opaque_id_with_transparency_hint`] when you know the image is fully opaque.
    pub fn opaque_id(id: ImageId) -> Self {
        Self::OpaqueId {
            id,
            may_have_transparency: true,
        }
    }

    /// Create an [`ImageSource`] from a pre-registered image handle,
    /// with an explicit hint about whether the image may have non-opaque pixels.
    pub fn opaque_id_with_transparency_hint(id: ImageId, may_have_transparency: bool) -> Self {
        Self::OpaqueId {
            id,
            may_have_transparency,
        }
    }

    /// Returns whether this image source may contain non-opaque pixels.
    pub fn may_have_transparency(&self) -> bool {
        match self {
            Self::Pixmap(p) => p.may_have_transparency(),
            Self::OpaqueId {
                may_have_transparency,
                ..
            } => *may_have_transparency,
        }
    }

    /// Convert a [`peniko::ImageData`] to an [`ImageSource`].
    ///
    /// This is a somewhat lossy conversion, as the image data data is transformed to
    /// [premultiplied RGBA8](`PremulRgba8`).
    ///
    /// # Panics
    ///
    /// This panics if `image` has a `width` or `height` greater than `u16::MAX`.
    pub fn from_peniko_image_data(image: &peniko::ImageData) -> Self {
        // TODO: how do we deal with `peniko::ImageFormat` growing? See also
        // <https://github.com/linebender/vello/pull/996#discussion_r2080510863>.
        let do_alpha_multiply = image.alpha_type != peniko::ImageAlphaType::AlphaPremultiplied;

        assert!(
            image.width <= u16::MAX as u32 && image.height <= u16::MAX as u32,
            "The image is too big. Its width and height can be no larger than {} pixels.",
            u16::MAX,
        );
        let width = image.width.try_into().unwrap();
        let height = image.height.try_into().unwrap();

        // TODO: SIMD
        let mut may_have_transparency = false;
        #[expect(clippy::cast_possible_truncation, reason = "This cannot overflow.")]
        let pixels = image
            .data
            .data()
            .chunks_exact(4)
            .map(|pixel| {
                let rgba: [u8; 4] = match image.format {
                    peniko::ImageFormat::Rgba8 => pixel.try_into().unwrap(),
                    peniko::ImageFormat::Bgra8 => [pixel[2], pixel[1], pixel[0], pixel[3]],
                    format => unimplemented!("Unsupported image format: {format:?}"),
                };
                may_have_transparency |= rgba[3] != 255;
                let alpha = u16::from(rgba[3]);
                let multiply = |component| ((alpha * u16::from(component)) / 255) as u8;
                if do_alpha_multiply {
                    PremulRgba8 {
                        r: multiply(rgba[0]),
                        g: multiply(rgba[1]),
                        b: multiply(rgba[2]),
                        a: rgba[3],
                    }
                } else {
                    PremulRgba8 {
                        r: rgba[0],
                        g: rgba[1],
                        b: rgba[2],
                        a: rgba[3],
                    }
                }
            })
            .collect();
        let pixmap = Pixmap::from_parts_with_opacity(pixels, width, height, may_have_transparency);

        Self::Pixmap(Arc::new(pixmap))
    }
}

/// An image.
pub type Image = peniko::ImageBrush<ImageSource>;

/// Trait for resolving opaque image IDs to pixmaps at rasterization time.
///
/// This allows delaying the resolution of `ImageSource::OpaqueId` until the
/// image is actually needed during rasterization, enabling patterns like
/// dynamic sprite atlases where the image data may be updated between
/// encoding and rendering.
pub trait ImageResolver: Send + Sync {
    /// Resolve an `ImageId` to its pixmap data.
    ///
    /// This method may be called repeatedly (dozens or even hundreds of times
    /// per frame) and should therefore be very fast.
    ///
    /// Returns `None` if the image ID is not found in the registry.
    fn resolve(&self, id: ImageId) -> Option<Arc<Pixmap>>;
}

/// A no-op image resolver that always returns `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpImageResolver;

impl ImageResolver for NoOpImageResolver {
    fn resolve(&self, _id: ImageId) -> Option<Arc<Pixmap>> {
        None
    }
}

/// A premultiplied color.
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct PremulColor {
    premul_u8: PremulRgba8,
    premul_f32: peniko::color::PremulColor<Srgb>,
}

impl PremulColor {
    /// Create a new premultiplied color.
    pub fn from_alpha_color(color: AlphaColor<Srgb>) -> Self {
        Self::from_premul_color(color.premultiply())
    }

    /// Create a new premultiplied color from `peniko::PremulColor`.
    pub fn from_premul_color(color: peniko::color::PremulColor<Srgb>) -> Self {
        Self {
            premul_u8: color.to_rgba8(),
            premul_f32: color,
        }
    }

    /// Return the color as a premultiplied RGBA8 color.
    pub fn as_premul_rgba8(&self) -> PremulRgba8 {
        self.premul_u8
    }

    /// Return the color as a premultiplied RGBAF32 color.
    pub fn as_premul_f32(&self) -> peniko::color::PremulColor<Srgb> {
        self.premul_f32
    }

    /// Return whether the color is opaque (i.e. doesn't have transparency).
    pub fn is_opaque(&self) -> bool {
        self.premul_f32.components[3] == 1.0
    }
}

/// How tint color is applied to an image.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TintMode {
    /// Alpha-mask tinting: `tint_premul * source.alpha`.
    ///
    /// The source image's alpha channel is used as a coverage mask,
    /// and the result is filled with the premultiplied tint color.
    /// This is the standard approach for glyph / monochrome image tinting.
    AlphaMask = 0,
    /// Component-wise multiply: `source * tint`.
    ///
    /// Each channel of the source pixel is multiplied by the corresponding
    /// channel of the tint color. This works well for full-color images.
    Multiply = 1,
}

impl TintMode {
    /// Return the discriminant as a `u32`.
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Opt-in coverage transfer applied to an alpha-mask tint's coverage before
/// tinting, used to sharpen glyph edges.
///
/// Analytic coverage is the exact area of the pixel covered by the outline: a
/// linear ramp of slope 1.0 alpha/px across an edge, which reads slightly
/// softer than mainstream text rasterizers. Coverage `a` is remapped as
///
/// ```text
/// a' = a + c * a * (1 - a) * (2a - 1)   // steepen: lerp(identity, smoothstep)
///        + w * a * (1 - a)              // weight bias: raises the 0.5 crossing by w/4
/// ```
///
/// with strengths `c` (contrast) and `w` (weight) in `[0, 1]`, quantized to
/// 8 bits so the CPU and GPU pipelines evaluate identical parameters.
///
/// Properties:
///
/// * `(0, 0)` is a bit-exact identity, and every consumer branches on
///   [`Self::is_none`], so the disabled path executes unchanged instructions.
///   Likewise `w = 0` leaves `c`-only output bit-identical: the appended
///   weight term is exactly `+0.0`.
/// * The `c` term is symmetric about `a = 0.5`: edges steepen, stem weight is
///   preserved. The `w` term is Skia's mask-contrast form and adds weight.
/// * With `t = a - 1/2` the derivative is `1 + c/2 - 6c*t^2 - 2w*t` — concave,
///   minimized at `a = 1` where it equals `1 - c - w`. [`Self::from_bits`]
///   therefore caps `w` at `1 - c`; under that invariant the curve is
///   monotone with range `[0, 1]` and no per-pixel clamp is needed.
/// * Peak slope of the `c` term is `1 + c/2`, spanning exact-area coverage
///   (1.0 alpha/px) through a full `smoothstep` (1.5 alpha/px).
///
/// The weight term compensates polarity: blending in sRGB-encoded space makes
/// light-on-dark text read thinner than dark-on-light.
/// [`Self::resolve_for_color`] scales the stored `w` by the text color's
/// approximate relative luminance, so black text keeps the weight-free curve
/// while white text gets the full stored strength. Because the correction
/// applies to sampled coverage rather than outlines, it never enters glyph
/// atlas keys — light and dark text share one cached rasterization. This is
/// distinct from glifo's `FontEmbolden`, which offsets outlines in em space
/// and re-keys the atlas: the `w` term is a sub-pixel, device-space
/// adjustment bounded by a fraction of a pixel, free to vary per draw.
///
/// Applies only to [`TintMode::AlphaMask`]; ignored for
/// [`TintMode::Multiply`] and untinted images.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CoverageContrast {
    /// Edge-steepening strength `c`.
    contrast: u8,
    /// Weight-bias strength `w`. Invariant: `contrast + weight <= 255`,
    /// enforced by [`Self::from_bits`].
    weight: u8,
}

impl CoverageContrast {
    /// No enhancement; coverage passes through bit-exact. The default.
    pub const NONE: Self = Self {
        contrast: 0,
        weight: 0,
    };

    /// Build from an edge-steepening strength in `[0, 1]`; `0` disables and
    /// `1` is a full `smoothstep`. The weight bias is `0`; dial it in with
    /// [`Self::with_weight_strength`]. Out-of-range values clamp; `NaN` maps
    /// to [`Self::NONE`].
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to [0, 1] before scaling, and float->int casts saturate"
    )]
    pub fn from_strength(strength: f32) -> Self {
        Self {
            contrast: (strength.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            weight: 0,
        }
    }

    /// Replace the weight-bias strength, in `[0, 1]`. Clamped and quantized
    /// like [`Self::from_strength`], then capped at the contrast's headroom.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to [0, 1] before scaling, and float->int casts saturate"
    )]
    pub fn with_weight_strength(self, strength: f32) -> Self {
        Self::from_bits(
            self.contrast,
            (strength.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        )
    }

    /// Build from the raw 8-bit strengths. The weight is capped at the
    /// contrast's headroom (`weight <= 255 - contrast`), the condition under
    /// which the curve is monotone with range `[0, 1]` (see the type docs).
    pub const fn from_bits(contrast: u8, weight: u8) -> Self {
        let headroom = 255 - contrast;
        Self {
            contrast,
            weight: if weight > headroom { headroom } else { weight },
        }
    }

    /// The raw 8-bit edge-steepening strength.
    pub const fn contrast_bits(self) -> u8 {
        self.contrast
    }

    /// The raw 8-bit weight-bias strength.
    pub const fn weight_bits(self) -> u8 {
        self.weight
    }

    /// The edge-steepening strength `c` in `[0, 1]`.
    pub fn contrast_strength(self) -> f32 {
        f32::from(self.contrast) * (1.0 / 255.0)
    }

    /// The weight-bias strength `w` in `[0, 1]`.
    pub fn weight_strength(self) -> f32 {
        f32::from(self.weight) * (1.0 / 255.0)
    }

    /// Whether this is [`Self::NONE`], i.e. coverage passes through untouched.
    pub const fn is_none(self) -> bool {
        self.contrast == 0 && self.weight == 0
    }

    /// Scale the weight bias by `color`'s approximate relative luminance,
    /// turning the stored white-text strength into the effective strength for
    /// text drawn in `color`. The contrast term passes through untouched.
    ///
    /// Luminance is `0.2126*r^2 + 0.7152*g^2 + 0.0722*b^2` on channels
    /// clamped to `[0, 1]` — a gamma-2 approximation of the sRGB transfer,
    /// chosen because this runs per glyph and the exact piecewise transfer
    /// costs a `powf` per channel. It is monotone per channel and exact at
    /// the endpoints: black resolves to `w = 0` (bit-identical to the
    /// `c`-only curve), white keeps the stored strength. Alpha is ignored.
    /// `NaN` channels resolve to `w = 0`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "luminance is clamped to [0, 1] before scaling, and float->int casts saturate"
    )]
    pub fn resolve_for_color(self, color: AlphaColor<Srgb>) -> Self {
        if self.weight == 0 {
            return self;
        }
        let [r, g, b, _] = color.components;
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        let y = (0.2126 * r * r + 0.7152 * g * g + 0.0722 * b * b).clamp(0.0, 1.0);
        // Scaling only reduces the weight, preserving the headroom invariant.
        Self::from_bits(self.contrast, (f32::from(self.weight) * y + 0.5) as u8)
    }

    /// Apply the curve to a coverage value in `[0, 1]`.
    ///
    /// The guarantees assume in-range coverage; samplers that can overshoot
    /// (e.g. bicubic filtering) produce out-of-range values for which the
    /// result is finite but unspecified.
    ///
    /// This is the reference definition; `vello_cpu`'s fine stages and the
    /// image branch of `render.wesl` evaluate the same expression in the same
    /// order and must stay in sync. The `w` term is appended rather than
    /// algebraically merged so that at `w = 0` it contributes exactly `+0.0`.
    #[inline(always)]
    pub fn apply(self, alpha: f32) -> f32 {
        if self.is_none() {
            return alpha;
        }
        let c = self.contrast_strength();
        let w = self.weight_strength();
        alpha + c * alpha * (1.0 - alpha) * (2.0 * alpha - 1.0) + w * alpha * (1.0 - alpha)
    }

    /// Apply the curve to an 8-bit coverage value, rounding as the 8-bit
    /// pipeline rounds.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`apply` stays in [0, 1], and float->int casts saturate"
    )]
    #[inline(always)]
    pub fn apply_u8(self, alpha: u8) -> u8 {
        if self.is_none() {
            return alpha;
        }
        (self.apply(f32::from(alpha) * (1.0 / 255.0)) * 255.0 + 0.5) as u8
    }

    /// Apply the curve in place to a buffer of 8-bit coverage values, as
    /// produced by strip generation.
    ///
    /// Each byte is remapped through [`Self::apply_u8`], so a glyph filled
    /// directly from its outline agrees with one round-tripped through an
    /// atlas, which stores linear 8-bit coverage and applies the transfer to
    /// the sampled value at tint time.
    pub fn apply_to_coverage(self, alphas: &mut [u8]) {
        if self.is_none() {
            return;
        }
        for alpha in alphas {
            *alpha = self.apply_u8(*alpha);
        }
    }
}

/// A tint applied to image paints.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tint {
    /// The tint color.
    pub color: Color,
    /// How the tint is applied.
    pub mode: TintMode,
    /// Coverage transfer applied to the mask's coverage before tinting.
    /// Only meaningful for [`TintMode::AlphaMask`]; defaults to
    /// [`CoverageContrast::NONE`], which is bit-exact with no transfer.
    pub contrast: CoverageContrast,
}

/// A kind of paint that can be used for filling and stroking shapes.
pub type PaintType = peniko::Brush<Image, Gradient>;

#[cfg(test)]
mod tests {
    use super::ImageSource;
    use alloc::sync::Arc;

    fn image_data(pixels: &[u8], alpha_type: peniko::ImageAlphaType) -> peniko::ImageData {
        peniko::ImageData {
            data: peniko::Blob::new(Arc::new(pixels.to_vec())),
            format: peniko::ImageFormat::Rgba8,
            alpha_type,
            width: (pixels.len() / 4) as u32,
            height: 1,
        }
    }

    #[test]
    fn from_peniko_image_data_computes_transparency_hint() {
        for alpha_type in [
            peniko::ImageAlphaType::Alpha,
            peniko::ImageAlphaType::AlphaPremultiplied,
        ] {
            let opaque = image_data(&[10, 20, 30, 255, 40, 50, 60, 255], alpha_type);
            assert!(!ImageSource::from_peniko_image_data(&opaque).may_have_transparency());

            let translucent = image_data(&[10, 20, 30, 255, 40, 50, 60, 128], alpha_type);
            assert!(ImageSource::from_peniko_image_data(&translucent).may_have_transparency());
        }
    }
}
