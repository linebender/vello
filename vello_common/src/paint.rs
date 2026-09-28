// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use crate::TextureId;
use crate::geometry::RectU16;
use crate::pixmap::{PixelMetadata, Pixmap};
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
    // TODO: Explore whether we can merge opaque ID and external texture in some form?
    /// Pixmap pixels were registered earlier; this is just a handle.
    OpaqueId {
        /// The image handle.
        id: ImageId,
        /// Whether the image may contain non-opaque pixels.
        may_have_transparency: bool,
    },
    /// An externally owned texture supplied to the renderer at render time.
    ExternalTexture {
        /// Opaque external texture handle.
        id: TextureId,
        /// Source region to sample from in texel coordinates.
        source_region: RectU16,
        /// Whether the source region may contain non-opaque pixels.
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

    /// Create an image source backed by a texture supplied to the renderer at render time.
    ///
    /// # Panics
    ///
    /// Panics if `source_region` is empty.
    pub fn external_texture(
        texture_id: TextureId,
        source_region: RectU16,
        may_have_transparency: bool,
    ) -> Self {
        assert!(
            !source_region.is_empty(),
            "external texture source regions must not be empty"
        );

        Self::ExternalTexture {
            id: texture_id,
            source_region,
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
            }
            | Self::ExternalTexture {
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
        assert!(
            image.width <= u16::MAX as u32 && image.height <= u16::MAX as u32,
            "The image is too big. Its width and height can be no larger than {} pixels.",
            u16::MAX,
        );
        let width = image.width.try_into().unwrap();
        let height = image.height.try_into().unwrap();

        // Unfortunately, we have to create a new allocation, because pixmap requires
        // a real vector.
        // TODO: Figure out a better story for this.
        let mut rgba = image.data.data().to_vec();
        match image.format {
            peniko::ImageFormat::Rgba8 => {}
            peniko::ImageFormat::Bgra8 => {
                // TODO: SIMDify
                for pixel in rgba.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }
            format => unimplemented!("Unsupported image format: {format:?}"),
        }

        let pixmap = Pixmap::from_parts(
            rgba,
            width,
            height,
            PixelMetadata::new(image.alpha_type, true),
        );

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

    /// Return whether the color is fully transparent.
    pub fn is_transparent(&self) -> bool {
        self.premul_f32.components[3] == 0.0
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
    /// The alpha is first remapped through the tint's [`CoverageContrast`].
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

/// An opt-in transfer curve for glyph coverage, used to sharpen text.
///
/// Coverage `a` is remapped as
///
/// ```text
/// a' = a + c·a(1 − a)(2a − 1) + w·a(1 − a)
/// ```
///
/// The contrast `c` steepens edges symmetrically around `a = 0.5` and keeps that midpoint
/// (`c = 1` is a smoothstep). The weight `w` adds coverage, which thickens stems. The slope
/// is smallest at `a = 1`, where it is `1 − c − w`, so constructors cap `w` at `1 − c`; this
/// keeps the curve monotonic and within `[0, 1]`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CoverageContrast {
    contrast: u8,
    /// Invariant: `contrast + weight <= 255`.
    weight: u8,
}

impl CoverageContrast {
    /// No transfer.
    pub const NONE: Self = Self {
        contrast: 0,
        weight: 0,
    };

    /// Create a transfer from contrast and weight strengths in `[0, 1]`.
    ///
    /// Both are clamped to `[0, 1]` (`NaN` becomes `0`) and stored with 8-bit precision, so
    /// that all pipelines evaluate the same curve. The weight is capped at `1 − c`.
    pub fn new(contrast: f32, weight: f32) -> Self {
        // `NaN` casts to 0.
        let quantize = |strength: f32| (strength.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        Self::from_bits(quantize(contrast), quantize(weight))
    }

    /// Create a transfer from 8-bit strengths, where `255` corresponds to `1.0`.
    ///
    /// `weight` is capped at `255 − contrast`.
    pub const fn from_bits(contrast: u8, weight: u8) -> Self {
        let max_weight = 255 - contrast;
        Self {
            contrast,
            weight: if weight > max_weight {
                max_weight
            } else {
                weight
            },
        }
    }

    /// The 8-bit contrast strength.
    pub const fn contrast_bits(self) -> u8 {
        self.contrast
    }

    /// The 8-bit weight strength.
    pub const fn weight_bits(self) -> u8 {
        self.weight
    }

    /// The contrast strength `c` in `[0, 1]`.
    pub fn contrast_strength(self) -> f32 {
        f32::from(self.contrast) * (1.0 / 255.0)
    }

    /// The weight strength `w` in `[0, 1]`.
    pub fn weight_strength(self) -> f32 {
        f32::from(self.weight) * (1.0 / 255.0)
    }

    /// Whether this is [`Self::NONE`].
    pub const fn is_none(self) -> bool {
        self.contrast == 0 && self.weight == 0
    }

    /// Apply the transfer to a coverage value in `[0, 1]`.
    ///
    /// This is the reference definition; the renderers evaluate the same expression in the
    /// same order.
    #[inline(always)]
    pub fn apply(self, coverage: f32) -> f32 {
        if self.is_none() {
            return coverage;
        }
        let a = coverage;
        let c = self.contrast_strength();
        let w = self.weight_strength();
        a + c * a * (1.0 - a) * (2.0 * a - 1.0) + w * a * (1.0 - a)
    }

    /// Apply the transfer to an 8-bit coverage value, rounding to the nearest value.
    #[inline(always)]
    pub fn apply_u8(self, coverage: u8) -> u8 {
        if self.is_none() {
            return coverage;
        }
        (self.apply(f32::from(coverage) * (1.0 / 255.0)) * 255.0 + 0.5) as u8
    }
}

/// A tint applied to image paints.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tint {
    /// The tint color.
    pub color: Color,
    /// How the tint is applied.
    pub mode: TintMode,
    /// A coverage transfer applied to the image's alpha before tinting.
    ///
    /// Only used with [`TintMode::AlphaMask`].
    pub contrast: CoverageContrast,
}

/// A kind of paint that can be used for filling and stroking shapes.
pub type PaintType = peniko::Brush<Image, Gradient>;

#[cfg(test)]
mod tests {
    use super::{CoverageContrast, ImageSource};
    use alloc::sync::Arc;

    /// Weight strengths sampled when sweeping all contrast strengths.
    const WEIGHT_SAMPLES: [u8; 6] = [0, 1, 64, 128, 191, 255];

    #[test]
    fn coverage_contrast_none_is_identity() {
        let none = CoverageContrast::NONE;
        for a in 0..=u8::MAX {
            assert_eq!(none.apply_u8(a), a);
        }
        for a in [0.0_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_eq!(none.apply(a).to_bits(), a.to_bits());
        }
    }

    /// Empty and fully covered pixels are unaffected, only edges change.
    #[test]
    fn coverage_contrast_preserves_endpoints() {
        for contrast in 0..=u8::MAX {
            for weight in WEIGHT_SAMPLES {
                let transfer = CoverageContrast::from_bits(contrast, weight);
                assert_eq!(transfer.apply(0.0).to_bits(), 0.0_f32.to_bits());
                assert_eq!(transfer.apply(1.0).to_bits(), 1.0_f32.to_bits());
                assert_eq!(transfer.apply_u8(0), 0);
                assert_eq!(transfer.apply_u8(255), 255);
            }
        }
    }

    /// Without weight, the curve is symmetric around `a = 0.5` and keeps that midpoint.
    #[test]
    fn coverage_contrast_without_weight_is_symmetric() {
        for bits in 0..=u8::MAX {
            let transfer = CoverageContrast::from_bits(bits, 0);
            assert_eq!(transfer.apply(0.5).to_bits(), 0.5_f32.to_bits());
            for i in 0..=1000_u16 {
                let a = f32::from(i) / 1000.0;
                let mirrored = 1.0 - transfer.apply(1.0 - a);
                assert!(
                    (transfer.apply(a) - mirrored).abs() < 1e-6,
                    "contrast {bits} at {a}"
                );
            }
        }
    }

    /// The weight cap keeps the curve monotonic and within `[0, 1]`.
    #[test]
    fn coverage_contrast_is_monotonic_and_in_range() {
        for contrast in 0..=u8::MAX {
            for weight in WEIGHT_SAMPLES {
                let transfer = CoverageContrast::from_bits(contrast, weight);
                let mut previous = 0.0;
                for i in 0..=2000_u16 {
                    let a = f32::from(i) / 2000.0;
                    let value = transfer.apply(a);
                    assert!(
                        (previous..=1.0).contains(&value),
                        "{transfer:?} at {a}: {value}"
                    );
                    previous = value;
                }
                for a in 1..=u8::MAX {
                    assert!(
                        transfer.apply_u8(a) >= transfer.apply_u8(a - 1),
                        "{transfer:?} at {a}"
                    );
                }
            }
        }

        // At the cap, the slope at `a = 1` is zero, so the curve must not overshoot just below
        // it.
        for contrast in [0_u8, 1, 64, 128, 191, 254] {
            let transfer = CoverageContrast::from_bits(contrast, u8::MAX);
            for i in 0..=10_000_u16 {
                let a = 1.0 - f32::from(i) * 1e-6;
                assert!(transfer.apply(a) <= 1.0, "{transfer:?} at {a}");
            }
        }
    }

    /// Without contrast, the weight only adds coverage.
    #[test]
    fn coverage_contrast_weight_adds_coverage() {
        for bits in [1_u8, 64, 128, 191, 255] {
            let transfer = CoverageContrast::from_bits(0, bits);
            for i in 0..=1000_u16 {
                let a = f32::from(i) / 1000.0;
                assert!(transfer.apply(a) >= a, "weight {bits} at {a}");
            }
            let midpoint = 0.5 + transfer.weight_strength() / 4.0;
            assert!(
                (transfer.apply(0.5) - midpoint).abs() < 1e-6,
                "weight {bits}"
            );
        }
    }

    #[test]
    fn coverage_contrast_constructors_clamp_quantize_and_cap() {
        assert_eq!(CoverageContrast::new(0.0, 0.0), CoverageContrast::NONE);
        assert_eq!(
            CoverageContrast::new(0.5, 0.2),
            CoverageContrast::from_bits(128, 51)
        );
        assert_eq!(
            CoverageContrast::new(-5.0, f32::NAN),
            CoverageContrast::NONE
        );
        assert_eq!(
            CoverageContrast::new(5.0, 0.0),
            CoverageContrast::from_bits(255, 0)
        );

        // The weight is capped at `1 − c`.
        assert_eq!(CoverageContrast::new(0.5, 1.0).weight_bits(), 127);
        assert_eq!(CoverageContrast::from_bits(200, 200).weight_bits(), 55);
        assert_eq!(CoverageContrast::from_bits(255, 255).weight_bits(), 0);
        assert_eq!(CoverageContrast::from_bits(0, 255).weight_bits(), 255);
    }

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
        let opaque = image_data(
            &[10, 20, 30, 255, 40, 50, 60, 255],
            peniko::ImageAlphaType::Alpha,
        );
        assert!(!ImageSource::from_peniko_image_data(&opaque).may_have_transparency());

        let translucent = image_data(
            &[10, 20, 30, 255, 40, 50, 60, 128],
            peniko::ImageAlphaType::Alpha,
        );
        assert!(ImageSource::from_peniko_image_data(&translucent).may_have_transparency());

        let premultiplied = image_data(
            &[10, 20, 30, 255, 40, 50, 60, 255],
            peniko::ImageAlphaType::AlphaPremultiplied,
        );
        assert!(ImageSource::from_peniko_image_data(&premultiplied).may_have_transparency());
    }
}
