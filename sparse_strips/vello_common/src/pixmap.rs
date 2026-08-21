// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "png")]
use std::io::{BufRead, Seek};

use crate::fearless_simd::{Level, Simd, SimdBase, SimdInt, SimdMask, dispatch, mask8x16};
use crate::peniko::{ImageAlphaType, color::PremulRgba8};
use crate::util::Div255Ext;

#[cfg(feature = "png")]
extern crate std;

/// A pixmap of premultiplied RGBA8 values backed by [`u8`][core::u8].
#[derive(Debug, Clone)]
pub struct Pixmap {
    /// Width of the pixmap in pixels.  
    width: u16,
    /// Height of the pixmap in pixels.
    height: u16,
    /// Buffer of the pixmap in RGBA8 format.
    buf: Vec<PremulRgba8>,
    /// Whether the pixmap may have non-opaque pixels.
    ///
    /// Note: This may become stale if pixels are modified via [`data_mut()`](Self::data_mut),
    /// [`data_as_u8_slice_mut()`](Self::data_as_u8_slice_mut), or [`set_pixel()`](Self::set_pixel).
    may_have_transparency: bool,
}

/// A mutable view into premultiplied RGBA8 pixmap data.
#[derive(Debug)]
pub struct PixmapMut<'a> {
    /// Width of the pixmap in pixels.
    width: u16,
    /// Height of the pixmap in pixels.
    height: u16,
    /// Buffer of the pixmap in RGBA8 format.
    buf: &'a mut [u8],
}

impl<'a> PixmapMut<'a> {
    /// Create a new mutable pixmap view.
    ///
    /// Returns `None` if `buf` is not exactly `width * height * 4` bytes long.
    pub fn new(width: u16, height: u16, buf: &'a mut [u8]) -> Option<Self> {
        if buf.len() == usize::from(width) * usize::from(height) * 4 {
            Some(Self { width, height, buf })
        } else {
            None
        }
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Returns a mutable reference to the underlying data as premultiplied RGBA8 bytes.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.buf
    }
}

impl<'a> From<&'a mut Pixmap> for PixmapMut<'a> {
    fn from(pixmap: &'a mut Pixmap) -> Self {
        pixmap.as_mut()
    }
}

impl Pixmap {
    /// Create a new pixmap with the given width and height in pixels.
    ///
    /// All pixels are initialized to transparent black.
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![PremulRgba8::from_u32(0); width as usize * height as usize];
        Self {
            width,
            height,
            buf,
            may_have_transparency: true,
        }
    }

    /// Create a new pixmap from the given buffer of bytes, representing pixel data.
    ///
    /// # Panics
    ///
    /// - Panics if `data` is not exactly `width * height * 4` bytes long.
    /// - Panics if the capacity of the vector is not a multiple of 4.
    pub fn from_parts(
        mut data: Vec<u8>,
        width: u16,
        height: u16,
        pixel_metadata: PixelMetadata,
    ) -> Self {
        let may_have_transparency = if pixel_metadata.may_have_transparency
            && pixel_metadata.alpha_type == ImageAlphaType::Alpha
        {
            // If there might be transparency and the data is not premultiplied yet, we need to
            // iterate over all pixels anyway. Rechecking the alpha values only adds little
            // overhead (around 5-10% from my benchmarks), and lets us downgrade a conservative
            // transparency hint to fully opaque.
            premultiply_rgba8(&mut data)
        } else {
            // If the data is already premultiplied, we want to avoid reloading all pixels from
            // memory just to _maybe_ downgrade the transparency hint, so we avoid doing that
            // and always return the hint directly.
            pixel_metadata.may_have_transparency
        };

        let data: Vec<PremulRgba8> = bytemuck::try_cast_vec(data)
            .map_err(|(error, _data)| error)
            .expect("The capacity of the vector needs to be divisible by 4.");
        assert_eq!(
            data.len(),
            usize::from(width) * usize::from(height),
            "Expected `data` to have length of exactly `width * height`"
        );

        Self {
            width,
            height,
            buf: data,
            may_have_transparency,
        }
    }

    /// Resizes the pixmap container to the given width and height; this does not resize the
    /// contained image.
    ///
    /// If the pixmap buffer has to grow to fit the new size, those pixels are set to transparent
    /// black. If the pixmap buffer is larger than required, the buffer is truncated and its
    /// reserved capacity is unchanged.
    pub fn resize(&mut self, width: u16, height: u16) {
        let new_len = usize::from(width) * usize::from(height);
        // If we're growing, new pixels are transparent black
        if new_len > self.buf.len() {
            self.may_have_transparency = true;
        }
        self.width = width;
        self.height = height;
        self.buf.resize(new_len, PremulRgba8::from_u32(0));
    }

    /// Shrink the capacity of the pixmap buffer to fit the pixmap's current size.
    pub fn shrink_to_fit(&mut self) {
        self.buf.shrink_to_fit();
    }

    /// The reserved capacity (in pixels) of this pixmap.
    ///
    /// When calling [`Pixmap::resize`] with a `width * height` smaller than this value, the pixmap
    /// does not need to reallocate.
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Returns whether the pixmap may have non-opaque pixels.
    ///
    /// This value is computed at construction time. It may become stale if pixels are
    /// modified directly via [`data_mut()`](Self::data_mut),
    /// [`data_as_u8_slice_mut()`](Self::data_as_u8_slice_mut), or [`set_pixel()`](Self::set_pixel).
    ///
    /// Use [`set_may_have_transparency()`](Self::set_may_have_transparency) to manually update the flag,
    /// or [`recompute_may_have_transparency()`](Self::recompute_may_have_transparency) to recalculate it
    /// by scanning all pixels.
    pub fn may_have_transparency(&self) -> bool {
        self.may_have_transparency
    }

    /// Manually set the `may_have_transparency` flag.
    ///
    /// Use this after modifying pixels via [`data_mut()`](Self::data_mut) or
    /// [`set_pixel()`](Self::set_pixel) when you know whether the image has
    /// non-opaque pixels.
    pub fn set_may_have_transparency(&mut self, may_have_transparency: bool) {
        self.may_have_transparency = may_have_transparency;
    }

    /// Recalculate `may_have_transparency` by scanning all pixels.
    ///
    /// Use this after modifying pixels via [`data_mut()`](Self::data_mut) or
    /// [`set_pixel()`](Self::set_pixel) when you need accurate opacity information.
    pub fn recompute_may_have_transparency(&mut self) {
        self.may_have_transparency = self.buf.iter().any(|pixel| pixel.a != 255);
    }

    /// Apply an alpha value to the whole pixmap.
    pub fn multiply_alpha(&mut self, alpha: u8) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cannot overflow in this case"
        )]
        let multiply = |component| ((u16::from(alpha) * u16::from(component)) / 255) as u8;

        for pixel in self.data_mut() {
            *pixel = PremulRgba8 {
                r: multiply(pixel.r),
                g: multiply(pixel.g),
                b: multiply(pixel.b),
                a: multiply(pixel.a),
            };
        }

        // If we applied a non-opaque alpha, the image now has transparency
        if alpha != 255 {
            self.may_have_transparency = true;
        }
    }

    /// Create a pixmap from a PNG file.
    #[cfg(feature = "png")]
    pub fn from_png(data: impl BufRead + Seek) -> Result<Self, png::DecodingError> {
        let mut decoder = png::Decoder::new(data);
        decoder.set_transformations(
            png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
        );

        let mut reader = decoder.read_info()?;
        let mut pixmap = {
            let info = reader.info();
            let width: u16 = info
                .width
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?;
            let height: u16 = info
                .height
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?;
            Self::new(width, height)
        };

        // Note `reader.info()` returns the pre-transformation color type output, whereas
        // `reader.output_color_type()` takes the transformation into account.
        let (color_type, bit_depth) = reader.output_color_type();
        debug_assert_eq!(
            bit_depth,
            png::BitDepth::Eight,
            "normalize_to_color8 means the bit depth is always 8."
        );

        match color_type {
            png::ColorType::Rgb | png::ColorType::Grayscale => {
                unreachable!("We set a transformation to always convert to alpha")
            }
            png::ColorType::Indexed => {
                unreachable!("Transformation should have expanded indexed images")
            }
            png::ColorType::Rgba => {
                debug_assert_eq!(
                    Some(pixmap.data_as_u8_slice().len()),
                    reader.output_buffer_size(),
                    "The pixmap buffer should have the same number of bytes as the image."
                );
                reader.next_frame(pixmap.data_as_u8_slice_mut())?;
            }
            png::ColorType::GrayscaleAlpha => {
                debug_assert_eq!(
                    Some(pixmap.data().len() * 2),
                    reader.output_buffer_size(),
                    "The pixmap buffer should have twice the number of bytes of the grayscale image."
                );
                let mut grayscale_data = vec![0; reader.output_buffer_size().unwrap_or_default()];
                reader.next_frame(&mut grayscale_data)?;

                for (grayscale_pixel, pixmap_pixel) in
                    grayscale_data.chunks_exact(2).zip(pixmap.data_mut())
                {
                    let [gray, alpha] = grayscale_pixel.try_into().unwrap();
                    *pixmap_pixel = PremulRgba8 {
                        r: gray,
                        g: gray,
                        b: gray,
                        a: alpha,
                    };
                }
            }
        };

        pixmap.may_have_transparency = premultiply_rgba8(pixmap.data_as_u8_slice_mut());

        Ok(pixmap)
    }

    /// Return the current content of the pixmap as a PNG.
    #[cfg(feature = "png")]
    pub fn into_png(self) -> Result<Vec<u8>, png::EncodingError> {
        let mut data = Vec::new();
        let mut encoder = png::Encoder::new(&mut data, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.take(ImageAlphaType::Alpha))?;
        writer.finish().map(|_| data)
    }

    /// Returns a reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order.
    pub fn data(&self) -> &[PremulRgba8] {
        &self.buf
    }

    // TODO: Now that we have `as_mut`, maybe we don't need the
    // mutable methods. If we add a `PixmapRef` we can also remove the
    // non-mutable ones.

    /// Returns a mutable reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order.
    pub fn data_mut(&mut self) -> &mut [PremulRgba8] {
        &mut self.buf
    }

    /// Returns a reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn data_as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.buf)
    }

    /// Returns a mutable reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn data_as_u8_slice_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.buf)
    }

    /// Return a mutable view into this pixmap's pixel data.
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            width: self.width,
            height: self.height,
            buf: bytemuck::cast_slice_mut(&mut self.buf),
        }
    }

    /// Sample a pixel from the pixmap.
    ///
    /// The pixel data is [premultiplied RGBA8][PremulRgba8].
    #[inline(always)]
    pub fn sample(&self, x: u16, y: u16) -> PremulRgba8 {
        let idx = self.width as usize * y as usize + x as usize;
        self.buf[idx]
    }

    /// Sample a pixel from a custom-calculated index. This index should be calculated assuming that
    /// the data is stored in row-major order.
    #[inline(always)]
    pub fn sample_idx(&self, idx: u32) -> PremulRgba8 {
        self.buf[idx as usize]
    }

    /// Set a pixel in the pixmap at the given coordinates.
    ///
    /// The pixel data should be [premultiplied RGBA8][PremulRgba8]. The coordinate system has
    /// its origin at the top-left corner, with `x` increasing to the right and `y` increasing
    /// downward.
    #[inline(always)]
    pub fn set_pixel(&mut self, x: u16, y: u16, pixel: PremulRgba8) {
        let idx = self.width as usize * y as usize + x as usize;
        self.buf[idx] = pixel;
    }

    /// Consume the pixmap, returning its RGBA8 bytes with the requested alpha representation.
    ///
    /// The pixels are in row-major order. Note that it's always cheapest to call this method
    /// with [`ImageAlphaType::AlphaPremultiplied`] since this is the internal representation
    /// of the pixmap.
    pub fn take(self, alpha_type: ImageAlphaType) -> Vec<u8> {
        let mut data = bytemuck::cast_vec(self.buf);
        if alpha_type == ImageAlphaType::Alpha {
            for pixel in data.chunks_exact_mut(4) {
                let alpha = pixel[3];
                if alpha != 0 {
                    let scale = 255.0 / f32::from(alpha);
                    for component in &mut pixel[..3] {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "deliberate quantization"
                        )]
                        let unpremultiplied = (f32::from(*component) * scale + 0.5) as u8;
                        *component = unpremultiplied;
                    }
                }
            }
        }
        data
    }
}

/// Metadata about the pixels of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelMetadata {
    /// Whether the pixels may be non-opaque.
    ///
    /// If unsure, always set this to `true`. Setting this to `false` is a strong guarantee that
    /// every pixel in the image **is guaranteed** to be opaque. Setting this to `false` mistakenly
    /// can lead to wrong rendering.
    pub may_have_transparency: bool,
    /// How the alpha channel is represented.
    pub alpha_type: ImageAlphaType,
}

impl PixelMetadata {
    /// Create a new pixel metadata description.
    pub const fn new(alpha_type: ImageAlphaType, may_have_transparency: bool) -> Self {
        Self {
            may_have_transparency,
            alpha_type,
        }
    }
}

impl Default for PixelMetadata {
    fn default() -> Self {
        Self::new(ImageAlphaType::AlphaPremultiplied, true)
    }
}

/// Premultiplies each RGBA8 pixel in `data`.
///
/// Returns `true` if at least one pixel is not fully opaque.
fn premultiply_rgba8(data: &mut [u8]) -> bool {
    // Unfortunately we need to construct a custom level here and cannot use the one
    // from the Vello CPU / Vello Hybrid context. This does mean we are not testing
    // all possible combinations in CI, but the used intrinsics are very simple and
    // also used in other parts of the pipeline, so risk is very low.
    let level = Level::try_detect().unwrap_or(Level::baseline());

    dispatch!(level, simd => premultiply_rgba8_impl(simd, data))
}

#[inline(always)]
fn premultiply_rgba8_impl<S: Simd>(simd: S, data: &mut [u8]) -> bool {
    let (body, tail) = data.as_chunks_mut::<64>();
    let mut transparency = mask8x16::splat(simd, 0);

    for chunk in body {
        let rgba = simd.load_interleaved_128_u8x64(chunk);
        let (rg, ba) = simd.split_u8x64(rgba);
        let (r, g) = simd.split_u8x32(rg);
        let (b, a) = simd.split_u8x32(ba);

        transparency |= !a.simd_eq(255);
        let premultiply = {
            #[inline(always)]
            |component| {
                let product = simd.widen_u8x16(component) * simd.widen_u8x16(a);
                simd.narrow_u16x16(product.div_255())
            }
        };
        let premultiplied = simd.combine_u8x32(
            simd.combine_u8x16(premultiply(r), premultiply(g)),
            simd.combine_u8x16(premultiply(b), a),
        );
        simd.store_interleaved_128_u8x64(premultiplied, chunk);
    }

    let mut may_have_transparency = transparency.any_true();
    for pixel in tail.chunks_exact_mut(4) {
        let alpha = u16::from(pixel[3]);
        may_have_transparency |= alpha != 255;
        let premultiply = |component| ((u16::from(component) * alpha + 255) >> 8) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }

    may_have_transparency
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{PixelMetadata, Pixmap};
    use crate::peniko::ImageAlphaType;

    #[test]
    fn straight_alpha_is_premultiplied_in_body_and_tail() {
        let pixmap = Pixmap::from_parts(
            vec![
                // SIMD body
                200, 100, 50, 128, 128, 64, 32, 128, 255, 128, 64, 64, 255, 100, 1, 0, 64, 32, 16,
                192, 10, 20, 30, 255, 240, 120, 60, 128, 80, 40, 20, 64, 100, 50, 25, 128, 32, 16,
                8, 192, 200, 150, 100, 64, 3, 2, 1, 128, 254, 253, 252, 128, 1, 2, 3, 64, 127, 63,
                31, 192, 9, 8, 7, 255, // Scalar tail
                80, 40, 20, 64,
            ],
            17,
            1,
            PixelMetadata::new(ImageAlphaType::Alpha, true),
        );

        assert!(pixmap.may_have_transparency());
        assert_eq!(
            pixmap.data_as_u8_slice(),
            [
                // SIMD body
                100, 50, 25, 128, 64, 32, 16, 128, 64, 32, 16, 64, 0, 0, 0, 0, 48, 24, 12, 192, 10,
                20, 30, 255, 120, 60, 30, 128, 20, 10, 5, 64, 50, 25, 13, 128, 24, 12, 6, 192, 50,
                38, 25, 64, 2, 1, 1, 128, 127, 127, 126, 128, 1, 1, 1, 64, 96, 48, 24, 192, 9, 8,
                7, 255, // Scalar tail
                20, 10, 5, 64,
            ]
        );
    }

    #[test]
    fn straight_alpha_is_premultiplied_with_only_tail() {
        let pixmap = Pixmap::from_parts(
            vec![200, 100, 50, 128, 9, 8, 7, 255],
            2,
            1,
            PixelMetadata::new(ImageAlphaType::Alpha, true),
        );

        assert!(pixmap.may_have_transparency());
        assert_eq!(pixmap.data_as_u8_slice(), [100, 50, 25, 128, 9, 8, 7, 255]);
    }

    #[test]
    fn straight_opaque_alpha_clears_transparency_hint_in_body_and_tail() {
        let data = vec![
            // SIMD body
            200, 100, 50, 255, 1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255, 13, 14,
            15, 255, 16, 17, 18, 255, 19, 20, 21, 255, 22, 23, 24, 255, 25, 26, 27, 255, 28, 29,
            30, 255, 31, 32, 33, 255, 34, 35, 36, 255, 37, 38, 39, 255, 40, 41, 42, 255, 43, 44,
            45, 255, // Scalar tail
            80, 40, 20, 255,
        ];
        let pixmap = Pixmap::from_parts(
            data.clone(),
            17,
            1,
            PixelMetadata::new(ImageAlphaType::Alpha, true),
        );

        assert!(!pixmap.may_have_transparency());
        assert_eq!(pixmap.data_as_u8_slice(), data);
    }

    #[test]
    fn straight_opaque_alpha_clears_transparency_hint_with_only_tail() {
        let data = vec![1, 2, 3, 255];
        let pixmap = Pixmap::from_parts(
            data.clone(),
            1,
            1,
            PixelMetadata::new(ImageAlphaType::Alpha, true),
        );

        assert!(!pixmap.may_have_transparency());
        assert_eq!(pixmap.data_as_u8_slice(), data);
    }

    #[test]
    fn take_returns_requested_alpha_type_as_bytes() {
        let data = vec![100, 50, 25, 128, 9, 8, 7, 255, 1, 2, 3, 0];
        let pixmap = Pixmap::from_parts(
            data.clone(),
            3,
            1,
            PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, true),
        );

        assert_eq!(
            pixmap.clone().take(ImageAlphaType::AlphaPremultiplied),
            data
        );
        assert_eq!(
            pixmap.take(ImageAlphaType::Alpha),
            [199, 100, 50, 128, 9, 8, 7, 255, 1, 2, 3, 0]
        );
    }
}
