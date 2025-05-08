// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

use alloc::vec;
use alloc::vec::Vec;

use crate::peniko::color::PremulRgba8;

#[cfg(feature = "png")]
extern crate std;

/// A pixmap of premultiplied RGBA8 values backed by [`u8`][core::u8].
#[derive(Debug, Clone)]
pub struct Pixmap {
    /// Width of the pixmap in pixels.  
    width: u16,
    /// Height of the pixmap in pixels.
    height: u16,
    /// Buffer of the pixmap in RGBA format.
    buf: Vec<u8>,
}

impl Pixmap {
    /// Create a new pixmap with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![0; width as usize * height as usize * 4];
        Self { width, height, buf }
    }

    /// Create a new pixmap with the given underlying data interpreted as premultiplied RGBA8.
    ///
    /// The `data` vector must be of length `width * height * 4` exactly.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    ///
    /// # Panics
    ///
    /// Panics if the `data` vector is not of length `width * height * 4`.
    pub fn from_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            usize::from(width) * usize::from(height) * 4,
            "Expected `data` to have length of exactly `width * height * 4`"
        );
        Self {
            width,
            height,
            buf: data,
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

    /// Apply an alpha value to the whole pixmap.
    pub fn multiply_alpha(&mut self, alpha: u8) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cannot overflow in this case"
        )]
        for comp in self.data_mut() {
            *comp = ((alpha as u16 * *comp as u16) / 255) as u8;
        }
    }

    /// Create a pixmap from a PNG file.
    #[cfg(feature = "png")]
    pub fn from_png(data: impl std::io::Read) -> Result<Self, png::DecodingError> {
        let mut decoder = png::Decoder::new(data);
        decoder.set_transformations(
            png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
        );

        let mut reader = decoder.read_info()?;
        let mut img_data = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut img_data)?;
        debug_assert_eq!(
            info.bit_depth,
            png::BitDepth::Eight,
            "normalize_to_color8 means the bit depth is always 8."
        );

        let decoded_data = match info.color_type {
            png::ColorType::Rgb | png::ColorType::Grayscale => {
                unreachable!("We set a transformation to always convert to alpha")
            }
            png::ColorType::Indexed => {
                unreachable!("Transformation should have expanded indexed images")
            }
            png::ColorType::Rgba => img_data,
            png::ColorType::GrayscaleAlpha => {
                let mut rgba_data = Vec::with_capacity(img_data.len() * 2);
                for slice in img_data.chunks(2) {
                    let gray = slice[0];
                    let alpha = slice[1];
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(alpha);
                }

                rgba_data
            }
        };

        let premultiplied = decoded_data
            .chunks_exact(4)
            .flat_map(|d| {
                let alpha = d[3] as u16;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "Overflow should be impossible."
                )]
                let premultiply = |e: u8| ((e as u16 * alpha) / 255) as u8;

                if alpha == 0 {
                    [0, 0, 0, 0]
                } else {
                    [
                        premultiply(d[0]),
                        premultiply(d[1]),
                        premultiply(d[2]),
                        d[3],
                    ]
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            width: info
                .width
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?,
            height: info
                .height
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?,
            buf: premultiplied,
        })
    }

    /// Returns a reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn data(&self) -> &[u8] {
        &self.buf
    }

    /// Returns a mutable reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Sample a pixel from the pixmap.
    ///
    /// The pixel data is [premultiplied RGBA8][PremulRgba8].
    #[inline(always)]
    pub fn sample(&self, x: u16, y: u16) -> PremulRgba8 {
        let idx = 4 * (self.width as usize * y as usize + x as usize);

        // The conversion here is a no-op, as `PremulRgba8`'s fields are in the same memory order
        // as the pixel data.
        PremulRgba8::from_u8_array(self.buf[idx..][..4].try_into().unwrap())
    }

    /// Consume the pixmap, returning the data as the underlying [`Vec`] of premultiplied RGBA8
    /// bytes.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn take(self) -> Vec<u8> {
        self.buf
    }

    /// Consume the pixmap, returning the data as (unpremultiplied) RGBA8 bytes.
    ///
    /// Not fast, but useful for saving to PNG etc.
    ///
    /// The pixels are in row-major order. Each pixel consists of four bytes in the order
    /// `[r, g, b, a]`.
    pub fn take_unpremultiplied(mut self) -> Vec<u8> {
        for rgba in self.buf.chunks_exact_mut(4) {
            let alpha = 255.0 / rgba[3] as f32;

            #[expect(clippy::cast_possible_truncation, reason = "deliberate quantization")]
            if rgba[3] != 0 {
                rgba[0] = (rgba[0] as f32 * alpha + 0.5) as u8;
                rgba[1] = (rgba[1] as f32 * alpha + 0.5) as u8;
                rgba[2] = (rgba[2] as f32 * alpha + 0.5) as u8;
            }
        }

        self.buf
    }
}
