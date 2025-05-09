// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

use alloc::vec;
use alloc::vec::Vec;
use peniko::color::Rgba8;

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
    /// Buffer of the pixmap in RGBA8 format.
    buf: Vec<PremulRgba8>,
}

impl Pixmap {
    /// Create a new pixmap with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![PremulRgba8::from_u32(0); width as usize * height as usize];
        Self { width, height, buf }
    }

    /// Create a new pixmap with the given premultiplied RGBA8 data.
    ///
    /// The `data` vector must be of length `width * height` exactly.
    ///
    /// The pixels are in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if the `data` vector is not of length `width * height`.
    pub fn from_parts(data: Vec<PremulRgba8>, width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            usize::from(width) * usize::from(height),
            "Expected `data` to have length of exactly `width * height`"
        );
        Self {
            width,
            height,
            buf: data,
        }
    }

    /// Resizes the pixmap container to the given width and height; this does not resize the
    /// contained image.
    ///
    /// If the pixmap buffer has to grow to fit the new size, those pixels are set to transparent
    /// black. If the pixmap buffer is larger than required, the buffer is truncated and its
    /// reserved capacity is unchanged.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.buf.resize(
            usize::from(width) * usize::from(height),
            PremulRgba8::from_u32(0),
        );
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
    }

    /// Create a pixmap from a PNG file.
    #[cfg(feature = "png")]
    pub fn from_png(data: impl std::io::Read) -> Result<Self, png::DecodingError> {
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
                    pixmap.data_as_u8_slice().len(),
                    reader.output_buffer_size(),
                    "The pixmap buffer should have the same number of bytes as the image."
                );
                reader.next_frame(pixmap.data_as_u8_slice_mut())?;
            }
            png::ColorType::GrayscaleAlpha => {
                debug_assert_eq!(
                    pixmap.data().len() * 2,
                    reader.output_buffer_size(),
                    "The pixmap buffer should have twice the number of bytes of the grayscale image."
                );
                let mut grayscale_data = vec![0; reader.output_buffer_size()];
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

        for pixel in pixmap.data_mut() {
            let alpha = u16::from(pixel.a);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Overflow should be impossible."
            )]
            let premultiply = |e: u8| ((u16::from(e) * alpha) / 255) as u8;
            pixel.r = premultiply(pixel.r);
            pixel.g = premultiply(pixel.g);
            pixel.b = premultiply(pixel.b);
        }

        Ok(pixmap)
    }

    /// Returns a reference to the underlying data as premultiplied RGBA8.
    ///
    /// The pixels are in row-major order.
    pub fn data(&self) -> &[PremulRgba8] {
        &self.buf
    }

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

    /// Sample a pixel from the pixmap.
    ///
    /// The pixel data is [premultiplied RGBA8][PremulRgba8].
    #[inline(always)]
    pub fn sample(&self, x: u16, y: u16) -> PremulRgba8 {
        let idx = self.width as usize * y as usize + x as usize;
        self.buf[idx]
    }

    /// Consume the pixmap, returning the data as the underlying [`Vec`] of premultiplied RGBA8.
    ///
    /// The pixels are in row-major order.
    pub fn take(self) -> Vec<PremulRgba8> {
        self.buf
    }

    /// Consume the pixmap, returning the data as (unpremultiplied) RGBA8.
    ///
    /// Not fast, but useful for saving to PNG etc.
    ///
    /// The pixels are in row-major order.
    pub fn take_unpremultiplied(self) -> Vec<Rgba8> {
        self.buf
            .into_iter()
            .map(|PremulRgba8 { r, g, b, a }| {
                let alpha = 255.0 / f32::from(a);
                if a != 0 {
                    #[expect(clippy::cast_possible_truncation, reason = "deliberate quantization")]
                    let unpremultiply = |component| (f32::from(component) * alpha + 0.5) as u8;
                    Rgba8 {
                        r: unpremultiply(r),
                        g: unpremultiply(g),
                        b: unpremultiply(b),
                        a,
                    }
                } else {
                    Rgba8 { r, g, b, a }
                }
            })
            .collect()
    }
}
