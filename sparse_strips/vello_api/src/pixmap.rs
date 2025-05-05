// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

use alloc::vec;
use alloc::vec::Vec;

/// A pixmap backed by u8.
#[derive(Debug, Clone)]
pub struct Pixmap {
    /// Width of the pixmap in pixels.  
    pub width: u16,
    /// Height of the pixmap in pixels.
    pub height: u16,
    /// Buffer of the pixmap in RGBA format.
    pub buf: Vec<u8>,
}

impl Pixmap {
    /// Create a new pixmap with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let buf = vec![0; width as usize * height as usize * 4];
        Self { width, height, buf }
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "cannot overflow in this case"
    )]
    /// Apply an alpha value to the whole pixmap.
    pub fn multiply_alpha(&mut self, alpha: u8) {
        for comp in self.data_mut() {
            *comp = ((alpha as u16 * *comp as u16) / 255) as u8;
        }
    }

    /// Create a pixmap from a PNG file.
    #[cfg(feature = "png")]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cannot overflow in this case"
    )]
    pub fn from_png(data: &[u8]) -> Result<Self, png::DecodingError> {
        let mut decoder = png::Decoder::new(data);
        decoder.set_transformations(png::Transformations::ALPHA);

        let mut reader = decoder.read_info()?;
        let mut img_data = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut img_data)?;

        let decoded_data = match info.color_type {
            // We set a transformation to always convert to alpha.
            png::ColorType::Rgb => unreachable!(),
            png::ColorType::Grayscale => unreachable!(),
            // I believe the above transformation also expands indexed images.
            png::ColorType::Indexed => unreachable!(),
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
            width: info.width as u16,
            height: info.height as u16,
            buf: premultiplied,
        })
    }

    /// Returns a reference to the underlying data as premultiplied RGBA8.
    pub fn data(&self) -> &[u8] {
        &self.buf
    }

    /// Returns a mutable reference to the underlying data as premultiplied RGBA8.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Sample a pixel from the pixmap.
    #[inline(always)]
    pub fn sample(&self, x: u16, y: u16) -> &[u8] {
        let idx = 4 * (self.width as usize * y as usize + x as usize);
        &self.buf[idx..][..4]
    }

    /// Convert from premultiplied to separate alpha.
    ///
    /// Not fast, but useful for saving to PNG etc.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cannot overflow in this case"
    )]
    pub fn unpremultiply(&mut self) {
        for rgba in self.buf.chunks_exact_mut(4) {
            let alpha = 255.0 / rgba[3] as f32;

            if rgba[3] != 0 {
                rgba[0] = (rgba[0] as f32 * alpha + 0.5) as u8;
                rgba[1] = (rgba[1] as f32 * alpha + 0.5) as u8;
                rgba[2] = (rgba[2] as f32 * alpha + 0.5) as u8;
            }
        }
    }
}
