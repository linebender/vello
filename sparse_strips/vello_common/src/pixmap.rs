// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

/// A pixmap backed by u8.
#[derive(Debug)]
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

    /// Returns the underlying data as premultiplied RGBA8.
    pub fn data(&self) -> &[u8] {
        &self.buf
    }

    /// Convert from premultiplied to separate alpha.
    ///
    /// Not fast, but useful for saving to PNG etc.
    pub fn unpremultiply(&mut self) {
        for rgba in self.buf.chunks_exact_mut(4) {
            let alpha = 255.0 / rgba[3] as f32;
            if alpha != 0.0 {
                rgba[0] = (rgba[0] as f32 * alpha).round().min(255.0) as u8;
                rgba[1] = (rgba[1] as f32 * alpha).round().min(255.0) as u8;
                rgba[2] = (rgba[2] as f32 * alpha).round().min(255.0) as u8;
            }
        }
    }
}
