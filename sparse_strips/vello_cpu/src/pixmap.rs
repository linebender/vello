// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple pixmap type.

/// A pixmap backed by u8.
#[derive(Debug)]
pub struct Pixmap {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) buf: Vec<u8>,
}

impl Pixmap {
    /// Create a new pixmap.
    pub fn new(width: usize, height: usize) -> Self {
        let buf = vec![0; width * height * 4];
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
            let alpha = rgba[3] as f32 * (1.0 / 255.0);
            if alpha != 0.0 {
                rgba[0] = (rgba[0] as f32 / alpha).round().min(255.0) as u8;
                rgba[1] = (rgba[1] as f32 / alpha).round().min(255.0) as u8;
                rgba[2] = (rgba[2] as f32 / alpha).round().min(255.0) as u8;
            }
        }
    }
}