// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Alpha and luminance masks.

use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// A mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    data: Arc<Vec<u8>>,
    width: u16,
    height: u16,
}

impl Mask {
    /// Create a new alpha mask from the pixmap.
    pub fn new_alpha(pixmap: &Pixmap) -> Self {
        Self::new_with(pixmap, true)
    }

    /// Create a new luminance mask from the pixmap.
    pub fn new_luminance(pixmap: &Pixmap) -> Self {
        Self::new_with(pixmap, false)
    }

    fn new_with(pixmap: &Pixmap, alpha_mask: bool) -> Self {
        let mut data = Vec::with_capacity(pixmap.width() as usize * pixmap.height() as usize);

        for pixel in pixmap.data().chunks_exact(4) {
            if alpha_mask {
                data.push(pixel[3]);
            } else {
                let mut r = pixel[0] as f32 / 255.0;
                let mut g = pixel[1] as f32 / 255.0;
                let mut b = pixel[2] as f32 / 255.0;
                let a = pixel[3] as f32 / 255.0;

                if pixel[3] != 0 {
                    r /= a;
                    g /= a;
                    b /= a;
                }

                // See https://www.w3.org/TR/filter-effects-1/#elementdef-fecolormatrix
                let luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
                #[allow(clippy::cast_possible_truncation, reason = "This cannot overflow")]
                data.push(((luma * a) * 255.0 + 0.5) as u8);
            }
        }

        Self {
            data: Arc::new(data),
            width: pixmap.width,
            height: pixmap.height,
        }
    }

    /// Return the width of the mask.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the mask.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Sample the value at a specific location.
    ///
    /// This function might panic or yield a wrong result if the location
    /// is out-of-bounds.
    pub fn sample(&self, x: u16, y: u16) -> u8 {
        debug_assert!(
            x < self.width && y < self.height,
            "cannot sample mask outside of its range"
        );

        self.data[y as usize * self.width as usize + x as usize]
    }
}
