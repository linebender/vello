// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Alpha and luminance masks.

use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;

#[derive(Debug, PartialEq, Eq)]
struct MaskRepr {
    data: Vec<u8>,
    width: i16,
    height: i16,
}

// Note that we are on purpose storing width and height inside the `Arc`
// to reduce the memory footprint of the struct.
/// A mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask(Arc<MaskRepr>);

impl Mask {
    /// Create a new alpha mask from the pixmap.
    pub fn new_alpha(pixmap: &Pixmap) -> Self {
        Self::new_with(pixmap, true)
    }

    /// Create a new luminance mask from the pixmap.
    pub fn new_luminance(pixmap: &Pixmap) -> Self {
        Self::new_with(pixmap, false)
    }

    /// Create a new mask from the given alpha data.
    ///
    /// The `data` vector must be of length `width * height` exactly.
    ///
    /// The pixels are in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if the `data` vector is not of length `width * height`.
    pub fn from_parts(data: Vec<u8>, width: i16, height: i16) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "`data` should have `width * height` length"
        );

        Self(Arc::new(MaskRepr {
            data,
            width,
            height,
        }))
    }

    fn new_with(pixmap: &Pixmap, alpha_mask: bool) -> Self {
        let data = pixmap
            .data()
            .iter()
            .map(|pixel| {
                if alpha_mask {
                    pixel.a
                } else {
                    let r = f32::from(pixel.r) / 255.;
                    let g = f32::from(pixel.g) / 255.;
                    let b = f32::from(pixel.b) / 255.;

                    // See CSS Masking Module Level 1 § 7.10.1
                    // <https://www.w3.org/TR/css-masking-1/#MaskValues>
                    // and Filter Effects Module Level 1 § 9.6
                    // <https://www.w3.org/TR/filter-effects-1/#elementdef-fecolormatrix>.
                    // Note r, g and b are premultiplied by alpha.
                    let luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
                    #[expect(clippy::cast_possible_truncation, reason = "This cannot overflow")]
                    {
                        (luma * 255.0 + 0.5) as u8
                    }
                }
            })
            .collect::<Vec<u8>>();

        Self(Arc::new(MaskRepr {
            data,
            width: pixmap.width(),
            height: pixmap.height(),
        }))
    }

    /// Return the width of the mask.
    #[inline(always)]
    pub fn width(&self) -> i16 {
        self.0.width
    }

    /// Return the height of the mask.
    #[inline(always)]
    pub fn height(&self) -> i16 {
        self.0.height
    }

    /// Sample the value at a specific location.
    ///
    /// This function might panic or yield a wrong result if the location
    /// is out-of-bounds.
    #[inline(always)]
    pub fn sample(&self, x: i16, y: i16) -> u8 {
        let repr = &*self.0;
        debug_assert!(
            x < repr.width && x > 0 && y < repr.height && y > 0,
            "cannot sample mask outside of its range"
        );

        // TODO
        repr.data[y as usize * repr.width as usize + x as usize]
    }
}
