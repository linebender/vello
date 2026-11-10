// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Alpha and luminance masks.

use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MaskRepr {
    data: Vec<u8>,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
}

// Note that we are on purpose storing the offset, width and height inside the
// `Arc` to reduce the memory footprint of the struct.
/// A mask.
///
/// A mask covers `width * height` pixels of the render context, starting at
/// its [offset](Self::with_offset), which is `(0, 0)` by default. Outside of
/// these pixels, its value is 0.
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
    pub fn from_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            usize::from(width) * usize::from(height),
            "`data` should have `width * height` length"
        );

        Self(Arc::new(MaskRepr {
            data,
            x: 0,
            y: 0,
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
            x: 0,
            y: 0,
            width: pixmap.width(),
            height: pixmap.height(),
        }))
    }

    /// Place the top-left pixel of the mask at `(x, y)` of the render context.
    ///
    /// This allows a mask to cover only the part of the render context where
    /// it is needed. If this mask shares its data with clones of it, the data
    /// is copied.
    #[must_use]
    pub fn with_offset(mut self, x: u16, y: u16) -> Self {
        let repr = Arc::make_mut(&mut self.0);
        repr.x = x;
        repr.y = y;

        self
    }

    /// Return the pixel of the render context where the top-left pixel of the
    /// mask is placed.
    #[inline(always)]
    pub fn offset(&self) -> (u16, u16) {
        (self.0.x, self.0.y)
    }

    /// Return the width of the mask.
    #[inline(always)]
    pub fn width(&self) -> u16 {
        self.0.width
    }

    /// Return the height of the mask.
    #[inline(always)]
    pub fn height(&self) -> u16 {
        self.0.height
    }

    /// Return the `width * height` values of the mask in row-major order.
    #[inline(always)]
    pub fn data(&self) -> &[u8] {
        &self.0.data
    }

    /// Sample the value at a specific location, relative to the top-left pixel
    /// of the mask.
    ///
    /// This function might panic or yield a wrong result if the location
    /// is out-of-bounds.
    #[inline(always)]
    pub fn sample(&self, x: u16, y: u16) -> u8 {
        let repr = &*self.0;
        debug_assert!(
            x < repr.width && y < repr.height,
            "cannot sample mask outside of its range"
        );

        repr.data[y as usize * repr.width as usize + x as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::Mask;
    use alloc::vec;

    #[test]
    fn with_offset_leaves_clones_unchanged() {
        let mask = Mask::from_parts(vec![1], 1, 1);
        let shifted = mask.clone().with_offset(2, 3);

        assert_eq!(mask.offset(), (0, 0));
        assert_eq!(shifted.offset(), (2, 3));
        assert_eq!(shifted.sample(0, 0), 1);
    }
}
