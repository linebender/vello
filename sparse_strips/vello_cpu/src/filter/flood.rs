// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flood filter implementation.

use super::{FilterBuffer, FilterEffect};
use vello_common::color::{AlphaColor, Srgb};

pub(crate) struct Flood {
    pub color: AlphaColor<Srgb>,
}

impl Flood {
    /// Create a new flood filter with the specified color.
    pub(crate) fn new(color: AlphaColor<Srgb>) -> Self {
        Self { color }
    }
}

impl FilterEffect for Flood {
    fn apply_u8(&self, buffer: &mut FilterBuffer<u8>) {
        // Convert AlphaColor to u8 [0-255] range
        // components[0] = red, components[1] = green, components[2] = blue, components[3] = alpha
        let flood_r = (self.color.components[0] * 255.0) as u8;
        let flood_g = (self.color.components[1] * 255.0) as u8;
        let flood_b = (self.color.components[2] * 255.0) as u8;
        let flood_a = (self.color.components[3] * 255.0) as u8;

        // Fill non-transparent pixels with flood color, preserving alpha mask
        for y in 0..buffer.height() {
            for x in 0..buffer.width() {
                let pixel = buffer.get_pixel(x, y);
                let existing_alpha = pixel[3];

                // Only fill pixels that have some alpha (non-transparent)
                if existing_alpha > 0 {
                    // Combine flood color with existing alpha to preserve transparency levels
                    let combined_alpha = ((existing_alpha as u16 * flood_a as u16) / 255) as u8;
                    buffer.set_pixel(x, y, [flood_r, flood_g, flood_b, combined_alpha]);
                }
            }
        }
    }

    fn apply_f32(&self, buffer: &mut FilterBuffer<f32>) {
        // AlphaColor is already in [0.0-1.0] range for f32
        // components[0] = red, components[1] = green, components[2] = blue, components[3] = alpha
        let flood_r = self.color.components[0];
        let flood_g = self.color.components[1];
        let flood_b = self.color.components[2];
        let flood_a = self.color.components[3];

        // Fill non-transparent pixels with flood color, preserving alpha mask
        for y in 0..buffer.height() {
            for x in 0..buffer.width() {
                let pixel = buffer.get_pixel(x, y);
                let existing_alpha = pixel[3];

                // Only fill pixels that have some alpha (non-transparent)
                if existing_alpha > 0.0 {
                    // Combine flood color with existing alpha to preserve transparency levels
                    let combined_alpha = existing_alpha * flood_a;
                    buffer.set_pixel(x, y, [flood_r, flood_g, flood_b, combined_alpha]);
                }
            }
        }
    }
}
