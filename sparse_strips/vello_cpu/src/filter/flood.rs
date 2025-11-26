// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flood filter implementation.

use super::FilterEffect;
use crate::layer_manager::LayerManager;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

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
    fn execute_lowp(&self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        // Convert AlphaColor to u8 [0-255] range
        let flood_r = (self.color.components[0] * 255.0) as u8;
        let flood_g = (self.color.components[1] * 255.0) as u8;
        let flood_b = (self.color.components[2] * 255.0) as u8;
        let flood_a = (self.color.components[3] * 255.0) as u8;

        // Premultiply RGB by alpha for PremulRgba8
        let premul_r = ((flood_r as u16 * flood_a as u16) / 255) as u8;
        let premul_g = ((flood_g as u16 * flood_a as u16) / 255) as u8;
        let premul_b = ((flood_b as u16 * flood_a as u16) / 255) as u8;

        let flood_color = PremulRgba8 {
            r: premul_r,
            g: premul_g,
            b: premul_b,
            a: flood_a,
        };

        // Fill ALL pixels with flood color (entire subregion)
        pixmap.data_mut().fill(flood_color);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        let flood_r = self.color.components[0];
        let flood_g = self.color.components[1];
        let flood_b = self.color.components[2];
        let flood_a = self.color.components[3];

        // Premultiply RGB by alpha for PremulRgba8
        let flood_color = PremulRgba8 {
            r: (flood_r * flood_a * 255.0) as u8,
            g: (flood_g * flood_a * 255.0) as u8,
            b: (flood_b * flood_a * 255.0) as u8,
            a: (flood_a * 255.0) as u8,
        };

        // Fill ALL pixels with flood color (entire subregion)
        pixmap.data_mut().fill(flood_color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_manager::LayerManager;
    use vello_common::color::Srgb;

    /// Test flood with semi-transparent color - verifies correct premultiplication.
    #[test]
    fn test_flood_semi_transparent_lowp() {
        let mut pixmap = Pixmap::new(2, 2);
        let mut layer_manager = LayerManager::new();

        // Semi-transparent white (50% alpha)
        let color = AlphaColor {
            components: [1.0, 1.0, 1.0, 0.5],
            cs: std::marker::PhantomData::<Srgb>,
        };
        let flood = Flood::new(color);
        flood.execute_lowp(&mut pixmap, &mut layer_manager);

        // RGB should be premultiplied by alpha: 255 * 0.5 = 127-128
        for y in 0..2 {
            for x in 0..2 {
                let pixel = pixmap.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 127,
                        g: 127,
                        b: 127,
                        a: 127
                    }
                );
            }
        }
    }

    /// Test flood highp with semi-transparent color - verifies correct premultiplication.
    #[test]
    fn test_flood_semi_transparent_highp() {
        let mut pixmap = Pixmap::new(2, 2);
        let mut layer_manager = LayerManager::new();

        // Semi-transparent white (50% alpha)
        let color = AlphaColor {
            components: [1.0, 1.0, 1.0, 0.5],
            cs: std::marker::PhantomData::<Srgb>,
        };
        let flood = Flood::new(color);
        flood.execute_highp(&mut pixmap, &mut layer_manager);

        // RGB should be premultiplied by alpha: 255 * 0.5 = 127-128
        for y in 0..2 {
            for x in 0..2 {
                let pixel = pixmap.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 127,
                        g: 127,
                        b: 127,
                        a: 127
                    }
                );
            }
        }
    }
}
