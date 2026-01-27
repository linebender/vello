// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flood filter implementation.

use super::FilterEffect;
use crate::layer_manager::LayerManager;
use vello_common::color::{AlphaColor, Srgb};
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
        pixmap.data_mut().fill(self.color.premultiply().to_rgba8());
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        pixmap.data_mut().fill(self.color.premultiply().to_rgba8());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer_manager::LayerManager;
    use vello_common::color::{PremulRgba8, Srgb};

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
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 128
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
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 128
                    }
                );
            }
        }
    }
}
