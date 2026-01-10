// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `feOffset` filter primitive implementation.

use vello_common::pixmap::Pixmap;

use super::FilterEffect;
use super::shift::offset_pixels;
use crate::layer_manager::LayerManager;

/// Translation/shift filter.
///
/// This shifts the input image by `(dx, dy)` in device pixel space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Offset {
    dx: f32,
    dy: f32,
}

impl Offset {
    pub(crate) fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    fn execute(self, pixmap: &mut Pixmap, _layer_manager: &mut LayerManager) {
        offset_pixels(pixmap, self.dx, self.dy);
    }
}

impl FilterEffect for Offset {
    fn execute_lowp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        self.execute(pixmap, layer_manager);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, layer_manager: &mut LayerManager) {
        self.execute(pixmap, layer_manager);
    }
}

#[cfg(test)]
mod tests {
    use super::Offset;
    use crate::filter::FilterEffect;
    use crate::layer_manager::LayerManager;
    use vello_common::peniko::color::PremulRgba8;
    use vello_common::pixmap::Pixmap;

    #[test]
    fn offset_moves_pixels_and_clears_uncovered_area() {
        let mut layer_manager = LayerManager::new();
        let mut pixmap = Pixmap::new(4, 3);
        pixmap.set_pixel(1, 1, PremulRgba8::from_u32(0xff_00_00_ff)); // premul red, opaque

        Offset::new(2.0, -1.0).execute_lowp(&mut pixmap, &mut layer_manager);

        // Original pixel (1,1) moved to (3,0).
        assert_eq!(pixmap.sample(3, 0), PremulRgba8::from_u32(0xff_00_00_ff));
        // Original location cleared.
        assert_eq!(pixmap.sample(1, 1), PremulRgba8::from_u32(0));
    }
}
