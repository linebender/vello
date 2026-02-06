// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `feOffset` filter primitive implementation.

use vello_common::filter::offset::Offset;
use vello_common::pixmap::Pixmap;

use super::FilterEffect;
use super::shift::offset_pixels;
use crate::layer_manager::LayerManager;

impl FilterEffect for Offset {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _: &mut LayerManager) {
        offset_pixels(pixmap, self.dx, self.dy);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _: &mut LayerManager) {
        offset_pixels(pixmap, self.dx, self.dy);
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
