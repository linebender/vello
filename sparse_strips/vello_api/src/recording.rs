// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recordings are convenience structures for drawing the same path multiple times.
//!
//! This can be used for (some) COLR glyphs.

use peniko::kurbo::Affine;

use crate::PaintScene;

pub trait RecordScene<Scene: PaintScene>: PaintScene {
    fn draw_into(
        &self,
        scene: &mut Scene,
        x_offset: i32,
        y_offset: i32,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>;
}

pub trait TransformedRecording<Scene: PaintScene>: PaintScene {
    fn draw_into(
        &self,
        scene: &mut Scene,
        transform: Affine,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>;
}
