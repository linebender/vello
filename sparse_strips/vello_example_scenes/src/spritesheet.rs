// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Spritesheet example scene.

use std::io::Cursor;

use vello_common::geometry::RectU16;
use vello_common::kurbo::Affine;
use vello_common::peniko::ImageQuality;
use vello_common::pixmap::Pixmap;
use vello_hybrid::{SampleRect, TextureId};

use crate::{ExampleScene, RenderingContext};

/// The [`TextureId`] under which the spritesheet texture must be bound at render time.
pub const SPRITESHEET_TEXTURE_ID: TextureId = TextureId(0);

/// The four sprite source regions within `glyphs_colr_noto.png`.
const SPRITES: [RectU16; 4] = [
    // Checkmark.
    RectU16::new(0, 0, 56, 56),
    // Eyes.
    RectU16::new(56, 0, 112, 42),
    // Confetti.
    RectU16::new(0, 56, 54, 112),
    // Cowboy hat face.
    RectU16::new(56, 42, 115, 100),
];

/// The spritesheet scene.
///
/// Draws many rectangular regions of one externally bound texture in a single
/// [`RenderingContext::draw_texture_rects`] call. The host is responsible for uploading the
/// spritesheet (see [`SpritesheetScene::read_spritesheet`]) and binding it under
/// [`SPRITESHEET_TEXTURE_ID`] at render time.
#[derive(Debug, Default)]
pub struct SpritesheetScene {}

impl SpritesheetScene {
    /// Create a new spritesheet scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode the spritesheet into a [`Pixmap`]. Hosts should call this once per device when
    /// uploading the texture.
    pub fn read_spritesheet() -> Pixmap {
        let data = include_bytes!("../../vello_sparse_tests/tests/assets/glyphs_colr_noto.png");
        Pixmap::from_png(Cursor::new(data)).unwrap()
    }
}

impl ExampleScene for SpritesheetScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        const SCALES: [f64; 5] = [0.5, 0.625, 0.75, 0.875, 1.];
        const ROTATIONS: [f64; 7] = [-0.42, -0.22, -0.08, 0., 0.10, 0.24, 0.40];
        const SKEWS: [(f64, f64); 11] = [
            (0., 0.),
            (0.18, 0.),
            (-0.18, 0.),
            (0., 0.16),
            (0., -0.16),
            (0.12, 0.10),
            (-0.12, 0.10),
            (0.12, -0.10),
            (-0.12, -0.10),
            (0.22, -0.06),
            (-0.22, 0.06),
        ];

        const COLS: usize = 22;
        const ROWS: usize = 15;
        const CELL_W: f64 = 80.;
        const CELL_H: f64 = 80.;

        ctx.set_transform(root_transform);

        let mut rects = Vec::with_capacity(COLS * ROWS);
        for row in 0..ROWS {
            for col in 0..COLS {
                let i = row * COLS + col;
                let sprite = SPRITES[i % SPRITES.len()];
                let scale = SCALES[i % SCALES.len()];
                let rotation = ROTATIONS[i % ROTATIONS.len()];
                let (skew_x, skew_y) = SKEWS[i % SKEWS.len()];

                let cx = col as f64 * CELL_W + CELL_W * 0.5;
                let cy = row as f64 * CELL_H + CELL_H * 0.5;

                let half_w = f64::from(sprite.width()) * 0.5;
                let half_h = f64::from(sprite.height()) * 0.5;

                // This per-rect transform maps the local source region (with origin (0,0) at
                // the sampled region's top-left corner) into the destination.
                let transform = Affine::translate((cx, cy))
                    * Affine::rotate(rotation)
                    * Affine::skew(skew_x, skew_y)
                    * Affine::scale(scale)
                    * Affine::translate((-half_w, -half_h));

                rects.push(SampleRect {
                    source_region: sprite,
                    transform,
                });
            }
        }

        ctx.draw_texture_rects(SPRITESHEET_TEXTURE_ID, ImageQuality::Medium, rects);
    }
}
