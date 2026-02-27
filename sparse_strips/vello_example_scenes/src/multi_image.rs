// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene that renders multiple axis-aligned images at randomized positions across the viewport.
//! Press "a"/"A" to add 50/1 images, "d"/"D" to remove 50/1 images.

use core::any::Any;
use parley_draw::ImageCache;

use crate::{ExampleScene, RenderingContext, TextConfig};
use std::fmt::{Debug, Formatter, Result};
use vello_common::color::palette::css::WHITE;
use vello_common::kurbo::{Affine, Rect, Vec2};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{Extend, ImageQuality, ImageSampler};

const BATCH_SIZE: usize = 50;
const IMG_W: f64 = 640.0;
const IMG_H: f64 = 480.0;
const MIN_SCALE: f64 = 0.1;
const MAX_SCALE: f64 = 1.5;

/// Pre-computed placement for a single image (normalised 0..1 coordinates).
struct ImagePlacement {
    nx: f64,
    ny: f64,
    scale: f64,
}

/// Scene state for the multi-image example.
pub struct MultiImageScene {
    img_source: ImageSource,
    rng: Rng,
    placements: Vec<ImagePlacement>,
}

impl Debug for MultiImageScene {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("MultiImageScene")
            .field("count", &self.placements.len())
            .finish_non_exhaustive()
    }
}

impl MultiImageScene {
    /// Create a new multi-image scene starting with `BATCH_SIZE` images.
    pub fn new(img_source: ImageSource) -> Self {
        let mut scene = Self {
            img_source,
            rng: Rng::new(0xDEAD_BEEF_CAFE_BABE),
            placements: Vec::new(),
        };
        scene.placements.push(ImagePlacement {
            nx: 0.1,
            ny: 0.1,
            scale: 2.0,
        });
        scene
    }

    fn add_one(&mut self) {
        self.placements.push(ImagePlacement {
            nx: self.rng.range_f64(0.0, 1.0),
            ny: self.rng.range_f64(0.0, 1.0),
            scale: self.rng.range_f64(MIN_SCALE, MAX_SCALE),
        });
    }

    fn add_batch(&mut self) {
        for _ in 0..BATCH_SIZE {
            self.add_one();
        }
    }
}

impl ExampleScene for MultiImageScene {
    fn render(
        &mut self,
        ctx: &mut impl RenderingContext,
        root_transform: Affine,
        _glyph_caches: &mut dyn Any,
        _image_cache: &mut ImageCache,
        _text_config: &TextConfig,
    ) {
        let vw = ctx.width() as f64;
        let vh = ctx.height() as f64;
        let t = root_transform.translation();
        let snapped_root = root_transform.with_translation(Vec2::new(t.x.round(), t.y.round()));

        for p in &self.placements {
            let x = (p.nx * vw).round();
            let y = (p.ny * vh).round();
            let w = (IMG_W * p.scale).round();
            let h = (IMG_H * p.scale).round();

            ctx.set_transform(snapped_root * Affine::translate((x, y)));
            ctx.set_paint_transform(Affine::scale(p.scale));
            ctx.set_paint(Image {
                image: self.img_source.clone(),
                sampler: ImageSampler {
                    x_extend: Extend::Pad,
                    y_extend: Extend::Pad,
                    quality: ImageQuality::Medium,
                    alpha: 1.0,
                },
            });
            ctx.fill_rect(&Rect::new(0.0, 0.0, w, h));
        }

        ctx.set_transform(Affine::IDENTITY);
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "a" => {
                self.add_batch();
                true
            }
            "A" => {
                self.add_one();
                true
            }
            "d" => {
                let new_len = self.placements.len().saturating_sub(BATCH_SIZE);
                self.placements.truncate(new_len);
                true
            }
            "D" => {
                self.placements.pop();
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!("{} images", self.placements.len()))
    }
}

/// Minimal xorshift64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let t =
            (self.next_u64() & 0x000F_FFFF_FFFF_FFFF) as f64 / (0x0010_0000_0000_0000_u64 as f64);
        lo + t * (hi - lo)
    }
}
