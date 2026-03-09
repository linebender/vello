// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene with bouncing, rotating, semi-transparent rectangles.
//! Press "a"/"d" to add/remove 100 rects, "r" to toggle rotation.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Rect};

const BATCH_SIZE: usize = 100;
const MIN_SIZE: f64 = 20.0;
const MAX_SIZE: f64 = 100.0;
const MIN_SPEED: f64 = 0.5;
const MAX_SPEED: f64 = 3.0;
const ALPHA: f32 = 0.6;

struct RectState {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    w: f64,
    h: f64,
    angle: f64,
    angular_vel: f64,
    color: [u8; 3],
}

/// Interactive scene with bouncing, rotating, semi-transparent rectangles.
pub struct RotatedRectsScene {
    rng: Rng,
    rects: Vec<RectState>,
    rotated: bool,
}

impl std::fmt::Debug for RotatedRectsScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotatedRectsScene")
            .field("count", &self.rects.len())
            .field("rotated", &self.rotated)
            .finish_non_exhaustive()
    }
}

impl RotatedRectsScene {
    /// Create a new scene with an initial batch of rectangles.
    pub fn new() -> Self {
        let mut scene = Self {
            rng: Rng::new(0xCAFE_BABE_DEAD_BEEF),
            rects: Vec::new(),
            rotated: false,
        };
        scene.add_batch();
        scene
    }

    fn add_one(&mut self, vw: f64, vh: f64) {
        let w = self.rng.range_f64(MIN_SIZE, MAX_SIZE);
        let h = self.rng.range_f64(MIN_SIZE, MAX_SIZE);
        self.rects.push(RectState {
            x: self.rng.range_f64(0.0, (vw - w).max(1.0)),
            y: self.rng.range_f64(0.0, (vh - h).max(1.0)),
            vx: self.rng.range_f64(MIN_SPEED, MAX_SPEED) * self.rng.sign(),
            vy: self.rng.range_f64(MIN_SPEED, MAX_SPEED) * self.rng.sign(),
            w,
            h,
            angle: self.rng.range_f64(0.0, std::f64::consts::TAU),
            angular_vel: self.rng.range_f64(0.005, 0.03) * self.rng.sign(),
            color: [self.rng.range_u8(), self.rng.range_u8(), self.rng.range_u8()],
        });
    }

    fn add_batch(&mut self) {
        for _ in 0..BATCH_SIZE {
            // Use a default viewport size; positions will clamp on first render.
            self.add_one(1920.0, 1080.0);
        }
    }

    fn step(&mut self, vw: f64, vh: f64) {
        for r in &mut self.rects {
            r.x += r.vx;
            r.y += r.vy;
            r.angle += r.angular_vel;

            if r.x < 0.0 {
                r.x = 0.0;
                r.vx = r.vx.abs();
            } else if r.x + r.w > vw {
                r.x = vw - r.w;
                r.vx = -r.vx.abs();
            }

            if r.y < 0.0 {
                r.y = 0.0;
                r.vy = r.vy.abs();
            } else if r.y + r.h > vh {
                r.y = vh - r.h;
                r.vy = -r.vy.abs();
            }
        }
    }
}

impl ExampleScene for RotatedRectsScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let vw = ctx.width() as f64;
        let vh = ctx.height() as f64;

        self.step(vw, vh);

        for r in &self.rects {
            let cx = r.x + r.w / 2.0;
            let cy = r.y + r.h / 2.0;

            let transform = if self.rotated {
                root_transform
                    * Affine::translate((cx, cy))
                    * Affine::rotate(r.angle)
                    * Affine::translate((-r.w / 2.0, -r.h / 2.0))
            } else {
                root_transform * Affine::translate((r.x, r.y))
            };

            ctx.set_transform(transform);
            ctx.set_paint(AlphaColor::<Srgb>::new([
                r.color[0] as f32 / 255.0,
                r.color[1] as f32 / 255.0,
                r.color[2] as f32 / 255.0,
                ALPHA,
            ]));
            ctx.fill_rect(&Rect::new(0.0, 0.0, r.w, r.h));
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "a" => {
                self.add_batch();
                true
            }
            "d" => {
                let new_len = self.rects.len().saturating_sub(BATCH_SIZE);
                self.rects.truncate(new_len);
                true
            }
            "r" => {
                self.rotated = !self.rotated;
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "{} rects{}",
            self.rects.len(),
            if self.rotated { " (rotated)" } else { "" }
        ))
    }
}

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

    fn range_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }

    fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }
}
