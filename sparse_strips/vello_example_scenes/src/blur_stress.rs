// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interactive blur stress test scene with reproducible random rectangles.
//!
//! Each rectangle gets a deterministic position, size, color, and blur std_deviation
//! derived from a seeded PRNG. Adding/removing rectangles always produces the same
//! sequence, so removing rect N and re-adding it yields an identical frame.
//!
//! Controls:
//!   `m` / `M` — add a blurred rectangle
//!   `n` / `N` — remove the last rectangle
//!   `b` / `B` — increase blur std_deviation cap (+1.0)
//!   `v` / `V` — decrease blur std_deviation cap (-1.0, min 0.5)

use crate::{ExampleScene, RenderingContext};
use vello_common::color::AlphaColor;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Rect};

const SEED: u64 = 0xB10E_5CEE_E5EE_D000;

/// Simple xorshift64 PRNG — deterministic, no external state.
fn xorshift64(state: u64) -> u64 {
    let mut s = state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
}

/// Extract a `f64` in `[lo, hi)` from a PRNG state, advancing it.
fn next_f64(rng: &mut u64, lo: f64, hi: f64) -> f64 {
    *rng = xorshift64(*rng);
    let t = (*rng as f64) / (u64::MAX as f64); // [0, 1)
    lo + t * (hi - lo)
}

/// Extract a `u8` from a PRNG state, advancing it.
fn next_u8(rng: &mut u64) -> u8 {
    *rng = xorshift64(*rng);
    (*rng & 0xFF) as u8
}

/// Blur stress test scene.
#[derive(Debug)]
pub struct BlurStressScene {
    /// Number of rectangles currently shown.
    count: usize,
    /// Maximum blur std_deviation (rectangles get a random value in `[0.5, max_std]`).
    max_std: f64,
}

impl BlurStressScene {
    pub fn new() -> Self {
        Self {
            count: 10,
            max_std: 10.0,
        }
    }
}

impl Default for BlurStressScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleScene for BlurStressScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        ctx.set_transform(root_transform);

        let mut rng = SEED;

        for _ in 0..self.count {
            // Size (60..800 px per side)
            let w = next_f64(&mut rng, 60.0, 800.0);
            let h = next_f64(&mut rng, 60.0, 800.0);

            // Blur std_deviation (generated before position so margin is known)
            let std_dev = next_f64(&mut rng, 0.5, self.max_std);

            // Inset by ~3*sigma so the blur halo stays within 3840x2160
            let margin = std_dev * 3.0;
            let x = next_f64(&mut rng, margin, (3840.0 - w - margin).max(margin));
            let y = next_f64(&mut rng, margin, (2160.0 - h - margin).max(margin));

            // Color (semi-transparent)
            let r = next_u8(&mut rng);
            let g = next_u8(&mut rng);
            let b = next_u8(&mut rng);
            let color = AlphaColor::from_rgba8(r, g, b, 180);

            let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: std_dev as f32,
                edge_mode: EdgeMode::None,
            });

            ctx.push_filter_layer(filter);
            ctx.set_paint(color);
            ctx.fill_rect(&Rect::new(x, y, x + w, y + h));
            ctx.pop_layer();
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "Blur Stress: {} rects, max σ = {:.1}  [m/n: ±rect, b/v: ±σ]",
            self.count, self.max_std
        ))
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "m" | "M" => {
                self.count += 10;
                true
            }
            "n" | "N" => {
                self.count = self.count.saturating_sub(10);
                true
            }
            "b" | "B" => {
                self.max_std += 1.0;
                true
            }
            "v" | "V" => {
                self.max_std = (self.max_std - 1.0).max(0.5);
                true
            }
            _ => false,
        }
    }
}
