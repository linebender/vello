// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blur benchmark scene for stress-testing Gaussian blur.
//!
//! Press `m` to add a blurred rectangle, `n` to remove one.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::AlphaColor;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Rect};

const INITIAL_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Simple xorshift64 step (pure function).
fn xorshift64(state: u64) -> u64 {
    let mut s = state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
}

/// Blur benchmark scene state.
#[derive(Debug)]
pub struct BlurBenchmarkScene {
    count: usize,
}

impl BlurBenchmarkScene {
    /// Create a new `BlurBenchmarkScene` with one initial rectangle.
    pub fn new() -> Self {
        Self { count: 1 }
    }
}

impl Default for BlurBenchmarkScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleScene for BlurBenchmarkScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        ctx.set_transform(root_transform);

        let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 2.0,
            edge_mode: EdgeMode::None,
        });

        let mut rng = INITIAL_SEED;
        for _ in 0..self.count {
            rng = xorshift64(rng);
            let x = (rng % 800) as f64;
            rng = xorshift64(rng);
            let y = (rng % 800) as f64;
            rng = xorshift64(rng);
            let r = (rng & 0xFF) as u8;
            let g = ((rng >> 8) & 0xFF) as u8;
            let b = ((rng >> 16) & 0xFF) as u8;
            let color = AlphaColor::from_rgba8(r, g, b, 128);

            ctx.push_filter_layer(filter.clone());
            ctx.set_paint(color);
            ctx.fill_rect(&Rect::new(x, y, x + 1000.0, y + 1000.0));
            ctx.pop_layer();
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!("Blur Benchmark ({} rects)", self.count))
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "m" | "M" => {
                self.count += 1;
                println!("Blur rectangles: {}", self.count);
                true
            }
            "n" | "N" => {
                if self.count > 0 {
                    self.count -= 1;
                    println!("Blur rectangles: {}", self.count);
                }
                true
            }
            _ => false,
        }
    }
}
