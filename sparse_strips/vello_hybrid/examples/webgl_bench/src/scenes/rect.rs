// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Animated rectangles benchmark scene.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use super::{BenchScene, Param, ParamKind};
use crate::rng::Rng;
use smallvec::smallvec;
use vello_common::kurbo::{Point, Rect};
use vello_common::peniko::{
    Color, ColorStop, ColorStops, Extend, Gradient, LinearGradientPosition, color::DynamicColor,
};
use vello_hybrid::Scene;

/// An animated rectangle with position, velocity, color.
#[derive(Debug)]
struct AnimatedRect {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    color: Color,
    color2: Color,
}

/// Benchmark scene that renders many animated rectangles.
#[derive(Debug)]
pub struct RectScene {
    num_rects: usize,
    speed: f64,
    paint_mode: u32,
    rect_size: f64,
    rects: Vec<AnimatedRect>,
    rng: Rng,
    last_time: f64,
}

impl Default for RectScene {
    fn default() -> Self {
        Self::new()
    }
}

impl RectScene {
    /// Create a new rectangle benchmark scene with default parameters.
    pub fn new() -> Self {
        Self {
            num_rects: 500,
            speed: 5.0,
            paint_mode: 0,
            rect_size: 50.0,
            rects: Vec::new(),
            rng: Rng::new(0xDEAD_BEEF),
            last_time: 0.0,
        }
    }

    /// Grow or shrink the rect list to match `self.num_rects`, preserving existing rects.
    fn resize_rects(&mut self, w: f64, h: f64) {
        if self.rects.len() < self.num_rects {
            self.rects.reserve(self.num_rects - self.rects.len());
            while self.rects.len() < self.num_rects {
                let r = random_rect(&mut self.rng, w, h);
                self.rects.push(r);
            }
        } else {
            self.rects.truncate(self.num_rects);
        }
    }
}

fn random_rect(rng: &mut Rng, w: f64, h: f64) -> AnimatedRect {
    AnimatedRect {
        x: rng.f64() * w,
        y: rng.f64() * h,
        vx: (rng.f64() - 0.5) * 200.0,
        vy: (rng.f64() - 0.5) * 200.0,
        color: rng.color(200),
        color2: rng.color(200),
    }
}

impl BenchScene for RectScene {
    fn name(&self) -> &str {
        "Rectangles"
    }

    fn params(&self) -> Vec<Param> {
        vec![
            Param {
                name: "num_rects",
                label: "Rectangles",
                kind: ParamKind::Slider {
                    min: 1.0,
                    max: 200_000.0,
                    step: 1.0,
                },
                value: self.num_rects as f64,
            },
            Param {
                name: "speed",
                label: "Speed",
                kind: ParamKind::Slider {
                    min: 1.0,
                    max: 30.0,
                    step: 0.1,
                },
                value: self.speed,
            },
            Param {
                name: "paint_mode",
                label: "Paint",
                kind: ParamKind::Select(vec![("Solid", 0.0), ("Gradient", 1.0)]),
                value: self.paint_mode as f64,
            },
            Param {
                name: "rect_size",
                label: "Rect Size",
                kind: ParamKind::Slider {
                    min: 5.0,
                    max: 500.0,
                    step: 1.0,
                },
                value: self.rect_size,
            },
        ]
    }

    fn set_param(&mut self, name: &str, value: f64) {
        match name {
            "num_rects" => self.num_rects = value as usize,
            "speed" => self.speed = value,
            "paint_mode" => self.paint_mode = value as u32,
            "rect_size" => self.rect_size = value,
            _ => {}
        }
    }

    fn render(&mut self, scene: &mut Scene, width: u32, height: u32, time: f64) {
        let w = width as f64;
        let h = height as f64;

        // Ensure rect count matches (preserving existing rects).
        if self.rects.len() != self.num_rects {
            self.resize_rects(w, h);
        }

        let dt = if self.last_time > 0.0 {
            ((time - self.last_time) / 1000.0) * self.speed
        } else {
            0.0
        };
        self.last_time = time;

        let size = self.rect_size;

        for r in &mut self.rects {
            r.x += r.vx * dt;
            r.y += r.vy * dt;

            if r.x < 0.0 {
                r.x = 0.0;
                r.vx = r.vx.abs();
            } else if r.x + size > w {
                r.x = w - size;
                r.vx = -r.vx.abs();
            }
            if r.y < 0.0 {
                r.y = 0.0;
                r.vy = r.vy.abs();
            } else if r.y + size > h {
                r.y = h - size;
                r.vy = -r.vy.abs();
            }

            let rect = Rect::new(r.x, r.y, r.x + size, r.y + size);

            if self.paint_mode == 0 {
                scene.set_paint(r.color);
            } else {
                let gradient = Gradient {
                    kind: LinearGradientPosition {
                        start: Point::new(r.x, r.y),
                        end: Point::new(r.x + size, r.y + size),
                    }
                    .into(),
                    stops: ColorStops(smallvec![
                        ColorStop {
                            offset: 0.0,
                            color: DynamicColor::from_alpha_color(r.color),
                        },
                        ColorStop {
                            offset: 1.0,
                            color: DynamicColor::from_alpha_color(r.color2),
                        },
                    ]),
                    extend: Extend::Pad,
                    ..Default::default()
                };
                scene.set_paint(gradient);
            }

            scene.fill_rect(&rect);
        }
    }
}
