// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stroked lines benchmark scene.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use super::{BenchScene, Param, ParamKind};
use crate::rng::Rng;
use vello_common::kurbo::{Affine, BezPath, Cap, Stroke};
use vello_common::peniko::Color;
use vello_hybrid::{Scene, WebGlRenderer};

/// An animated line with two endpoints, velocities, and a color.
#[derive(Debug)]
struct AnimatedLine {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    vx0: f64,
    vy0: f64,
    vx1: f64,
    vy1: f64,
    color: Color,
}

/// Benchmark scene that strokes many animated random lines.
#[derive(Debug)]
pub struct LinesScene {
    num_lines: usize,
    stroke_width: f64,
    /// 0 = Butt, 1 = Square, 2 = Round
    cap: u32,
    speed: f64,
    lines: Vec<AnimatedLine>,
    rng: Rng,
    last_time: f64,
}

impl Default for LinesScene {
    fn default() -> Self {
        Self::new()
    }
}

impl LinesScene {
    /// Create a new lines benchmark scene.
    pub fn new() -> Self {
        Self {
            num_lines: 200,
            stroke_width: 3.0,
            cap: 0,
            speed: 5.0,
            lines: Vec::new(),
            rng: Rng::new(0xBEEF_CAFE),
            last_time: 0.0,
        }
    }

    fn resize_lines(&mut self, w: f64, h: f64) {
        if self.lines.len() < self.num_lines {
            self.lines.reserve(self.num_lines - self.lines.len());
            while self.lines.len() < self.num_lines {
                self.lines.push(random_line(&mut self.rng, w, h));
            }
        } else {
            self.lines.truncate(self.num_lines);
        }
    }
}

fn random_line(rng: &mut Rng, w: f64, h: f64) -> AnimatedLine {
    AnimatedLine {
        x0: rng.f64() * w,
        y0: rng.f64() * h,
        x1: rng.f64() * w,
        y1: rng.f64() * h,
        vx0: (rng.f64() - 0.5) * 200.0,
        vy0: (rng.f64() - 0.5) * 200.0,
        vx1: (rng.f64() - 0.5) * 200.0,
        vy1: (rng.f64() - 0.5) * 200.0,
        color: rng.color(150),
    }
}

fn bounce(pos: &mut f64, vel: &mut f64, max: f64) {
    if *pos < 0.0 {
        *pos = 0.0;
        *vel = vel.abs();
    } else if *pos > max {
        *pos = max;
        *vel = -vel.abs();
    }
}

impl BenchScene for LinesScene {
    fn name(&self) -> &str {
        "Lines"
    }

    fn params(&self) -> Vec<Param> {
        vec![
            Param {
                name: "num_lines",
                label: "Lines",
                kind: ParamKind::Slider {
                    min: 1.0,
                    max: 3_000.0,
                    step: 1.0,
                },
                value: self.num_lines as f64,
            },
            Param {
                name: "stroke_width",
                label: "Stroke Width",
                kind: ParamKind::Slider {
                    min: 0.5,
                    max: 50.0,
                    step: 0.5,
                },
                value: self.stroke_width,
            },
            Param {
                name: "cap",
                label: "Cap",
                kind: ParamKind::Select(vec![
                    ("Butt", 0.0),
                    ("Square", 1.0),
                    ("Round", 2.0),
                ]),
                value: self.cap as f64,
            },
            Param {
                name: "speed",
                label: "Speed",
                kind: ParamKind::Slider {
                    min: 0.0,
                    max: 30.0,
                    step: 0.1,
                },
                value: self.speed,
            },
        ]
    }

    fn set_param(&mut self, name: &str, value: f64) {
        match name {
            "num_lines" => self.num_lines = value as usize,
            "stroke_width" => self.stroke_width = value,
            "cap" => self.cap = value as u32,
            "speed" => self.speed = value,
            _ => {}
        }
    }

    fn render(
        &mut self,
        scene: &mut Scene,
        _renderer: &mut WebGlRenderer,
        width: u32,
        height: u32,
        time: f64,
        view: Affine,
    ) {
        let w = width as f64;
        let h = height as f64;

        if self.lines.len() != self.num_lines {
            self.resize_lines(w, h);
        }

        let dt = if self.last_time > 0.0 {
            ((time - self.last_time) / 1000.0) * self.speed
        } else {
            0.0
        };
        self.last_time = time;

        let cap = match self.cap {
            1 => Cap::Square,
            2 => Cap::Round,
            _ => Cap::Butt,
        };

        scene.set_transform(view);
        scene.set_stroke(Stroke::new(self.stroke_width).with_caps(cap));

        let mut path = BezPath::new();
        for line in &mut self.lines {
            line.x0 += line.vx0 * dt;
            line.y0 += line.vy0 * dt;
            line.x1 += line.vx1 * dt;
            line.y1 += line.vy1 * dt;

            bounce(&mut line.x0, &mut line.vx0, w);
            bounce(&mut line.y0, &mut line.vy0, h);
            bounce(&mut line.x1, &mut line.vx1, w);
            bounce(&mut line.y1, &mut line.vy1, h);

            path.move_to((line.x0, line.y0));
            path.line_to((line.x1, line.y1));
            scene.set_paint(line.color);
            scene.stroke_path(&path);
            path.truncate(0);
        }

        scene.reset_transform();
    }
}
