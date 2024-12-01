// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A benchmark based on MotionMark 1.2's path benchmark.
//! This is roughly comparable to:
//!
//! <https://browserbench.org/MotionMark1.2/developer.html?warmup-length=2000&warmup-frame-count=30&first-frame-minimum-length=0&test-interval=15&display=minimal&tiles=big&controller=adaptive&frame-rate=50&time-measurement=performance&suite-name=MotionMark&test-name=Paths&complexity=1>
//!
//! However, at this point it cannot be directly compared, as we don't accurately
//! implement the stroke style parameters, and it has not been carefully validated.

use std::cmp::Ordering;

use rand::seq::SliceRandom;
use rand::Rng;
use vello::kurbo::{Affine, BezPath, CubicBez, Line, ParamCurve, PathSeg, Point, QuadBez, Stroke};
use vello::peniko::Color;
use vello::Scene;

use crate::{SceneParams, TestScene};

const WIDTH: usize = 1600;
const HEIGHT: usize = 900;

const GRID_WIDTH: i64 = 80;
const GRID_HEIGHT: i64 = 40;

pub struct MMark {
    elements: Vec<Element>,
}

struct Element {
    seg: PathSeg,
    color: Color,
    width: f64,
    is_split: bool,
    grid_point: GridPoint,
}

#[derive(Clone, Copy)]
struct GridPoint(i64, i64);

impl MMark {
    pub fn new(n: usize) -> MMark {
        let mut result = MMark { elements: vec![] };
        result.resize(n);
        result
    }

    fn resize(&mut self, n: usize) {
        let old_n = self.elements.len();
        match n.cmp(&old_n) {
            Ordering::Less => self.elements.truncate(n),
            Ordering::Greater => {
                let mut last = self
                    .elements
                    .last()
                    .map(|e| e.grid_point)
                    .unwrap_or(GridPoint(GRID_WIDTH / 2, GRID_HEIGHT / 2));
                self.elements.extend((old_n..n).map(|_| {
                    let element = Element::new_rand(last);
                    last = element.grid_point;
                    element
                }));
            }
            _ => (),
        }
    }
}

impl TestScene for MMark {
    fn render(&mut self, scene: &mut Scene, params: &mut SceneParams<'_>) {
        let c = params.complexity;
        let n = if c < 10 {
            (c + 1) * 1000
        } else {
            ((c - 8) * 10000).min(120_000)
        };
        self.resize(n);
        let mut rng = rand::thread_rng();
        let mut path = BezPath::new();
        let len = self.elements.len();
        for (i, element) in self.elements.iter_mut().enumerate() {
            if path.is_empty() {
                path.move_to(element.seg.start());
            }
            match element.seg {
                PathSeg::Line(l) => path.line_to(l.p1),
                PathSeg::Quad(q) => path.quad_to(q.p1, q.p2),
                PathSeg::Cubic(c) => path.curve_to(c.p1, c.p2, c.p3),
            }
            if element.is_split || i == len {
                // This gets color and width from the last element, original
                // gets it from the first, but this should not matter.
                scene.stroke(
                    &Stroke::new(element.width),
                    Affine::IDENTITY,
                    element.color,
                    None,
                    &path,
                );
                path.truncate(0); // Should have clear method, to avoid allocations.
            }
            if rng.r#gen::<f32>() > 0.995 {
                element.is_split ^= true;
            }
        }
        let label = format!("mmark test: {} path elements (up/down to adjust)", n);
        params.text.add(
            scene,
            None,
            40.0,
            None,
            Affine::translate((100.0, 1100.0)),
            &label,
        );
    }
}

const COLORS: &[Color] = &[
    Color::from_rgba8(0x10, 0x10, 0x10, 0xff),
    Color::from_rgba8(0x80, 0x80, 0x80, 0xff),
    Color::from_rgba8(0xc0, 0xc0, 0xc0, 0xff),
    Color::from_rgba8(0x10, 0x10, 0x10, 0xff),
    Color::from_rgba8(0x80, 0x80, 0x80, 0xff),
    Color::from_rgba8(0xc0, 0xc0, 0xc0, 0xff),
    Color::from_rgba8(0xe0, 0x10, 0x40, 0xff),
];

impl Element {
    fn new_rand(last: GridPoint) -> Element {
        let mut rng = rand::thread_rng();
        let seg_type = rng.gen_range(0..4);
        let next = GridPoint::random_point(last);
        let (grid_point, seg) = if seg_type < 2 {
            (
                next,
                PathSeg::Line(Line::new(last.coordinate(), next.coordinate())),
            )
        } else if seg_type < 3 {
            let p2 = GridPoint::random_point(next);
            (
                p2,
                PathSeg::Quad(QuadBez::new(
                    last.coordinate(),
                    next.coordinate(),
                    p2.coordinate(),
                )),
            )
        } else {
            let p2 = GridPoint::random_point(next);
            let p3 = GridPoint::random_point(next);
            (
                p3,
                PathSeg::Cubic(CubicBez::new(
                    last.coordinate(),
                    next.coordinate(),
                    p2.coordinate(),
                    p3.coordinate(),
                )),
            )
        };
        let color = *COLORS.choose(&mut rng).unwrap();
        let width = rng.r#gen::<f64>().powi(5) * 20.0 + 1.0;
        let is_split = rng.r#gen();
        Element {
            seg,
            color,
            width,
            is_split,
            grid_point,
        }
    }
}

const OFFSETS: &[(i64, i64)] = &[(-4, 0), (2, 0), (1, -2), (1, 2)];

impl GridPoint {
    fn random_point(last: GridPoint) -> GridPoint {
        let mut rng = rand::thread_rng();

        let offset = OFFSETS.choose(&mut rng).unwrap();
        let mut x = last.0 + offset.0;
        if !(0..=GRID_WIDTH).contains(&x) {
            x -= offset.0 * 2;
        }
        let mut y = last.1 + offset.1;
        if !(0..=GRID_HEIGHT).contains(&y) {
            y -= offset.1 * 2;
        }
        GridPoint(x, y)
    }

    fn coordinate(self) -> Point {
        let scale_x = WIDTH as f64 / ((GRID_WIDTH + 1) as f64);
        let scale_y = HEIGHT as f64 / ((GRID_HEIGHT + 1) as f64);
        Point::new(
            (self.0 as f64 + 0.5) * scale_x,
            100.0 + (self.1 as f64 + 0.5) * scale_y,
        )
    }
}
