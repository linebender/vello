// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene with many elements each wrapped in a random filter effect.
//! Press +/= to add 10 elements, - to remove 10.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Circle, Rect, RoundedRect, Shape, Stroke};
use vello_common::peniko::Color;

const BATCH_SIZE: usize = 10;

/// The different shape kinds we can draw.
#[derive(Clone, Copy)]
enum ShapeKind {
    /// Solid-color filled rectangle.
    FilledRect,
    /// Rounded rectangle.
    RoundedRect,
    /// A star-shaped path.
    Star,
    /// A circle with a stroked outline.
    StrokedCircle,
    /// A filled circle.
    FilledCircle,
}

const SHAPE_KINDS: [ShapeKind; 5] = [
    ShapeKind::FilledRect,
    ShapeKind::RoundedRect,
    ShapeKind::Star,
    ShapeKind::StrokedCircle,
    ShapeKind::FilledCircle,
];

/// The filter applied to an element.
#[derive(Clone, Copy)]
enum FilterKind {
    /// No filter.
    None,
    /// Gaussian blur.
    Blur { std_deviation: f32 },
    /// Drop shadow.
    DropShadow { std_deviation: f32 },
}

const COLORS: [Color; 8] = [
    css::TOMATO,
    css::ROYAL_BLUE,
    css::SEA_GREEN,
    css::GOLD,
    css::VIOLET,
    css::CORAL,
    css::DEEP_SKY_BLUE,
    css::MEDIUM_SPRING_GREEN,
];

struct Element {
    shape: ShapeKind,
    filter: FilterKind,
    /// Normalised position (0..1).
    nx: f64,
    ny: f64,
    /// Size in pixels (before viewport scaling).
    size: f64,
    /// Rotation in radians.
    rotation: f64,
    /// Index into COLORS.
    color_idx: usize,
}

/// Scene with many elements each wrapped in a random filter. Press +/= to add 10, - to remove 10.
pub struct FilterElementsScene {
    rng: Rng,
    elements: Vec<Element>,
}

impl std::fmt::Debug for FilterElementsScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterElementsScene")
            .field("count", &self.elements.len())
            .finish_non_exhaustive()
    }
}

impl FilterElementsScene {
    /// Create a new `FilterElementsScene`.
    pub fn new() -> Self {
        let mut scene = Self {
            rng: Rng::new(0xCAFE_BABE),
            elements: Vec::new(),
        };
        scene.add_batch();
        scene
    }

    fn random_element(&mut self) -> Element {
        let shape_idx = usize::try_from(self.rng.next_u32()).unwrap() % SHAPE_KINDS.len();
        let filter_idx = usize::try_from(self.rng.next_u32()).unwrap() % 3;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "std_deviation range is small and positive"
        )]
        let filter = match filter_idx {
            0 => FilterKind::None,
            1 => FilterKind::Blur {
                std_deviation: self.rng.range_f64(4.0, 200.0) as f32,
            },
            _ => FilterKind::DropShadow {
                std_deviation: self.rng.range_f64(4.0, 200.0) as f32,
            },
        };
        Element {
            shape: SHAPE_KINDS[shape_idx],
            filter,
            nx: self.rng.range_f64(0.0, 1.0),
            ny: self.rng.range_f64(0.0, 1.0),
            size: self.rng.range_f64(80.0, 250.0),
            rotation: self.rng.range_f64(0.0, std::f64::consts::TAU),
            color_idx: usize::try_from(self.rng.next_u32()).unwrap() % COLORS.len(),
        }
    }

    fn add_batch(&mut self) {
        for _ in 0..BATCH_SIZE {
            let el = self.random_element();
            self.elements.push(el);
        }
    }
}

impl Default for FilterElementsScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleScene for FilterElementsScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        let vw = ctx.width() as f64;
        let vh = ctx.height() as f64;

        for el in &self.elements {
            let x = el.nx * vw;
            let y = el.ny * vh;
            let s = el.size;
            let color = COLORS[el.color_idx];
            let transform = root_transform
                * Affine::translate((x, y))
                * Affine::rotate(el.rotation)
                * Affine::translate((-s / 2.0, -s / 2.0));

            let has_filter = !matches!(el.filter, FilterKind::None);
            if has_filter {
                ctx.set_transform(root_transform);
                let filter = match el.filter {
                    FilterKind::None => unreachable!(),
                    FilterKind::Blur { std_deviation } => {
                        Filter::from_primitive(FilterPrimitive::GaussianBlur {
                            std_deviation,
                            edge_mode: EdgeMode::None,
                        })
                    }
                    FilterKind::DropShadow { std_deviation } => {
                        Filter::from_primitive(FilterPrimitive::DropShadow {
                            dx: 12.0,
                            dy: 12.0,
                            std_deviation,
                            color: vello_common::color::AlphaColor::from_rgba8(255, 255, 255, 220),
                            edge_mode: EdgeMode::None,
                        })
                    }
                };
                ctx.push_filter_layer(filter);
            }

            // Draw the shape.
            ctx.set_transform(transform);
            ctx.set_paint(color);
            match el.shape {
                ShapeKind::FilledRect => {
                    ctx.fill_rect(&Rect::new(0.0, 0.0, s, s));
                }
                ShapeKind::RoundedRect => {
                    let r = s * 0.2;
                    let path = RoundedRect::new(0.0, 0.0, s, s, r).to_path(0.1);
                    ctx.fill_path(&path);
                }
                ShapeKind::Star => {
                    let path = star_path(s / 2.0, s / 2.0, s * 0.45, s * 0.2, 5);
                    ctx.fill_path(&path);
                }
                ShapeKind::StrokedCircle => {
                    let circle = Circle::new((s / 2.0, s / 2.0), s * 0.4).to_path(0.1);
                    ctx.set_stroke(Stroke::new(s * 0.08));
                    ctx.stroke_path(&circle);
                }
                ShapeKind::FilledCircle => {
                    let circle = Circle::new((s / 2.0, s / 2.0), s * 0.4).to_path(0.1);
                    ctx.fill_path(&circle);
                }
            }

            if has_filter {
                ctx.pop_layer();
            }
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "+" | "=" => {
                self.add_batch();
                true
            }
            "-" => {
                let new_len = self.elements.len().saturating_sub(BATCH_SIZE);
                self.elements.truncate(new_len);
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!("{} elements", self.elements.len()))
    }
}

fn star_path(cx: f64, cy: f64, outer_r: f64, inner_r: f64, points: usize) -> BezPath {
    let mut path = BezPath::new();
    let step = std::f64::consts::TAU / (points as f64 * 2.0);
    for i in 0..(points * 2) {
        let r = if i % 2 == 0 { outer_r } else { inner_r };
        let angle = step * i as f64 - std::f64::consts::FRAC_PI_2;
        let px = cx + r * angle.cos();
        let py = cy + r * angle.sin();
        if i == 0 {
            path.move_to((px, py));
        } else {
            path.line_to((px, py));
        }
    }
    path.close_path();
    path
}

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u32() >> 8) as f64 / ((1_u32 << 24) as f64);
        lo + t * (hi - lo)
    }
}
