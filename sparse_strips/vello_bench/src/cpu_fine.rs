// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::{BLUE, GREEN, RED, ROYAL_BLUE, YELLOW};
use vello_common::kurbo::Point;
use vello_common::paint::{IndexedPaint, Paint};
use vello_common::peniko;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;
use vello_cpu::paint::{EncodedPaint, LinearGradient, Stop, SweepGradient};

pub fn fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine/fill");

    macro_rules! fill_single {
        ($name:ident, $paint:expr, $paints:expr) => {
            g.bench_function(stringify!($name), |b| {
                let mut out = vec![];
                let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                b.iter(|| {
                    fine.fill(0, 0, 0, WideTile::WIDTH as usize, $paint, $paints);

                    std::hint::black_box(&fine);
                })
            });
        };
    }

    fill_single!(
        opaque,
        &Paint::Solid(ROYAL_BLUE.premultiply().to_rgba8()),
        &[]
    );
    fill_single!(
        transparent,
        &Paint::Solid(ROYAL_BLUE.with_alpha(0.2).premultiply().to_rgba8()),
        &[]
    );

    macro_rules! fill_single_linear {
        ($name:ident, $extend:ident, $stops:expr) => {
            let paints = [(LinearGradient {
                p0: Point::new(80.0, 0.0),
                p1: Point::new(120.0, 0.0),
                stops: $stops,
                extend: peniko::Extend::$extend,
            })
            .encode()
            .into()];

            fill_single!($name, &Paint::Indexed(IndexedPaint::new(0)), &paints[..]);
        };
    }

    fill_single_linear!(
        linear_gradient_opaque,
        Pad,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_linear!(linear_gradient_pad, Pad, stops_blue_green_red_yellow());
    fill_single_linear!(
        linear_gradient_repeat,
        Repeat,
        stops_blue_green_red_yellow()
    );
    // Reflect is just a special case of repeat, so no extra benchmarks.

    macro_rules! fill_single_sweep {
        ($name:ident, $extend:ident, $stops:expr) => {
            let paints = [(SweepGradient {
                center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                start_angle: 120.0,
                end_angle: 240.0,
                stops: $stops,
                extend: peniko::Extend::$extend,
            })
            .encode()
            .into()];

            fill_single!($name, &Paint::Indexed(IndexedPaint::new(0)), &paints[..]);
        };
    }
    fill_single_sweep!(sweep_gradient_pad, Pad, stops_blue_green_red_yellow());
}

pub fn strip(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine/strip");
    let mut rng = StdRng::from_seed(SEED);

    let mut alphas = vec![];

    for _ in 0..WideTile::WIDTH * Tile::HEIGHT {
        alphas.push(rng.random());
    }

    macro_rules! strip_single {
        ($name:ident, $paint:expr, $paints:expr) => {
            g.bench_function(stringify!($name), |b| {
                let mut out = vec![];
                let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                b.iter(|| {
                    fine.strip(0, 0, 0, WideTile::WIDTH as usize, &alphas, $paint, $paints);

                    std::hint::black_box(&fine);
                })
            });
        };
    }

    strip_single!(
        basic,
        &Paint::Solid(ROYAL_BLUE.premultiply().to_rgba8()),
        &[]
    );

    macro_rules! strip_single_linear {
        ($name:ident, $extend:ident) => {
            let paints = [(LinearGradient {
                p0: Point::new(0.0, 0.0),
                p1: Point::new(WideTile::WIDTH as f64, 0.0),
                stops: stops_blue_green_red_yellow(),
                extend: peniko::Extend::$extend,
            })
            .encode()
            .into()];

            strip_single!($name, &Paint::Indexed(IndexedPaint::new(0)), &paints[..]);
        };
    }

    strip_single_linear!(linear_gradient_pad, Pad);
    strip_single_linear!(linear_gradient_repeat, Repeat);
    // Reflect is just a special case of repeat, so not extra benchmarks.
}

fn stops_blue_green_red_yellow_opaque() -> Vec<Stop> {
    vec![
        Stop {
            offset: 0.0,
            color: BLUE,
        },
        Stop {
            offset: 0.33,
            color: GREEN,
        },
        Stop {
            offset: 0.66,
            color: RED,
        },
        Stop {
            offset: 1.0,
            color: YELLOW,
        },
    ]
}

fn stops_blue_green_red_yellow() -> Vec<Stop> {
    vec![
        Stop {
            offset: 0.0,
            color: BLUE,
        },
        Stop {
            offset: 0.33,
            color: GREEN.with_alpha(0.5),
        },
        Stop {
            offset: 0.66,
            color: RED,
        },
        Stop {
            offset: 1.0,
            color: YELLOW.with_alpha(0.7),
        },
    ]
}
