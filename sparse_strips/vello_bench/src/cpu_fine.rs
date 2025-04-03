// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::{BLUE, GREEN, RED, ROYAL_BLUE, YELLOW};
use vello_common::kurbo::{Affine, Point};
use vello_common::paint::Paint;
use vello_common::peniko;
use vello_common::peniko::GradientKind;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;
use vello_cpu::paint::{Gradient, Stop};

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
        solid_opaque,
        &Paint::Solid(ROYAL_BLUE.premultiply().to_rgba8()),
        &[]
    );
    fill_single!(
        sold_transparent,
        &Paint::Solid(ROYAL_BLUE.with_alpha(0.2).premultiply().to_rgba8()),
        &[]
    );

    macro_rules! fill_single_linear {
        ($name:ident, $extend:ident, $stops:expr) => {
            let mut paints = vec![];
            let grad = Gradient {
                kind: GradientKind::Linear {
                    start: Point::new(0.0, 0.0),
                    end: Point::new(WideTile::WIDTH as f64, Tile::HEIGHT as f64),
                },
                stops: $stops,
                extend: peniko::Extend::$extend,
                transform: Affine::IDENTITY,
            };

            let paint = grad.encode_into(&mut paints);

            fill_single!($name, &paint, &paints);
        };
    }

    fill_single_linear!(
        linear_gradient_pad,
        Pad,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_linear!(
        linear_gradient_repeat,
        Repeat,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_linear!(
        linear_gradient_reflect,
        Repeat,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_linear!(
        linear_gradient_transparent,
        Pad,
        stops_blue_green_red_yellow()
    );

    macro_rules! fill_single_sweep {
        ($name:ident, $extend:ident, $stops:expr) => {
            let mut paints = vec![];
            let grad = Gradient {
                kind: GradientKind::Sweep {
                    center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                    start_angle: 150.0,
                    end_angle: 210.0,
                },
                stops: $stops,
                extend: peniko::Extend::$extend,
                transform: Affine::default(),
            };

            let paint = grad.encode_into(&mut paints);

            fill_single!($name, &paint, &paints);
        };
    }

    fill_single_sweep!(
        sweep_gradient_pad,
        Pad,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_sweep!(
        sweep_gradient_repeat,
        Repeat,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_sweep!(
        sweep_gradient_reflect,
        Reflect,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_sweep!(
        sweep_gradient_transparent,
        Pad,
        stops_blue_green_red_yellow()
    );

    macro_rules! fill_single_radial {
        ($name:ident, $extend:ident, $stops:expr) => {
            let mut paints = vec![];
            let grad = Gradient {
                kind: GradientKind::Radial {
                    start_center: Point::new(
                        WideTile::WIDTH as f64 / 2.0,
                        (Tile::HEIGHT / 2) as f64,
                    ),
                    start_radius: 25.0,
                    end_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                    end_radius: 75.0,
                },
                stops: $stops,
                extend: peniko::Extend::$extend,
                transform: Affine::default(),
            };

            let paint = grad.encode_into(&mut paints);

            fill_single!($name, &paint, &paints);
        };
    }

    fill_single_radial!(
        radial_gradient_pad,
        Pad,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_radial!(
        radial_gradient_repeat,
        Repeat,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_radial!(
        radial_gradient_reflect,
        Reflect,
        stops_blue_green_red_yellow_opaque()
    );
    fill_single_radial!(
        radial_gradient_transparent,
        Pad,
        stops_blue_green_red_yellow()
    );
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

    // There is not really a need to measure performance of complex paint types
    // for stripping because the code path for generating the gradient data is exactly the same
    // as for filling.
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
