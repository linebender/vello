// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use criterion::{Bencher, Criterion};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use smallvec::smallvec;
use vello_common::coarse::WideTile;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, ROYAL_BLUE, YELLOW};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::kurbo::{Affine, Point};
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko;
use vello_common::peniko::{
    BlendMode, ColorStop, ColorStops, Compose, Gradient, GradientKind, Mix,
};
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineType, SCRATCH_BUF_SIZE};
use vello_dev_macros::vello_bench;

pub fn fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine/fill");

    macro_rules! fill_single {
        ($name:ident, $paint:expr, $paints:expr, $width:expr) => {
            g.bench_function(stringify!($name), |b| {
                let mut fine = Fine::<u8>::new(WideTile::WIDTH, Tile::HEIGHT);

                let paint = $paint;
                let paints: &[EncodedPaint] = $paints;

                b.iter(|| {
                    fine.fill(
                        0,
                        $width,
                        paint,
                        BlendMode::new(Mix::Normal, Compose::SrcOver),
                        paints,
                    );

                    std::hint::black_box(&fine);
                })
            });
        };
    }

    fill_single!(
        solid_opaque,
        &Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE)),
        &[],
        WideTile::WIDTH as usize
    );
    fill_single!(
        solid_opaque_short,
        &Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE)),
        &[],
        16
    );
    fill_single!(
        solid_transparent,
        &Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE.with_alpha(0.2))),
        &[],
        WideTile::WIDTH as usize
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
                ..Default::default()
            };

            let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

            fill_single!($name, &paint, &paints, WideTile::WIDTH as usize);
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
                ..Default::default()
            };

            let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

            fill_single!($name, &paint, &paints, WideTile::WIDTH as usize);
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
                ..Default::default()
            };

            let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

            fill_single!($name, &paint, &paints, WideTile::WIDTH as usize);
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

fn stops_blue_green_red_yellow_opaque() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(RED),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW),
        },
    ])
}

fn stops_blue_green_red_yellow() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(GREEN.with_alpha(0.5)),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(RED),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW.with_alpha(0.7)),
        },
    ])
}
