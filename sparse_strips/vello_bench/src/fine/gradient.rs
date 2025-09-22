// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use crate::fine::{default_blend, fill_single};
use criterion::{Bencher, Criterion};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use smallvec::{SmallVec, smallvec};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::{BLUE, GREEN, RED, YELLOW};
use vello_common::color::{AlphaColor, DynamicColor, Srgb};
use vello_common::encode::EncodeExt;
use vello_common::fearless_simd::Simd;
use vello_common::kurbo::{Affine, Point};
use vello_common::peniko;
use vello_common::peniko::{ColorStop, ColorStops, Gradient, GradientKind};
use vello_cpu::fine::{Fine, FineKernel};
use vello_cpu::peniko::LinearGradientPosition;
use vello_dev_macros::vello_bench;

pub fn gradient(c: &mut Criterion) {
    linear::opaque(c);
    radial::opaque(c);
    radial::opaque_conical(c);
    sweep::opaque(c);

    extend::pad(c);
    extend::repeat(c);
    extend::reflect(c);

    many_stops(c);
    transparent(c);
}

#[vello_bench]
fn many_stops<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let kind = LinearGradientPosition {
        start: Point::new(128.0, 128.0),
        end: Point::new(134.0, 134.0),
    }
    .into();

    gradient_base(b, fine, peniko::Extend::Repeat, kind, get_many_stops());
}

#[vello_bench]
fn transparent<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let kind = LinearGradientPosition {
        start: Point::new(128.0, 128.0),
        end: Point::new(134.0, 134.0),
    }
    .into();

    gradient_base(
        b,
        fine,
        peniko::Extend::Pad,
        kind,
        stops_blue_green_red_yellow_transparent(),
    );
}

mod extend {
    use crate::fine::gradient::{gradient_base, stops_blue_green_red_yellow_opaque};
    use criterion::Bencher;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Point;
    use vello_common::peniko;
    use vello_cpu::{
        fine::{Fine, FineKernel},
        peniko::LinearGradientPosition,
    };
    use vello_dev_macros::vello_bench;

    fn extend<S: Simd, N: FineKernel<S>>(
        b: &mut Bencher<'_>,
        fine: &mut Fine<S, N>,
        extend: peniko::Extend,
    ) {
        let kind = LinearGradientPosition {
            start: Point::new(128.0, 128.0),
            end: Point::new(134.0, 134.0),
        }
        .into();

        gradient_base(b, fine, extend, kind, stops_blue_green_red_yellow_opaque());
    }

    #[vello_bench]
    pub(super) fn pad<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        extend(b, fine, peniko::Extend::Pad);
    }

    #[vello_bench]
    pub(super) fn reflect<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        extend(b, fine, peniko::Extend::Reflect);
    }

    #[vello_bench]
    pub(super) fn repeat<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        extend(b, fine, peniko::Extend::Repeat);
    }
}

mod linear {
    use crate::fine::gradient::{gradient_base, stops_blue_green_red_yellow_opaque};
    use criterion::Bencher;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Point;
    use vello_common::peniko;

    use vello_cpu::{
        fine::{Fine, FineKernel},
        peniko::LinearGradientPosition,
    };
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub(super) fn opaque<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        let kind = LinearGradientPosition {
            start: Point::new(128.0, 128.0),
            end: Point::new(134.0, 134.0),
        }
        .into();

        gradient_base(
            b,
            fine,
            peniko::Extend::Pad,
            kind,
            stops_blue_green_red_yellow_opaque(),
        );
    }
}

mod radial {

    use crate::fine::gradient::{gradient_base, stops_blue_green_red_yellow_opaque};
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Point;
    use vello_common::peniko;
    use vello_common::tile::Tile;
    use vello_cpu::{
        fine::{Fine, FineKernel},
        peniko::RadialGradientPosition,
    };
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub(super) fn opaque<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        let kind = RadialGradientPosition {
            start_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
            start_radius: 25.0,
            end_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
            end_radius: 75.0,
        }
        .into();

        gradient_base(
            b,
            fine,
            peniko::Extend::Pad,
            kind,
            stops_blue_green_red_yellow_opaque(),
        );
    }

    #[vello_bench]
    pub(super) fn opaque_conical<S: Simd, N: FineKernel<S>>(
        b: &mut Bencher<'_>,
        fine: &mut Fine<S, N>,
    ) {
        let kind = RadialGradientPosition {
            start_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
            start_radius: 25.0,
            end_center: Point::new(
                WideTile::WIDTH as f64 / 2.0 + 5.0,
                (Tile::HEIGHT / 2) as f64 + 5.0,
            ),
            end_radius: 75.0,
        }
        .into();

        gradient_base(
            b,
            fine,
            peniko::Extend::Pad,
            kind,
            stops_blue_green_red_yellow_opaque(),
        );
    }
}

mod sweep {

    use crate::fine::gradient::{gradient_base, stops_blue_green_red_yellow_opaque};
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Point;
    use vello_common::peniko;
    use vello_common::tile::Tile;
    use vello_cpu::{
        fine::{Fine, FineKernel},
        peniko::SweepGradientPosition,
    };
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub(super) fn opaque<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
        let kind = SweepGradientPosition {
            center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
            start_angle: 70.0_f32.to_radians(),
            end_angle: 250.0_f32.to_radians(),
        }
        .into();

        gradient_base(
            b,
            fine,
            peniko::Extend::Pad,
            kind,
            stops_blue_green_red_yellow_opaque(),
        );
    }
}

fn gradient_base<S: Simd, N: FineKernel<S>>(
    b: &mut Bencher<'_>,
    fine: &mut Fine<S, N>,
    extend: peniko::Extend,
    kind: GradientKind,
    stops: ColorStops,
) {
    let mut paints = vec![];

    let grad = Gradient {
        kind,
        stops,
        extend,
        ..Default::default()
    };

    let paint = grad.encode_into(&mut paints, Affine::IDENTITY);
    fill_single(
        &paint,
        &paints,
        WideTile::WIDTH as usize,
        b,
        default_blend(),
        fine,
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

fn stops_blue_green_red_yellow_transparent() -> ColorStops {
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

fn get_many_stops() -> ColorStops {
    let mut vec = SmallVec::new();
    let mut rng = StdRng::from_seed(SEED);
    let max = 120;

    for i in 0..=120 {
        let offset = i as f32 / max as f32;
        let color = DynamicColor::from_alpha_color(AlphaColor::<Srgb>::new([
            rng.random::<f32>(),
            rng.random::<f32>(),
            rng.random::<f32>(),
            rng.random::<f32>(),
        ]));

        vec.push(ColorStop { offset, color });
    }

    ColorStops(vec)
}
