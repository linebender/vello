use criterion::Criterion;
use smallvec::smallvec;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, YELLOW};
use vello_common::peniko::{ColorStop, ColorStops};

pub fn gradient(c: &mut Criterion) {
    linear::opaque(c);
    radial::opaque(c);
    sweep::opaque(c);

    extend::pad(c);
    extend::repeat(c);
    extend::reflect(c);
}

mod extend {
    use crate::fine::fill_single;
    use crate::fine::gradient::stops_blue_green_red_yellow_opaque;
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::encode::EncodeExt;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::{Gradient, GradientKind};
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    pub fn extend<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>, extend: peniko::Extend) {
        let mut paints = vec![];

        let grad = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(128.0, 128.0),
                end: Point::new(134.0, 134.0),
            },
            stops: stops_blue_green_red_yellow_opaque(),
            extend,
            ..Default::default()
        };

        let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

        fill_single(&paint, &paints, WideTile::WIDTH as usize, b, fine);
    }

    #[vello_bench]
    pub fn pad<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend(b, fine, peniko::Extend::Pad)
    }

    #[vello_bench]
    pub fn reflect<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend(b, fine, peniko::Extend::Reflect)
    }

    #[vello_bench]
    pub fn repeat<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend(b, fine, peniko::Extend::Repeat)
    }
}

mod linear {
    use crate::fine::fill::fill_single;
    use crate::fine::gradient::stops_blue_green_red_yellow_opaque;
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::encode::EncodeExt;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::{Gradient, GradientKind};
    use vello_common::tile::Tile;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub fn opaque<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let mut paints = vec![];

        let grad = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(WideTile::WIDTH as f64, Tile::HEIGHT as f64),
            },
            stops: stops_blue_green_red_yellow_opaque(),
            extend: peniko::Extend::Pad,
            ..Default::default()
        };

        let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

        fill_single(&paint, &paints, WideTile::WIDTH as usize, b, fine);
    }
}

mod radial {
    use crate::fine::fill::fill_single;
    use crate::fine::gradient::stops_blue_green_red_yellow_opaque;
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::encode::EncodeExt;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::{Gradient, GradientKind};
    use vello_common::tile::Tile;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub fn opaque<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let mut paints = vec![];

        let grad = Gradient {
            kind: GradientKind::Radial {
                start_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                start_radius: 25.0,
                end_center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                end_radius: 75.0,
            },
            stops: stops_blue_green_red_yellow_opaque(),
            extend: peniko::Extend::Pad,
            ..Default::default()
        };

        let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

        fill_single(&paint, &paints, WideTile::WIDTH as usize, b, fine);
    }
}

mod sweep {
    use crate::fine::fill::fill_single;
    use crate::fine::gradient::stops_blue_green_red_yellow_opaque;
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::encode::EncodeExt;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::{Gradient, GradientKind};
    use vello_common::tile::Tile;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    pub fn opaque<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let mut paints = vec![];

        let grad = Gradient {
            kind: GradientKind::Sweep {
                center: Point::new(WideTile::WIDTH as f64 / 2.0, (Tile::HEIGHT / 2) as f64),
                start_angle: 70.0,
                end_angle: 250.0,
            },
            stops: stops_blue_green_red_yellow_opaque(),
            extend: peniko::Extend::Pad,
            ..Default::default()
        };

        let paint = grad.encode_into(&mut paints, Affine::IDENTITY);

        fill_single(&paint, &paints, WideTile::WIDTH as usize, b, fine);
    }
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
