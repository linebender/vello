mod util;

use crate::util::{check_ref, get_ctx, star_path};
use std::f64::consts::PI;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_cpu::paint::{LinearGradient, Stop, SweepGradient};

#[test]
fn gradient_on_3_wide_tiles() {
    let mut ctx = get_ctx(600, 32, false);
    let rect = Rect::new(4.0, 4.0, 596.0, 28.0);

    let gradient = LinearGradient {
        p0: Point::new(0.0, 0.0),
        p1: Point::new(600.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_on_3_wide_tiles");
}

#[test]
fn gradient_linear_2_stops() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(10.0, 0.0),
        p1: Point::new(90.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_2_stops");
}

#[test]
fn gradient_linear_2_stops_with_alpha() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(10.0, 0.0),
        p1: Point::new(90.0, 0.0),
        stops: stops_green_blue_with_alpha(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_2_stops_with_alpha");
}

#[test]
fn gradient_linear_negative_direction() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(90.0, 0.0),
        p1: Point::new(10.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_negative_direction");
}

#[test]
fn gradient_linear_with_downward_y() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(20.0, 20.0),
        p1: Point::new(80.0, 80.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_with_downward_y");
}

#[test]
fn gradient_linear_with_upward_y() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(20.0, 80.0),
        p1: Point::new(80.0, 20.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_with_upward_y");
}

#[test]
fn gradient_spread_method_pad() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(35.0, 0.0),
        p1: Point::new(65.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_spread_method_pad");
}

#[test]
fn gradient_spread_method_repeat() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(45.0, 0.0),
        p1: Point::new(55.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Repeat,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_spread_method_repeat");
}

#[test]
fn gradient_spread_method_reflect() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(45.0, 0.0),
        p1: Point::new(55.0, 0.0),
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Reflect,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_spread_method_reflect");
}

#[test]
fn gradient_linear_4_stops() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(10.0, 0.0),
        p1: Point::new(90.0, 0.0),
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_4_stops");
}

#[test]
fn gradient_complex_shape() {
    let mut ctx = get_ctx(100, 100, false);
    let path = star_path();

    let gradient = LinearGradient {
        p0: Point::new(0.0, 0.0),
        p1: Point::new(100.0, 0.0),
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_path(&path);

    check_ref(&ctx, "gradient_complex_shape");
}

#[test]
fn gradient_linear_with_y_repeat() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(47.5, 47.5),
        p1: Point::new(50.5, 52.5),
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Repeat,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_with_y_repeat");
}

#[test]
fn gradient_linear_with_y_reflect() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = LinearGradient {
        p0: Point::new(47.5, 47.5),
        p1: Point::new(50.5, 52.5),
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Reflect,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_linear_with_y_reflect");
}

macro_rules! gradient_with_path_transform {
    ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new($p0, $p1, $p2, $p3);

        let gradient = LinearGradient {
            p0: Point::new($p0, $p1),
            p1: Point::new($p2, $p3),
            stops: stops_green_blue(),
            transform: Affine::IDENTITY,
            extend: vello_common::peniko::Extend::Pad,
        };

        ctx.set_transform($transform);
        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_linear_with_path_transform_1() {
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_1",
        Affine::translate((25.0, 25.0)),
        0.0,
        0.0,
        50.0,
        50.0
    );
}

#[test]
fn gradient_linear_with_path_transform_2() {
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_2",
        Affine::scale(2.0),
        12.5,
        12.5,
        37.5,
        37.5
    );
}

#[test]
fn gradient_linear_with_path_transform_3() {
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_3",
        Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
        0.0,
        0.0,
        25.0,
        25.0
    );
}

#[test]
fn gradient_linear_with_path_transform_4() {
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_4",
        Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_linear_with_path_transform_5() {
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_5",
        Affine::scale_non_uniform(1.0, 2.0),
        25.0,
        12.5,
        75.0,
        37.5
    );
}

// Not working yet
#[test]
fn gradient_linear_with_path_transform_6() {
    let transform = Affine::translate((-37.5, 0.0)) * Affine::skew(PI / 4.0, 0.0);
    gradient_with_path_transform!(
        "gradient_linear_with_path_transform_6",
        transform,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

// TODO: Add test case for skewing transforms

#[test]
fn gradient_sweep_2_stops() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = SweepGradient {
        center: Point::new(50.0, 50.0),
        start_angle: 0.0,
        end_angle: 360.0,
        stops: stops_green_blue(),
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_sweep_2_stops");
}

fn stops_green_blue() -> Vec<Stop> {
    vec![
        Stop {
            offset: 0.0,
            color: GREEN,
        },
        Stop {
            offset: 1.0,
            color: BLUE,
        },
    ]
}

fn stops_green_blue_with_alpha() -> Vec<Stop> {
    vec![
        Stop {
            offset: 0.0,
            color: GREEN.with_alpha(0.25),
        },
        Stop {
            offset: 1.0,
            color: BLUE.with_alpha(0.75),
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
