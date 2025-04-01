use crate::util::{
    check_ref, get_ctx, star_path, stops_blue_green_red_yellow, stops_green_blue,
    stops_green_blue_with_alpha,
};
use std::f64::consts::PI;
use vello_common::kurbo::{Affine, Point, Rect};
use vello_cpu::paint::{LinearGradient, Stop, SweepGradient};

mod util;

macro_rules! basic {
    ($stops:expr, $name:expr, $center:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = SweepGradient {
            center: $center,
            start_angle: 0.0,
            end_angle: 360.0,
            stops: $stops,
            extend: vello_common::peniko::Extend::Pad,
            transform: Affine::IDENTITY,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_sweep_2_stops() {
    basic!(
        stops_green_blue(),
        "gradient_sweep_2_stops",
        Point::new(50.0, 50.0)
    );
}

#[test]
fn gradient_sweep_2_stops_with_alpha() {
    basic!(
        stops_green_blue_with_alpha(),
        "gradient_sweep_2_stops_with_alpha",
        Point::new(50.0, 50.0)
    );
}

#[test]
fn gradient_sweep_4_stops() {
    basic!(
        stops_blue_green_red_yellow(),
        "gradient_sweep_4_stops",
        Point::new(50.0, 50.0)
    );
}

#[test]
fn gradient_sweep_not_in_center() {
    basic!(
        stops_green_blue(),
        "gradient_sweep_not_in_center",
        Point::new(30.0, 30.0)
    );
}

#[test]
fn gradient_sweep_complex_shape() {
    let mut ctx = get_ctx(100, 100, false);
    let path = star_path();

    let gradient = SweepGradient {
        center: Point::new(50.0, 50.0),
        start_angle: 0.0,
        end_angle: 360.0,
        stops: stops_blue_green_red_yellow(),
        extend: vello_common::peniko::Extend::Pad,
        transform: Affine::IDENTITY,
    };

    ctx.set_paint(gradient);
    ctx.fill_path(&path);

    check_ref(&ctx, "gradient_sweep_complex_shape");
}

macro_rules! spread_method {
    ($name:expr, $extend:path) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = SweepGradient {
            center: Point::new(50.0, 50.0),
            start_angle: 150.0,
            end_angle: 210.0,
            stops: stops_blue_green_red_yellow(),
            extend: $extend,
            transform: Affine::IDENTITY,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_sweep_spread_method_pad() {
    spread_method!(
        "gradient_sweep_spread_method_pad",
        vello_common::peniko::Extend::Pad
    );
}

#[test]
fn gradient_sweep_spread_method_repeat() {
    spread_method!(
        "gradient_sweep_spread_method_repeat",
        vello_common::peniko::Extend::Repeat
    );
}

#[test]
fn gradient_sweep_spread_method_reflect() {
    spread_method!(
        "gradient_sweep_spread_method_reflect",
        vello_common::peniko::Extend::Reflect
    );
}

macro_rules! gradient_with_path_transform {
    ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new($p0, $p1, $p2, $p3);

        let gradient = SweepGradient {
            center: Point::new(($p0 + $p2) / 2.0, ($p1 + $p3) / 2.0),
            start_angle: 150.0,
            end_angle: 210.0,
            stops: stops_blue_green_red_yellow(),
            extend: vello_common::peniko::Extend::Pad,
            transform: Affine::IDENTITY,
        };

        ctx.set_transform($transform);
        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_sweep_with_path_transform_identity() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_identity",
        Affine::IDENTITY,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_sweep_with_path_transform_1() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_1",
        Affine::translate((25.0, 25.0)),
        0.0,
        0.0,
        50.0,
        50.0
    );
}

#[test]
fn gradient_sweep_with_path_transform_2() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_2",
        Affine::scale(2.0),
        12.5,
        12.5,
        37.5,
        37.5
    );
}

#[test]
fn gradient_sweep_with_path_transform_3() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_3",
        Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
        0.0,
        0.0,
        25.0,
        25.0
    );
}

#[test]
fn gradient_sweep_with_path_transform_4() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_4",
        Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_sweep_with_path_transform_5() {
    gradient_with_path_transform!(
        "gradient_sweep_with_path_transform_5",
        Affine::scale_non_uniform(1.0, 2.0),
        25.0,
        12.5,
        75.0,
        37.5
    );
}
