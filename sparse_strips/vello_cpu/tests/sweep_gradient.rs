use crate::util::{
    check_ref, get_ctx, star_path, stops_blue_green_red_yellow, stops_green_blue,
    stops_green_blue_with_alpha,
};
use std::f64::consts::PI;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, RED, WHITE, YELLOW};
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
