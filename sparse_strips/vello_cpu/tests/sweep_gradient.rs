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
    ($stops:expr, $name:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = SweepGradient {
            center: Point::new(50.0, 50.0),
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
    basic!(stops_green_blue(), "gradient_sweep_2_stops");
}

#[test]
fn gradient_sweep_2_stops_with_alpha() {
    basic!(
        stops_green_blue_with_alpha(),
        "gradient_sweep_2_stops_with_alpha"
    );
}

#[test]
fn gradient_sweep_4_stops() {
    basic!(stops_blue_green_red_yellow(), "gradient_sweep_4_stops");
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

#[test]
fn gradient_sweep_spread_method_pad() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = SweepGradient {
        center: Point::new(50.0, 50.0),
        start_angle: 150.0,
        end_angle: 210.0,
        stops: stops_green_blue(),
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_sweep_spread_method_pad");
}
