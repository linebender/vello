use crate::util::{
    check_ref, get_ctx, stops_blue_green_red_yellow, stops_green_blue, stops_green_blue_with_alpha,
};
use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape};
use vello_cpu::paint::{LinearGradient, RadialGradient};

mod util;

macro_rules! simple {
    ($stops:expr, $name:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = RadialGradient {
            c1: Point::new(50.0, 50.0),
            r1: 10.0,
            c2: Point::new(50.0, 50.0),
            r2: 40.0,
            stops: $stops,
            transform: Affine::IDENTITY,
            extend: vello_common::peniko::Extend::Pad,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_radial_2_stops() {
    simple!(stops_green_blue(), "gradient_radial_2_stops");
}

#[test]
fn gradient_radial_4_stops() {
    simple!(stops_blue_green_red_yellow(), "gradient_radial_4_stops");
}

#[test]
fn gradient_radial_2_stops_with_alpha() {
    simple!(
        stops_green_blue_with_alpha(),
        "gradient_radial_2_stops_with_alpha"
    );
}

macro_rules! gradient_pad {
    ($extend:path, $name:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = RadialGradient {
            c1: Point::new(50.0, 50.0),
            r1: 20.0,
            c2: Point::new(50.0, 50.0),
            r2: 25.0,
            stops: stops_blue_green_red_yellow(),
            transform: Affine::IDENTITY,
            extend: $extend,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_radial_spread_method_pad() {
    gradient_pad!(
        vello_common::peniko::Extend::Pad,
        "gradient_radial_spread_method_pad"
    );
}

#[test]
fn gradient_radial_spread_method_reflect() {
    gradient_pad!(
        vello_common::peniko::Extend::Reflect,
        "gradient_radial_spread_method_reflect"
    );
}

#[test]
fn gradient_radial_spread_method_repeat() {
    gradient_pad!(
        vello_common::peniko::Extend::Repeat,
        "gradient_radial_spread_method_repeat"
    );
}

#[test]
fn gradient_radial_center_offset() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = RadialGradient {
        c1: Point::new(30.0, 30.0),
        r1: 2.0,
        c2: Point::new(50.0, 50.0),
        r2: 40.0,
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Repeat,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_radial_center_offset");
}
