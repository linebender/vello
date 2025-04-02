use crate::util::{
    check_ref, get_ctx, star_path, stops_blue_green_red_yellow, stops_green_blue,
    stops_green_blue_with_alpha,
};
use std::f64::consts::PI;
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

macro_rules! offset {
    ($point:expr,$name:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = RadialGradient {
            c1: $point,
            r1: 2.0,
            c2: Point::new(50.0, 50.0),
            r2: 40.0,
            stops: stops_blue_green_red_yellow(),
            transform: Affine::IDENTITY,
            extend: vello_common::peniko::Extend::Repeat,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_radial_center_offset_top_left() {
    offset!(
        Point::new(30.0, 30.0),
        "gradient_radial_center_offset_top_left"
    );
}

#[test]
fn gradient_radial_center_offset_top_right() {
    offset!(
        Point::new(70.0, 30.0),
        "gradient_radial_center_offset_top_right"
    );
}

#[test]
fn gradient_radial_center_offset_bottom_left() {
    offset!(
        Point::new(30.0, 70.0),
        "gradient_radial_center_offset_bottom_left"
    );
}

#[test]
fn gradient_radial_center_offset_bottom_right() {
    offset!(
        Point::new(70.0, 70.0),
        "gradient_radial_center_offset_bottom_right"
    );
}

#[test]
fn gradient_radial_c0_bigger() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = RadialGradient {
        c1: Point::new(50.0, 50.0),
        r1: 40.0,
        c2: Point::new(50.0, 50.0),
        r2: 10.0,
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_radial_circle_1_bigger_radius");
}

macro_rules! non_overlapping {
    ($radius:expr,$name:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = RadialGradient {
            c1: Point::new(30.0, 50.0),
            r1: $radius,
            c2: Point::new(70.0, 50.0),
            r2: 20.0,
            stops: stops_blue_green_red_yellow(),
            transform: Affine::IDENTITY,
            extend: vello_common::peniko::Extend::Pad,
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn gradient_radial_non_overlapping_same_size() {
    non_overlapping!(20.0, "gradient_radial_non_overlapping_same_size");
}

#[test]
fn gradient_radial_non_overlapping_c0_smaller() {
    non_overlapping!(15.0, "gradient_radial_non_overlapping_c0_smaller");
}

#[test]
fn gradient_radial_non_overlapping_c0_larger() {
    non_overlapping!(25.0, "gradient_radial_non_overlapping_c0_larger");
}

#[test]
fn gradient_radial_non_overlapping_cone() {
    non_overlapping!(5.0, "gradient_radial_non_overlapping_cone");
}

#[test]
fn gradient_radial_complex_shape() {
    let mut ctx = get_ctx(100, 100, false);
    let path = star_path();

    let gradient = RadialGradient {
        c1: Point::new(50.0, 50.0),
        r1: 5.0,
        c2: Point::new(50.0, 50.0),
        r2: 35.0,
        stops: stops_blue_green_red_yellow(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_path(&path);

    check_ref(&ctx, "gradient_radial_complex_shape");
}

macro_rules! gradient_with_transform {
    ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new($p0, $p1, $p2, $p3);
        let point = Point::new(($p0 + $p2) / 2.0, ($p1 + $p3) / 2.0);

        let gradient = RadialGradient {
            c1: point,
            r1: 5.0,
            c2: point,
            r2: 35.0,
            stops: stops_blue_green_red_yellow(),
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
fn gradient_radial_with_transform_identity() {
    gradient_with_transform!(
        "gradient_radial_with_transform_identity",
        Affine::IDENTITY,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_translate() {
    gradient_with_transform!(
        "gradient_radial_with_transform_translate",
        Affine::translate((25.0, 25.0)),
        0.0,
        0.0,
        50.0,
        50.0
    );
}

#[test]
fn gradient_radial_with_transform_scale() {
    gradient_with_transform!(
        "gradient_radial_with_transform_scale",
        Affine::scale(2.0),
        12.5,
        12.5,
        37.5,
        37.5
    );
}

#[test]
fn gradient_radial_with_transform_negative_scale() {
    gradient_with_transform!(
        "gradient_radial_with_transform_negative_scale",
        Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
        12.5,
        12.5,
        37.5,
        37.5
    );
}

#[test]
fn gradient_radial_with_transform_scale_and_translate() {
    gradient_with_transform!(
        "gradient_radial_with_transform_scale_and_translate",
        Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
        0.0,
        0.0,
        25.0,
        25.0
    );
}

#[test]
fn gradient_radial_with_transform_rotate_1() {
    gradient_with_transform!(
        "gradient_radial_with_transform_rotate_1",
        Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_rotate_2() {
    gradient_with_transform!(
        "gradient_radial_with_transform_rotate_2",
        Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_scale_non_uniform() {
    gradient_with_transform!(
        "gradient_radial_with_transform_scale_non_uniform",
        Affine::scale_non_uniform(1.0, 2.0),
        25.0,
        12.5,
        75.0,
        37.5
    );
}

#[test]
fn gradient_radial_with_transform_skew_x_1() {
    let transform = Affine::translate((-37.5, 0.0)) * Affine::skew(PI / 4.0, 0.0);
    gradient_with_transform!(
        "gradient_radial_with_transform_skew_x_1",
        transform,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_skew_x_2() {
    let transform = Affine::translate((37.5, 0.0)) * Affine::skew(-PI / 4.0, 0.0);
    gradient_with_transform!(
        "gradient_radial_with_transform_skew_x_2",
        transform,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_skew_y_1() {
    let transform = Affine::translate((0.0, 37.5)) * Affine::skew(0.0, -PI / 4.0);
    gradient_with_transform!(
        "gradient_radial_with_transform_skew_y_1",
        transform,
        25.0,
        25.0,
        75.0,
        75.0
    );
}

#[test]
fn gradient_radial_with_transform_skew_y_2() {
    let transform = Affine::translate((0.0, -37.5)) * Affine::skew(0.0, PI / 4.0);
    gradient_with_transform!(
        "gradient_radial_with_transform_skew_y_2",
        transform,
        25.0,
        25.0,
        75.0,
        75.0
    );
}
