use crate::util::{check_ref, get_ctx, star_path};
use std::f64::consts::PI;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_cpu::paint::{LinearGradient, Stop, SweepGradient};

mod util;

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
