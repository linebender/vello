mod util;

use crate::util::{check_ref, get_ctx, star_path, stops_green_blue};
use std::f64::consts::PI;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::peniko::GradientKind;
use vello_cpu::paint::{Gradient, LinearGradient, Stop, SweepGradient};

#[test]
fn gradient_on_3_wide_tiles() {
    let mut ctx = get_ctx(600, 32, false);
    let rect = Rect::new(4.0, 4.0, 596.0, 28.0);

    let gradient = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(600.0, 0.0),
        },
        stops: stops_green_blue(),
        transform: Affine::IDENTITY,
        extend: vello_common::peniko::Extend::Pad,
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_on_3_wide_tiles");
}
