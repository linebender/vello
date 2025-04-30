// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use smallvec::smallvec;
use vello_common::color::{ColorSpaceTag, DynamicColor};
use vello_common::color::palette::css::{BLACK, BLUE, WHITE, YELLOW};
use vello_common::kurbo::{Point, Rect};
use vello_common::paint::Gradient;
use vello_common::peniko::{ColorStop, ColorStops, GradientKind};
use crate::util::{check_ref, get_ctx, stops_blue_green_red_yellow, stops_green_blue};

pub(crate) const fn tan_45() -> f64 {
    1.0
}

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
        ..Default::default()
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_on_3_wide_tiles");
}

#[test]
fn gradient_with_global_alpha() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0),
        },
        stops: stops_blue_green_red_yellow(),
        ..Default::default()
    }.multiply_alpha(0.5);

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "gradient_with_global_alpha");
}

fn gradient_with_color_spaces(name: &str, stops: ColorStops) {
    const COLOR_SPACES: &[ColorSpaceTag] = &[
        ColorSpaceTag::Srgb, 
        ColorSpaceTag::LinearSrgb,
        ColorSpaceTag::Oklab,
    ];
    
    const NUM_COLOR_SPACES: u16 = COLOR_SPACES.len() as u16;

    let mut ctx = get_ctx(200, NUM_COLOR_SPACES * 40 + 10, false);
    
    let mut cur_y = 10.0;
    
    for cs in COLOR_SPACES {
        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(10.0, 0.0),
                end: Point::new(190.0, 0.0),
            },
            stops: stops.clone(),
            interpolation_cs: *cs,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&Rect::new(10.0, cur_y, 190.0, cur_y + 30.0));
        cur_y += 40.0;
    }

    check_ref(&ctx, name);
}

#[test]
fn gradient_with_color_spaces_1() {
    let stops = ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLACK),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(WHITE),
        },
    ]);
    
    gradient_with_color_spaces("gradient_with_color_spaces_1", stops);
}

#[test]
fn gradient_with_color_spaces_2() {
    let stops = ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW),
        },
    ]);

    gradient_with_color_spaces("gradient_with_color_spaces_2", stops);
}

#[test]
fn gradient_with_color_spaces_3() {
    gradient_with_color_spaces("gradient_with_color_spaces_3", stops_blue_green_red_yellow());
}

mod linear {
    use crate::util::{
        check_ref, crossed_line_star, get_ctx, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use std::f64::consts::PI;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::paint::Gradient;
    use vello_common::peniko::GradientKind;
    use crate::gradient::tan_45;

    #[test]
    fn gradient_linear_2_stops() {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            },
            stops: stops_green_blue(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_linear_2_stops");
    }

    #[test]
    fn gradient_linear_2_stops_with_alpha() {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            },
            stops: stops_green_blue_with_alpha(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_linear_2_stops_with_alpha");
    }
    
    macro_rules! directional {
        ($name:expr, $start:expr, $end:expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    
            let gradient = Gradient {
                kind: GradientKind::Linear {
                    start: $start,
                    end: $end,
                },
                stops: stops_green_blue(),
                ..Default::default()
            };
    
            ctx.set_paint(gradient);
            ctx.fill_rect(&rect);
            
             check_ref(&ctx, $name);
        };
    }

    #[test]
    fn gradient_linear_negative_direction() {
        directional!("gradient_linear_negative_direction", Point::new(90.0, 0.0), Point::new(10.0, 0.0));
    }

    #[test]
    fn gradient_linear_with_downward_y() {
        directional!("gradient_linear_with_downward_y", Point::new(20.0, 20.0), Point::new(80.0, 80.0));
    }

    #[test]
    fn gradient_linear_with_upward_y() {
        directional!("gradient_linear_with_upward_y", Point::new(20.0, 80.0), Point::new(80.0, 20.0));
    }

    #[test]
    fn gradient_linear_vertical() {
        directional!("gradient_linear_vertical", Point::new(0.0, 10.0), Point::new(0.0, 90.0));
    }

    macro_rules! gradient_pad {
        ($extend:path, $name:expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

            let gradient = Gradient {
                kind: GradientKind::Linear {
                    start: Point::new(40.0, 40.0),
                    end: Point::new(60.0, 60.0),
                },
                stops: stops_blue_green_red_yellow(),
                extend: $extend,
                ..Default::default()
            };

            ctx.set_paint(gradient);
            ctx.fill_rect(&rect);

            check_ref(&ctx, $name);
        };
    }

    #[test]
    fn gradient_linear_spread_method_pad() {
        gradient_pad!(
            vello_common::peniko::Extend::Pad,
            "gradient_linear_with_pad"
        );
    }

    #[test]
    fn gradient_linear_spread_method_repeat() {
        gradient_pad!(
            vello_common::peniko::Extend::Repeat,
            "gradient_linear_with_repeat"
        );
    }

    #[test]
    fn gradient_linear_spread_method_reflect() {
        gradient_pad!(
            vello_common::peniko::Extend::Reflect,
            "gradient_linear_with_reflect"
        );
    }

    #[test]
    fn gradient_linear_4_stops() {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            },
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_linear_4_stops");
    }

    #[test]
    fn gradient_linear_complex_shape() {
        let mut ctx = get_ctx(100, 100, false);
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(100.0, 0.0),
            },
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_path(&path);

        check_ref(&ctx, "gradient_linear_complex_shape");
    }

    #[test]
    fn gradient_linear_with_y_repeat() {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(47.5, 47.5),
                end: Point::new(50.5, 52.5),
            },
            stops: stops_blue_green_red_yellow(),
            extend: vello_common::peniko::Extend::Repeat,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_linear_with_y_repeat");
    }

    #[test]
    fn gradient_linear_with_y_reflect() {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(47.5, 47.5),
                end: Point::new(50.5, 52.5),
            },
            stops: stops_blue_green_red_yellow(),
            extend: vello_common::peniko::Extend::Reflect,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_linear_with_y_reflect");
    }

    macro_rules! gradient_with_transform {
        ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new($p0, $p1, $p2, $p3);

            let gradient = Gradient {
                kind: GradientKind::Linear {
                    start: Point::new($p0, $p1),
                    end: Point::new($p2, $p3),
                },
                stops: stops_blue_green_red_yellow(),
                ..Default::default()
            };

            ctx.set_transform($transform);
            ctx.set_paint(gradient);
            ctx.fill_rect(&rect);

            check_ref(&ctx, $name);
        };
    }

    #[test]
    fn gradient_linear_with_transform_identity() {
        gradient_with_transform!(
            "gradient_linear_with_transform_identity",
            Affine::IDENTITY,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_translate() {
        gradient_with_transform!(
            "gradient_linear_with_transform_translate",
            Affine::translate((25.0, 25.0)),
            0.0,
            0.0,
            50.0,
            50.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_scale() {
        gradient_with_transform!(
            "gradient_linear_with_transform_scale",
            Affine::scale(2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
    }

    #[test]
    fn gradient_linear_with_transform_negative_scale() {
        gradient_with_transform!(
            "gradient_linear_with_transform_negative_scale",
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
    }

    #[test]
    fn gradient_linear_with_transform_scale_and_translate() {
        gradient_with_transform!(
            "gradient_linear_with_transform_scale_and_translate",
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_rotate_1() {
        gradient_with_transform!(
            "gradient_linear_with_transform_rotate_1",
            Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_rotate_2() {
        gradient_with_transform!(
            "gradient_linear_with_transform_rotate_2",
            Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_scaling_non_uniform() {
        gradient_with_transform!(
            "gradient_linear_with_transform_scaling_non_uniform",
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5
        );
    }

    #[test]
    fn gradient_linear_with_transform_skew_x_1() {
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
        gradient_with_transform!(
            "gradient_linear_with_transform_skew_x_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_skew_x_2() {
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
        gradient_with_transform!(
            "gradient_linear_with_transform_skew_x_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_skew_y_1() {
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
        gradient_with_transform!(
            "gradient_linear_with_transform_skew_y_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_linear_with_transform_skew_y_2() {
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform!(
            "gradient_linear_with_transform_skew_y_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }
}

mod radial {
    use crate::util::{
        check_ref, crossed_line_star, get_ctx, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use std::f64::consts::PI;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::GradientKind::Radial;
    use vello_common::paint::Gradient;
    use crate::gradient::tan_45;

    macro_rules! simple {
        ($stops:expr, $name:expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

            let gradient = Gradient {
                kind: Radial {
                    start_center: Point::new(50.0, 50.0),
                    start_radius: 10.0,
                    end_center: Point::new(50.0, 50.0),
                    end_radius: 40.0,
                },
                stops: $stops,
                ..Default::default()
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

            let gradient = Gradient {
                kind: Radial {
                    start_center: Point::new(50.0, 50.0),
                    start_radius: 20.0,
                    end_center: Point::new(50.0, 50.0),
                    end_radius: 25.0,
                },
                stops: stops_blue_green_red_yellow(),
                extend: $extend,
                ..Default::default()
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

            let gradient = Gradient {
                kind: Radial {
                    start_center: $point,
                    start_radius: 2.0,
                    end_center: Point::new(50.0, 50.0),
                    end_radius: 40.0,
                },
                stops: stops_blue_green_red_yellow(),
                extend: vello_common::peniko::Extend::Repeat,
                ..Default::default()
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

        let gradient = Gradient {
            kind: Radial {
                start_center: Point::new(50.0, 50.0),
                start_radius: 40.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 10.0,
            },
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);

        check_ref(&ctx, "gradient_radial_circle_1_bigger_radius");
    }

    macro_rules! non_overlapping {
        ($radius:expr,$name:expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

            let gradient = Gradient {
                kind: Radial {
                    start_center: Point::new(30.0, 50.0),
                    start_radius: $radius,
                    end_center: Point::new(70.0, 50.0),
                    end_radius: 20.0,
                },
                stops: stops_blue_green_red_yellow(),
                ..Default::default()
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
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: Radial {
                start_center: Point::new(50.0, 50.0),
                start_radius: 5.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 35.0,
            },
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
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

            let gradient = Gradient {
                kind: Radial {
                    start_center: point,
                    start_radius: 5.0,
                    end_center: point,
                    end_radius: 35.0,
                },
                stops: stops_blue_green_red_yellow(),
                ..Default::default()
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
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
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
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
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
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
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
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform!(
            "gradient_radial_with_transform_skew_y_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }
}

mod sweep {
    use crate::util::{
        check_ref, crossed_line_star, get_ctx, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use std::f64::consts::PI;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::GradientKind;
    use vello_common::paint::Gradient;
    use crate::gradient::tan_45;

    macro_rules! basic {
        ($stops:expr, $name:expr, $center:expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

            let gradient = Gradient {
                kind: GradientKind::Sweep {
                    center: $center,
                    start_angle: 0.0,
                    end_angle: 360.0,
                },
                stops: $stops,
                ..Default::default()
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
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: GradientKind::Sweep {
                center: Point::new(50.0, 50.0),
                start_angle: 0.0,
                end_angle: 360.0,
            },
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_path(&path);

        check_ref(&ctx, "gradient_sweep_complex_shape");
    }

    macro_rules! spread_method {
        ($name:expr, $extend:path) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

            let gradient = Gradient {
                kind: GradientKind::Sweep {
                    center: Point::new(50.0, 50.0),
                    start_angle: 150.0,
                    end_angle: 210.0,
                },
                stops: stops_blue_green_red_yellow(),
                extend: $extend,
                ..Default::default()
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

    macro_rules! gradient_with_transform {
        ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
            let mut ctx = get_ctx(100, 100, false);
            let rect = Rect::new($p0, $p1, $p2, $p3);

            let gradient = Gradient {
                kind: GradientKind::Sweep {
                    center: Point::new(($p0 + $p2) / 2.0, ($p1 + $p3) / 2.0),
                    start_angle: 150.0,
                    end_angle: 210.0,
                },
                stops: stops_blue_green_red_yellow(),
                ..Default::default()
            };

            ctx.set_transform($transform);
            ctx.set_paint(gradient);
            ctx.fill_rect(&rect);

            check_ref(&ctx, $name);
        };
    }

    #[test]
    fn gradient_sweep_with_transform_identity() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_identity",
            Affine::IDENTITY,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_translate() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_translate",
            Affine::translate((25.0, 25.0)),
            0.0,
            0.0,
            50.0,
            50.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_scale() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_scale",
            Affine::scale(2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
    }

    #[test]
    fn gradient_sweep_with_transform_negative_scale() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_negative_scale",
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
    }

    #[test]
    fn gradient_sweep_with_transform_scale_and_translate() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_scale_and_translate",
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_rotate_1() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_rotate_1",
            Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_rotate_2() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_rotate_2",
            Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_scale_non_uniform() {
        gradient_with_transform!(
            "gradient_sweep_with_transform_scale_non_uniform",
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5
        );
    }

    #[test]
    fn gradient_sweep_with_transform_skew_x_1() {
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
        gradient_with_transform!(
            "gradient_sweep_with_transform_skew_x_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_skew_x_2() {
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
        gradient_with_transform!(
            "gradient_sweep_with_transform_skew_x_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_skew_y_1() {
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
        gradient_with_transform!(
            "gradient_sweep_with_transform_skew_y_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }

    #[test]
    fn gradient_sweep_with_transform_skew_y_2() {
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform!(
            "gradient_sweep_with_transform_skew_y_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
    }
}
