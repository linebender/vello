// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use crate::util::{stops_blue_green_red_yellow, stops_green_blue};
use smallvec::smallvec;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, WHITE, YELLOW};
use vello_common::color::{ColorSpaceTag, DynamicColor};
use vello_common::kurbo::{Point, Rect};
use vello_common::peniko::{ColorStop, ColorStops, Gradient};
use vello_cpu::peniko::LinearGradientPosition;
use vello_dev_macros::vello_test;

pub(crate) const fn tan_45() -> f64 {
    1.0
}

#[vello_test(width = 600, height = 32)]
fn gradient_on_3_wide_tiles(ctx: &mut impl Renderer) {
    let rect = Rect::new(4.0, 4.0, 596.0, 28.0);

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(0.0, 0.0),
            end: Point::new(600.0, 0.0),
        }
        .into(),
        stops: stops_green_blue(),
        ..Default::default()
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
}

#[vello_test]
fn gradient_with_global_alpha(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0),
        }
        .into(),
        stops: stops_blue_green_red_yellow(),
        ..Default::default()
    }
    .multiply_alpha(0.5);

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
}

fn gradient_with_color_spaces(ctx: &mut impl Renderer, stops: ColorStops) {
    const COLOR_SPACES: &[ColorSpaceTag] = &[
        ColorSpaceTag::Srgb,
        ColorSpaceTag::LinearSrgb,
        ColorSpaceTag::Oklab,
    ];

    let mut cur_y = 10.0;

    for cs in COLOR_SPACES {
        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(10.0, 0.0),
                end: Point::new(190.0, 0.0),
            }
            .into(),
            stops: stops.clone(),
            interpolation_cs: *cs,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&Rect::new(10.0, cur_y, 190.0, cur_y + 30.0));
        cur_y += 40.0;
    }
}

#[vello_test(width = 200, height = 130)]
fn gradient_with_color_spaces_1(ctx: &mut impl Renderer) {
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

    gradient_with_color_spaces(ctx, stops);
}

#[vello_test(width = 200, height = 130)]
fn gradient_with_color_spaces_2(ctx: &mut impl Renderer) {
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

    gradient_with_color_spaces(ctx, stops);
}

#[vello_test(width = 200, height = 130)]
fn gradient_with_color_spaces_3(ctx: &mut impl Renderer) {
    gradient_with_color_spaces(ctx, stops_blue_green_red_yellow());
}

fn padded_stops(ctx: &mut impl Renderer, offset_1: f32, offset_2: f32) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0),
        }
        .into(),
        stops: ColorStops(smallvec![
            ColorStop {
                offset: offset_1,
                color: DynamicColor::from_alpha_color(GREEN),
            },
            ColorStop {
                offset: offset_2,
                color: DynamicColor::from_alpha_color(BLUE),
            },
        ]),
        ..Default::default()
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
}

#[vello_test]
fn gradient_padded_first_stop(ctx: &mut impl Renderer) {
    padded_stops(ctx, 0.5, 1.0);
}

#[vello_test]
fn gradient_padded_last_stop(ctx: &mut impl Renderer) {
    padded_stops(ctx, 0.0, 0.5);
}

#[vello_test]
fn gradient_padded_stops(ctx: &mut impl Renderer) {
    padded_stops(ctx, 0.25, 0.75);
}

mod linear {
    use crate::gradient::tan_45;
    use crate::renderer::Renderer;
    use crate::util::{
        crossed_line_star, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use peniko::Extend;
    use std::f64::consts::PI;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::{self, Gradient};
    use vello_cpu::peniko::LinearGradientPosition;
    use vello_dev_macros::vello_test;

    #[vello_test]
    fn gradient_linear_2_stops(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            }
            .into(),
            stops: stops_green_blue(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_linear_2_stops_with_alpha(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            }
            .into(),
            stops: stops_green_blue_with_alpha(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    fn directional(ctx: &mut impl Renderer, start: Point, end: Point) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition { start, end }.into(),
            stops: stops_green_blue(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_linear_negative_direction(ctx: &mut impl Renderer) {
        directional(ctx, Point::new(90.0, 0.0), Point::new(10.0, 0.0));
    }

    #[vello_test]
    fn gradient_linear_with_downward_y(ctx: &mut impl Renderer) {
        directional(ctx, Point::new(20.0, 20.0), Point::new(80.0, 80.0));
    }

    #[vello_test]
    fn gradient_linear_with_upward_y(ctx: &mut impl Renderer) {
        directional(ctx, Point::new(20.0, 80.0), Point::new(80.0, 20.0));
    }

    #[vello_test]
    fn gradient_linear_vertical(ctx: &mut impl Renderer) {
        directional(ctx, Point::new(0.0, 10.0), Point::new(0.0, 90.0));
    }

    fn gradient_pad(ctx: &mut impl Renderer, extend: Extend) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(40.0, 40.0),
                end: Point::new(50.0, 60.0),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            extend,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_linear_spread_method_pad(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Pad);
    }

    #[vello_test]
    fn gradient_linear_spread_method_repeat(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Repeat);
    }

    #[vello_test]
    fn gradient_linear_spread_method_reflect(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Reflect);
    }

    #[vello_test]
    fn gradient_linear_4_stops(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(10.0, 0.0),
                end: Point::new(90.0, 0.0),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test(cpu_u8_tolerance = 1)]
    fn gradient_linear_complex_shape(ctx: &mut impl Renderer) {
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(0.0, 0.0),
                end: Point::new(100.0, 0.0),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_path(&path);
    }

    // vello_hybrid:
    // - diff_pixels = 2: It’s likely that the issue comes from accumulated rounding errors.
    // When the gradient’s t-value falls right on the edge of the texture ramp, it may yield
    // a different result than in vello_cpu.
    #[vello_test(diff_pixels = 2)]
    fn gradient_linear_with_y_repeat(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(47.5, 47.5),
                end: Point::new(50.5, 52.5),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            extend: Extend::Repeat,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_linear_with_y_reflect(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(47.5, 47.5),
                end: Point::new(50.5, 52.5),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            extend: Extend::Reflect,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    fn gradient_with_transform(
        ctx: &mut impl Renderer,
        transform: Affine,
        l: f64,
        t: f64,
        r: f64,
        b: f64,
    ) {
        let rect = Rect::new(l, t, r, b);

        let gradient = Gradient {
            kind: LinearGradientPosition {
                start: Point::new(l, t),
                end: Point::new(r, b),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_transform(transform);
        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_linear_with_transform_identity(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::IDENTITY, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_linear_with_transform_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::translate((25.0, 25.0)), 0.0, 0.0, 50.0, 50.0);
    }

    #[vello_test]
    fn gradient_linear_with_transform_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::scale(2.0), 12.5, 12.5, 37.5, 37.5);
    }

    #[vello_test]
    fn gradient_linear_with_transform_negative_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_linear_with_transform_scale_and_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0,
        );
    }

    #[vello_test]
    fn gradient_linear_with_transform_rotate_1(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_linear_with_transform_rotate_2(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_linear_with_transform_scaling_non_uniform(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_linear_with_transform_skew_x_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_linear_with_transform_skew_x_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_linear_with_transform_skew_y_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_linear_with_transform_skew_y_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }
}

mod radial {
    use crate::gradient::tan_45;
    use crate::renderer::Renderer;
    use crate::util::{
        crossed_line_star, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use peniko::Extend;
    use std::f64::consts::PI;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::GradientKind::Radial;
    use vello_common::peniko::{self, ColorStops, Gradient};
    use vello_cpu::peniko::RadialGradientPosition;
    use vello_dev_macros::vello_test;

    fn simple(ctx: &mut impl Renderer, stops: ColorStops) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 10.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 40.0,
            }),
            stops,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_2_stops(ctx: &mut impl Renderer) {
        simple(ctx, stops_green_blue());
    }

    #[vello_test]
    fn gradient_radial_4_stops(ctx: &mut impl Renderer) {
        simple(ctx, stops_blue_green_red_yellow());
    }

    #[vello_test]
    fn gradient_radial_2_stops_with_alpha(ctx: &mut impl Renderer) {
        simple(ctx, stops_green_blue_with_alpha());
    }

    fn gradient_pad(ctx: &mut impl Renderer, extend: Extend) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 20.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 25.0,
            }),
            stops: stops_blue_green_red_yellow(),
            extend,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_spread_method_pad(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Pad);
    }

    #[vello_test]
    fn gradient_radial_spread_method_reflect(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Reflect);
    }

    #[vello_test]
    fn gradient_radial_spread_method_repeat(ctx: &mut impl Renderer) {
        gradient_pad(ctx, Extend::Repeat);
    }

    fn offset(ctx: &mut impl Renderer, point: Point) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: point,
                start_radius: 2.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 40.0,
            }),
            stops: stops_blue_green_red_yellow(),
            extend: Extend::Repeat,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_center_offset_top_left(ctx: &mut impl Renderer) {
        offset(ctx, Point::new(30.0, 30.0));
    }

    #[vello_test]
    fn gradient_radial_center_offset_top_right(ctx: &mut impl Renderer) {
        offset(ctx, Point::new(70.0, 30.0));
    }

    #[vello_test]
    fn gradient_radial_center_offset_bottom_left(ctx: &mut impl Renderer) {
        offset(ctx, Point::new(30.0, 70.0));
    }

    #[vello_test]
    fn gradient_radial_center_offset_bottom_right(ctx: &mut impl Renderer) {
        offset(ctx, Point::new(70.0, 70.0));
    }

    #[vello_test]
    fn gradient_radial_c0_bigger(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 40.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 10.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    fn non_overlapping(ctx: &mut impl Renderer, radius: f32) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(30.0, 50.0),
                start_radius: radius,
                end_center: Point::new(70.0, 50.0),
                end_radius: 20.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_non_overlapping_same_size(ctx: &mut impl Renderer) {
        non_overlapping(ctx, 20.0);
    }

    #[vello_test]
    fn gradient_radial_non_overlapping_c0_smaller(ctx: &mut impl Renderer) {
        non_overlapping(ctx, 15.0);
    }

    #[vello_test]
    fn gradient_radial_non_overlapping_c0_larger(ctx: &mut impl Renderer) {
        non_overlapping(ctx, 25.0);
    }

    #[vello_test]
    fn gradient_radial_non_overlapping_cone(ctx: &mut impl Renderer) {
        non_overlapping(ctx, 5.0);
    }

    #[vello_test]
    fn gradient_radial_complex_shape(ctx: &mut impl Renderer) {
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 5.0,
                end_center: Point::new(50.0, 50.0),
                end_radius: 35.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_path(&path);
    }

    fn gradient_with_transform(
        ctx: &mut impl Renderer,
        transform: Affine,
        l: f64,
        t: f64,
        r: f64,
        b: f64,
    ) {
        let rect = Rect::new(l, t, r, b);
        let point = Point::new((l + r) / 2.0, (t + b) / 2.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: point,
                start_radius: 5.0,
                end_center: point,
                end_radius: 35.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_transform(transform);
        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_with_transform_identity(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::IDENTITY, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_radial_with_transform_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::translate((25.0, 25.0)), 0.0, 0.0, 50.0, 50.0);
    }

    #[vello_test]
    fn gradient_radial_with_transform_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::scale(2.0), 12.5, 12.5, 37.5, 37.5);
    }

    #[vello_test]
    fn gradient_radial_with_transform_negative_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_radial_with_transform_scale_and_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0,
        );
    }

    #[vello_test]
    fn gradient_radial_with_transform_rotate_1(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_radial_with_transform_rotate_2(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_radial_with_transform_scale_non_uniform(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_radial_with_transform_skew_x_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_radial_with_transform_skew_x_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_radial_with_transform_skew_y_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_radial_with_transform_skew_y_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_radial_natively_focal(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 0.0,
                end_center: Point::new(75.0, 75.0),
                end_radius: 40.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_focal_on_circle(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(50.0, 50.0),
                start_radius: 0.0,
                end_center: Point::new(75.0, 50.0),
                end_radius: 25.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_radial_swapped(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(30.0, 50.0),
                start_radius: 40.0,
                end_center: Point::new(60.0, 50.0),
                end_radius: 0.0,
            }),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    // See <https://github.com/linebender/vello/issues/1212>
    #[vello_test]
    fn gradient_radial_smaller_r1_with_reflect(ctx: &mut impl Renderer) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: Radial(RadialGradientPosition {
                start_center: Point::new(30.0, 50.0),
                start_radius: 20.0,
                end_center: Point::new(70.0, 50.0),
                end_radius: 5.0,
            }),
            stops: stops_blue_green_red_yellow(),
            extend: Extend::Reflect,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }
}

mod sweep {
    use crate::gradient::tan_45;
    use crate::renderer::Renderer;
    use crate::util::{
        crossed_line_star, stops_blue_green_red_yellow, stops_green_blue,
        stops_green_blue_with_alpha,
    };
    use peniko::Extend;
    use vello_common::kurbo::{Affine, Point, Rect};
    use vello_common::peniko::{self, ColorStops, Gradient};
    use vello_cpu::peniko::SweepGradientPosition;
    use vello_dev_macros::vello_test;

    fn basic(ctx: &mut impl Renderer, stops: ColorStops, center: Point) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: SweepGradientPosition {
                center,
                start_angle: 0.0_f32.to_radians(),
                end_angle: 360.0_f32.to_radians(),
            }
            .into(),
            stops,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_sweep_2_stops(ctx: &mut impl Renderer) {
        basic(ctx, stops_green_blue(), Point::new(50.0, 50.0));
    }

    #[vello_test]
    fn gradient_sweep_2_stops_with_alpha(ctx: &mut impl Renderer) {
        basic(ctx, stops_green_blue_with_alpha(), Point::new(50.0, 50.0));
    }

    #[vello_test]
    fn gradient_sweep_4_stops(ctx: &mut impl Renderer) {
        basic(ctx, stops_blue_green_red_yellow(), Point::new(50.0, 50.0));
    }

    #[vello_test]
    fn gradient_sweep_not_in_center(ctx: &mut impl Renderer) {
        basic(ctx, stops_green_blue(), Point::new(30.0, 30.0));
    }

    // A single pixel deviates from the reference image by at most 2 units per color channel.
    #[vello_test(diff_pixels = 1)]
    fn gradient_sweep_complex_shape(ctx: &mut impl Renderer) {
        let path = crossed_line_star();

        let gradient = Gradient {
            kind: SweepGradientPosition {
                center: Point::new(50.0, 50.0),
                start_angle: 0.0_f32.to_radians(),
                end_angle: 360.0_f32.to_radians(),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_path(&path);
    }

    fn spread_method(ctx: &mut impl Renderer, extend: Extend) {
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

        let gradient = Gradient {
            kind: SweepGradientPosition {
                center: Point::new(50.0, 50.0),
                start_angle: 150.0_f32.to_radians(),
                end_angle: 210.0_f32.to_radians(),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            extend,
            ..Default::default()
        };

        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_sweep_spread_method_pad(ctx: &mut impl Renderer) {
        spread_method(ctx, Extend::Pad);
    }

    #[vello_test]
    fn gradient_sweep_spread_method_repeat(ctx: &mut impl Renderer) {
        spread_method(ctx, Extend::Repeat);
    }

    #[vello_test]
    fn gradient_sweep_spread_method_reflect(ctx: &mut impl Renderer) {
        spread_method(ctx, Extend::Reflect);
    }

    fn gradient_with_transform(
        ctx: &mut impl Renderer,
        transform: Affine,
        l: f64,
        t: f64,
        r: f64,
        b: f64,
    ) {
        let rect = Rect::new(l, t, r, b);

        let gradient = Gradient {
            kind: SweepGradientPosition {
                center: Point::new((l + r) / 2.0, (t + b) / 2.0),
                start_angle: 150.0_f32.to_radians(),
                end_angle: 210.0_f32.to_radians(),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        };

        ctx.set_transform(transform);
        ctx.set_paint(gradient);
        ctx.fill_rect(&rect);
    }

    #[vello_test]
    fn gradient_sweep_with_transform_identity(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::IDENTITY, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_sweep_with_transform_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::translate((25.0, 25.0)), 0.0, 0.0, 50.0, 50.0);
    }

    #[vello_test]
    fn gradient_sweep_with_transform_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(ctx, Affine::scale(2.0), 12.5, 12.5, 37.5, 37.5);
    }

    #[vello_test]
    fn gradient_sweep_with_transform_negative_scale(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_sweep_with_transform_scale_and_translate(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0,
        );
    }

    #[vello_test]
    fn gradient_sweep_with_transform_rotate_1(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about(50.0_f64.to_radians(), Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_sweep_with_transform_rotate_2(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::rotate_about((-50.0_f64).to_radians(), Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0,
        );
    }

    #[vello_test]
    fn gradient_sweep_with_transform_scale_non_uniform(ctx: &mut impl Renderer) {
        gradient_with_transform(
            ctx,
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5,
        );
    }

    #[vello_test]
    fn gradient_sweep_with_transform_skew_x_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test]
    fn gradient_sweep_with_transform_skew_x_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test(diff_pixels = 4)]
    fn gradient_sweep_with_transform_skew_y_1(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }

    #[vello_test(diff_pixels = 3)]
    fn gradient_sweep_with_transform_skew_y_2(ctx: &mut impl Renderer) {
        let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
        gradient_with_transform(ctx, transform, 25.0, 25.0, 75.0, 75.0);
    }
}
