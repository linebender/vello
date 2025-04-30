// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use smallvec::smallvec;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLACK, LIME, RED, YELLOW};
use vello_common::kurbo::{Point, Rect};
use vello_common::mask::Mask;
use vello_common::paint::Gradient;
use vello_common::peniko::{ColorStop, ColorStops, GradientKind};
use vello_cpu::{Pixmap, RenderContext};
use vello_macros::v_test;

pub(crate) fn example_mask(alpha_mask: bool) -> Mask {
    let mut mask_pix = Pixmap::new(100, 100);
    let mut mask_ctx = RenderContext::new(100, 100);

    let grad = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0),
        },
        stops: ColorStops(smallvec![
            ColorStop {
                offset: 0.0,
                color: DynamicColor::from_alpha_color(RED),
            },
            ColorStop {
                offset: 0.5,
                color: DynamicColor::from_alpha_color(YELLOW.with_alpha(0.5)),
            },
            ColorStop {
                offset: 1.0,
                color: DynamicColor::from_alpha_color(LIME.with_alpha(0.0)),
            },
        ]),
        ..Default::default()
    };

    mask_ctx.set_paint(grad);
    mask_ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    mask_ctx.render_to_pixmap(&mut mask_pix);

    if alpha_mask {
        Mask::new_alpha(&mask_pix)
    } else {
        Mask::new_luminance(&mask_pix)
    }
}

fn mask(ctx: &mut impl Renderer, alpha_mask: bool) {
    let mask = example_mask(alpha_mask);

    ctx.set_paint(BLACK);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_mask_layer(mask);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.pop_layer();
}

#[v_test]
fn mask_alpha(ctx: &mut impl Renderer) {
    mask(ctx, true);
}

#[v_test]
fn mask_luminance(ctx: &mut impl Renderer) {
    mask(ctx, false);
}
