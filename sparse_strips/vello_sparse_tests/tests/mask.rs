// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use smallvec::smallvec;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLACK, LIME, RED, YELLOW};
use vello_common::kurbo::{Point, Rect};
use vello_common::mask::Mask;
use vello_common::paint::Gradient;
use vello_common::peniko::{ColorStop, ColorStops, GradientKind};
use vello_cpu::{Pixmap, RenderContext};
use crate::util::{check_ref, get_ctx};

pub(crate) fn example_mask(alpha_mask: bool) -> Mask {
    let mut mask_pix = Pixmap::new(100, 100);
    let mut mask_ctx = RenderContext::new(100, 100);

    let grad = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0)
        },
        stops: ColorStops(
            smallvec![
                    ColorStop {
                        offset: 0.0 , 
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
                ]
        ),
        ..Default::default()
    };

    mask_ctx.set_paint(grad);
    mask_ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    mask_ctx.render_to_pixmap(&mut mask_pix);

    if alpha_mask {
        Mask::new_alpha(&mask_pix)
    }   else {
        Mask::new_luminance(&mask_pix)
    }
}

macro_rules! mask {
    ($name:expr, $mask:expr) => {
        let mask = example_mask($mask);
        
        let mut ctx = get_ctx(100, 100, false);
        ctx.set_paint(BLACK);
        ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
        ctx.push_mask_layer(mask);
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
        ctx.pop_layer();
    
        check_ref(&ctx, $name);
    };
}

#[test]
fn mask_alpha() {
    mask!("mask_alpha", true);
}

#[test]
fn mask_luminance() {
    mask!("mask_luminance", false);
}
