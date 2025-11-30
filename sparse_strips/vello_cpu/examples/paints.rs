// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Using gradients and patterns paints using Vello CPU.

use std::path::Path;
use std::sync::Arc;
use vello_cpu::{
    Image, ImageSource, Pixmap, RenderContext,
    color::palette::css::{BLUE, CYAN, DEEP_PINK, MAGENTA, NAVY, ORANGE, PURPLE, WHITE, YELLOW},
    kurbo::{Affine, Point, Rect},
    peniko::{
        ColorStop, ColorStops, Extend, Gradient, ImageSampler, LinearGradientPosition,
        RadialGradientPosition, SweepGradientPosition,
    },
};

fn main() {
    // Apart from drawing shapes using a single color, you can also draw them
    // using gradients and image patterns.

    let mut ctx = RenderContext::new(500, 500);
    let base_rect = Rect::new(25.0, 25.0, 225.0, 225.0);

    // All you need to is to construct the given gradient/pattern and activate
    // it by using the `set_paint` method. In this case, we are drawing a
    // linear gradient.
    ctx.set_paint(linear_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    // Since we are applying a transform to the render context, the radial
    // gradient we are about to apply will also be affected to that.
    ctx.set_transform(Affine::translate((250.0, 0.0)));
    ctx.set_paint(radial_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    // Sweep gradient are also supported.
    ctx.set_transform(Affine::translate((0.0, 250.0)));
    ctx.set_paint(sweep_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    // Finally, we can use image patterns with different extend modes.
    ctx.set_transform(Affine::translate((250.0, 250.0)));
    ctx.set_paint(pattern());
    // If you want to apply an additional transform that just applies to the
    // paint instead of the shape you are drawing, you can do that using the
    // `set_paint_transform` method.
    ctx.set_paint_transform(Affine::scale(15.0));
    ctx.fill_rect(&base_rect);

    ctx.flush();

    save_pixmap(&ctx, "example_paint");
}

fn linear_gradient(rect: &Rect) -> Gradient {
    Gradient {
        kind: LinearGradientPosition {
            start: Point::new(rect.x0, rect.y0),
            end: Point::new(rect.x1, rect.y1),
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, DEEP_PINK)),
            ColorStop::from((0.55, ORANGE)),
            ColorStop::from((1.0, YELLOW)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn radial_gradient(rect: &Rect) -> Gradient {
    let center = rect.center();
    #[allow(clippy::cast_possible_truncation, reason = "necessary for conversion")]
    let radius = (rect.width().min(rect.height()) * 0.5) as f32;
    Gradient {
        kind: RadialGradientPosition {
            start_center: center,
            start_radius: radius * 0.2,
            end_center: center,
            end_radius: radius,
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, CYAN)),
            ColorStop::from((0.6, BLUE)),
            ColorStop::from((1.0, NAVY)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn sweep_gradient(rect: &Rect) -> Gradient {
    let center = rect.center();
    Gradient {
        kind: SweepGradientPosition {
            center,
            start_angle: 0.0,
            end_angle: std::f32::consts::PI * 2.0,
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, MAGENTA)),
            ColorStop::from((0.5, PURPLE)),
            ColorStop::from((1.0, WHITE)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn pattern() -> Image {
    // For patterns, you need to get access to a pixmap. You can either do this
    // by rendering your own pixmap or converting for example a png image into
    // one.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vello_sparse_tests/tests/assets/rgb_image_2x2.png");
    let file = std::fs::read(path).unwrap();
    let pixmap = Pixmap::from_png(file.as_slice()).unwrap();

    Image {
        // Note that only ImageSource::Pixmap is currently supported. Don't
        // use ImageSource::OpaqueId.
        image: ImageSource::Pixmap(Arc::new(pixmap)),
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            ..Default::default()
        },
    }
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_to_pixmap(&mut pixmap);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}

fn color_stops(stops: [ColorStop; 3]) -> ColorStops {
    ColorStops::from(stops.as_slice())
}
