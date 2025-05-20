use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};
use smallvec::smallvec;
use std::io::Cursor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, YELLOW};
use vello_common::color::{AlphaColor, DynamicColor};
use vello_common::kurbo::{Point, Shape};
use vello_common::peniko::{ColorStop, ColorStops, Gradient};
use vello_common::pixmap::Pixmap;
use vello_cpu::kurbo::Rect;
use vello_cpu::peniko::GradientKind::Radial;
use vello_cpu::{RenderContext, RenderMode, peniko};

fn main() {
    let mut ctx = RenderContext::new(100, 100);
    let rect = &Rect::new(0.0, 0.0, 100.0, 100.0);

    let gradient = Gradient {
        // Concentric
        // kind: Radial {
        //     start_center: Point::new(50.0, 50.0),
        //     start_radius: 10.0,
        //     end_center: Point::new(50.0, 50.0),
        //     end_radius: 40.0,
        // },
        // Strip
        // kind: Radial {
        //     start_center: Point::new(20.0, 50.0),
        //     start_radius: 20.0,
        //     end_center: Point::new(80.0, 50.0),
        //     end_radius: 20.0,
        // },
        kind: Radial {
            start_center: Point::new(30.0, 50.0),
            start_radius: 40.0,
            end_center: Point::new(60.0, 50.0),
            end_radius: 0.0,
        },
        stops: stops_blue_green_red_yellow(),
        extend: peniko::Extend::Pad,
        ..Default::default()
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(rect);

    let mut pix = Pixmap::new(100, 100);

    ctx.render_to_pixmap(&mut pix, RenderMode::OptimizeQuality);

    let img_buf = pix.take_unpremultiplied();

    let mut png_data = Vec::new();
    let cursor = Cursor::new(&mut png_data);
    let encoder = PngEncoder::new(cursor);
    encoder
        .write_image(
            bytemuck::cast_slice(&img_buf),
            100,
            100,
            ExtendedColorType::Rgba8,
        )
        .expect("Failed to encode image");

    std::fs::write("out.png", png_data);
}

fn stops_blue_green_red_yellow() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(RED),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW),
        },
    ])
}

fn stops2() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(AlphaColor::from_rgba8(255, 0, 0, 255)),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(AlphaColor::from_rgba8(255, 255, 0, 255)),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(AlphaColor::from_rgba8(0, 255, 0, 255)),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(AlphaColor::from_rgba8(0, 255, 255, 255)),
        },
    ])
}
