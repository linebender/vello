// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(clippy::cast_possible_truncation, reason = "we're doing it on purpose")]

use std::io::BufWriter;

use vello_api::peniko::color::palette;
use vello_api::peniko::kurbo::{BezPath, Point, Stroke, Vec2};
use vello_api::RenderCtx;
use vello_hybrid::{CsRenderCtx, Pixmap};

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

pub fn main() {
    let mut ctx = CsRenderCtx::new(WIDTH, HEIGHT);
    draw_simple_scene(&mut ctx);
    if let Some(filename) = std::env::args().nth(1) {
        let mut pixmap = Pixmap::new(WIDTH, HEIGHT);
        ctx.render_to_pixmap(&mut pixmap);
        pixmap.unpremultiply();
        let file = std::fs::File::create(filename).unwrap();
        let w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(pixmap.data()).unwrap();
    } else {
        ctx.debug_dump();
    }
}

fn star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    path.move_to(center + Vec2::new(outer, 0.));
    for i in 1..n * 2 {
        let th = i as f64 * std::f64::consts::PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}

fn draw_simple_scene(ctx: &mut CsRenderCtx) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 180.0));
    path.close_path();
    // Note: we plan to change the API to have `into`.
    let piet_path = path.into();
    let stroke = Stroke::new(5.0);
    ctx.stroke(&piet_path, &stroke, palette::css::DARK_BLUE.into());
    let star_path = star(Point::new(100., 100.), 13, 50., 95.);
    ctx.clip(&star_path.into());
    ctx.fill(&piet_path, palette::css::REBECCA_PURPLE.into());
}
