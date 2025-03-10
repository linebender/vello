// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple rendering example for the sparse strips CPU renderer
//!
//! This example demonstrates drawing basic shapes using the sparse strips CPU renderer.
//! It creates a simple scene with various shapes and exports it as a PNG.

use std::io::BufWriter;

use kurbo::Affine;
use peniko::color::palette;
use peniko::kurbo::{BezPath, Point, Stroke, Vec2};
use vello_common::pixmap::Pixmap;
use vello_hybrid::RenderContext;

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

pub fn main() {
    let mut args = std::env::args().skip(1);
    let output_filename: String = args.next().expect("output filename is first arg");
    let mut ctx = RenderContext::new(WIDTH as u16, HEIGHT as u16);
    draw_simple_scene(&mut ctx);
    let mut pixmap = Pixmap::new(WIDTH as u16, HEIGHT as u16);
    ctx.render_to_pixmap(&mut pixmap);
    pixmap.unpremultiply();
    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixmap.data()).unwrap();
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

fn draw_simple_scene(ctx: &mut RenderContext) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 180.0));
    path.close_path();
    let piet_path = path.into();
    let stroke = Stroke::new(5.0);
    ctx.set_transform(Affine::scale(5.0));
    ctx.set_stroke(stroke);
    ctx.set_paint(palette::css::DARK_BLUE.into());
    ctx.stroke_path(&piet_path);
    ctx.set_paint(palette::css::REBECCA_PURPLE.into());
    ctx.fill_path(&piet_path);
}
