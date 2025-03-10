// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless rendering to file

mod common;

use common::pico_svg::PicoSvg;
use common::render_svg;
use std::io::BufWriter;
use vello_common::pixmap::Pixmap;
use vello_hybrid::{DimensionConstraints, RenderContext};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let output_filename: String = args.next().expect("output filename is second arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = (parsed.size.width * render_scale) as u32;
    let svg_height = (parsed.size.height * render_scale) as u32;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let mut render_ctx = RenderContext::new(width as u16, height as u16);
    render_svg(&mut render_ctx, render_scale, &parsed.items);
    let mut pixmap = Pixmap::new(width as u16, height as u16);
    render_ctx.render_to_pixmap(&mut pixmap).await;

    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&pixmap.buf).unwrap();
}
