// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::cast_possible_truncation, reason = "we're doing it on purpose")]

//! Offline SVG renderer

mod common;

use common::pico_svg::PicoSvg;
use common::render_svg;
use std::io::BufWriter;
use vello_hybrid::{DimensionConstraints, RenderContext, RenderTarget, Renderer};

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

    let render_data = render_ctx.prepare_render_data();

    let renderer = Renderer::new(RenderTarget::Headless { width, height }, &render_data).await;
    renderer.prepare(&render_data);
    let buffer = renderer.render_to_texture(&render_data, width, height);

    // Convert buffer from BGRA to RGBA format
    let mut rgba_buffer = Vec::with_capacity(buffer.len());
    for chunk in buffer.chunks_exact(4) {
        // BGRA to RGBA conversion (swapping B and R channels)
        rgba_buffer.push(chunk[2]); // R (was B)
        rgba_buffer.push(chunk[1]); // G (unchanged)
        rgba_buffer.push(chunk[0]); // B (was R)
        rgba_buffer.push(chunk[3]); // A (unchanged)
    }

    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&rgba_buffer).unwrap();
}
