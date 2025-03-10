// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example for headless rendering
//!
//! This example demonstrates rendering an SVG file without a window or display.
//! It takes an input SVG file and renders it to a PNG file using the hybrid CPU/GPU renderer.

mod common;

use common::render_svg;
use std::io::BufWriter;
use vello_common::pico_svg::PicoSvg;
use vello_common::pixmap::Pixmap;
use vello_hybrid::{DimensionConstraints, RenderContext};

/// Main entry point for the headless rendering example.
/// Takes two command line arguments:
/// - Input SVG filename to render
/// - Output PNG filename to save the rendered result
///
/// Renders the SVG using the hybrid CPU/GPU renderer and saves the output as a PNG file.
fn main() {
    pollster::block_on(run());
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "Width and height are expected to fit within u16 range"
)]
async fn run() {
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let output_filename: String = args.next().expect("output filename is second arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = parsed.size.width * render_scale;
    let svg_height = parsed.size.height * render_scale;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let mut render_ctx = RenderContext::new(width as u16, height as u16);
    render_svg(&mut render_ctx, render_scale, &parsed.items);
    let mut pixmap = Pixmap::new(width as u16, height as u16);
    render_ctx.render_to_pixmap(&mut pixmap).await;

    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&pixmap.buf).unwrap();
}
