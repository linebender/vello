// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG example for sparse strips CPU renderer
//!
//! This example demonstrates loading and rendering an SVG file using the sparse strips CPU renderer.
//! It processes the SVG file and outputs the rendered result to a PNG file.

mod common;

use std::io::BufWriter;

use common::render_svg;
use vello_common::pico_svg::PicoSvg;
use vello_common::pixmap::Pixmap;
use vello_hybrid::RenderContext;

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

/// Main entry point for the SVG CPU renderer example.
///
/// Takes two command line arguments:
/// - Input SVG filename to render
/// - Output PNG filename to save the rendered result
///
/// Renders the SVG using the sparse strips CPU renderer and saves the output as a PNG file.
pub fn main() {
    let mut ctx = RenderContext::new(WIDTH as u16, HEIGHT as u16);
    let mut args = std::env::args().skip(1);
    let svg_filename = args.next().expect("svg filename is first arg");
    let out_filename = args.next().expect("png out filename is second arg");

    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
    let mut pixmap = Pixmap::new(WIDTH as u16, HEIGHT as u16);
    ctx.reset();
    let start = std::time::Instant::now();
    render_svg(&mut ctx, 5.0, &parsed.items);
    let coarse_time = start.elapsed();
    ctx.render_to_pixmap(&mut pixmap);
    println!(
        "time to coarse: {coarse_time:?}, time to fine: {:?}",
        start.elapsed()
    );
    pixmap.unpremultiply();
    let file = std::fs::File::create(out_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixmap.data()).unwrap();
}
