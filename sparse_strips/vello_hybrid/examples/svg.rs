// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::cast_possible_truncation, reason = "we're doing it on purpose")]

//! SVG example for hybrid renderer

mod pico_svg;

use std::io::BufWriter;

use kurbo::{Affine, Stroke};
use pico_svg::{Item, PicoSvg};
use vello_cpu::pixmap::Pixmap;
use vello_hybrid::RenderContext;

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

/// The main function of the example. The German word for main is "Haupt".
pub fn main() {
    let mut ctx = RenderContext::new(WIDTH as u16, HEIGHT as u16);
    let mut args = std::env::args().skip(1);
    let svg_filename = args.next().expect("svg filename is first arg");
    let out_filename = args.next().expect("png out filename is second arg");

    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
    let mut pixmap = Pixmap::new(WIDTH as u16, HEIGHT as u16);
    // Hacky code for crude measurements; change this to arg parsing
    // for i in 0..200 {
    ctx.reset();
    let start = std::time::Instant::now();
    render_svg(&mut ctx, &parsed.items);
    let coarse_time = start.elapsed();
    ctx.render_to_pixmap(&mut pixmap);
    // if i % 100 == 0 {
    println!(
        "time to coarse: {coarse_time:?}, time to fine: {:?}",
        start.elapsed()
    );
    // }
    // }
    pixmap.unpremultiply();
    let file = std::fs::File::create(out_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixmap.data()).unwrap();
}

fn render_svg(ctx: &mut RenderContext, items: &[Item]) {
    ctx.set_transform(Affine::scale(5.0));
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color.into());
                ctx.fill_path(&fill_item.path.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color.into());
                ctx.stroke_path(&stroke_item.path.path);
            }
            Item::Group(group_item) => {
                // TODO: apply transform from group
                render_svg(ctx, &group_item.children);
            }
        }
    }
}
