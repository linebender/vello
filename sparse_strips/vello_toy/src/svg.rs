// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A very basic SVG renderer built on top of `usvg` and `vello_cpu`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "this is only a debug tool, so we can ignore them"
)]

use clap::Parser;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};
use peniko::Fill;
use peniko::color::AlphaColor;
use peniko::kurbo::Stroke;
use std::io::Cursor;
use std::path;
use std::path::Path;
use std::time::{Duration, Instant};
use usvg::tiny_skia_path::PathSegment;
use usvg::{Node, Paint, PaintOrder};
use vello_common::kurbo::{Affine, BezPath};
use vello_cpu::{Pixmap, RenderContext};

fn main() {
    let args = Args::parse();
    let scale = args.scale;
    let path = Path::new(&args.path);
    let file_name = path.file_name().unwrap().to_str().unwrap();

    let svg = std::fs::read_to_string(path).expect("error reading SVG file");
    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    let width = (tree.size().width() * scale).ceil() as u16;
    let height = (tree.size().height() * scale).ceil() as u16;

    let mut num_iters = 0;
    let mut pixmap;
    let mut ctx;
    let mut runtime = Duration::default();

    loop {
        let mut sctx = SVGContext::new_with_scale(scale as f64);
        pixmap = Pixmap::new(width, height);
        ctx = RenderContext::new(width, height);

        let start = Instant::now();

        render_tree(&mut ctx, &mut sctx, &tree);
        ctx.render_to_pixmap(&mut pixmap);

        runtime += start.elapsed();
        num_iters += 1;

        if runtime.as_millis() > args.runtime as u128 {
            break;
        }
    }

    let avg_runtime = (runtime.as_millis() as f32) / (num_iters as f32);

    eprintln!(
        "ran {} iterations, with an average runtime of {}ms to render {}.",
        num_iters, avg_runtime, file_name
    );

    write_pixmap(&mut pixmap);
}

fn write_pixmap(pixmap: &mut Pixmap) {
    pixmap.unpremultiply();

    let mut png_data = Vec::new();
    let cursor = Cursor::new(&mut png_data);
    let encoder = PngEncoder::new(cursor);
    encoder
        .write_image(
            pixmap.data(),
            pixmap.width() as u32,
            pixmap.height() as u32,
            ExtendedColorType::Rgba8,
        )
        .expect("failed to encode image");

    let path = path::absolute("svg.png").unwrap();
    std::fs::write(&path, png_data).unwrap();
    eprintln!("saved rendered SVG to '{}'", path.display());
}

#[derive(Parser, Debug)]
struct Args {
    /// How much to scale the output image.
    #[arg(long, default_value_t = 1.0)]
    pub scale: f32,
    /// The path to the SVG file.
    #[arg(long)]
    pub path: String,
    /// The target runtime for calculating how many iterations to run, in milliseconds.
    #[arg(long, default_value_t = 2000)]
    pub runtime: u32,
}

struct SVGContext {
    transforms: Vec<Affine>,
}

impl Default for SVGContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SVGContext {
    fn new() -> Self {
        Self {
            transforms: vec![Affine::IDENTITY],
        }
    }

    fn new_with_scale(scale: f64) -> Self {
        Self {
            transforms: vec![Affine::scale(scale)],
        }
    }

    fn push_transform(&mut self, affine: &Affine) {
        let new = *self.transforms.last().unwrap() * *affine;
        self.transforms.push(new);
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn get_transform(&self) -> Affine {
        *self.transforms.last().unwrap()
    }
}

fn render_tree(ctx: &mut RenderContext, sctx: &mut SVGContext, tree: &usvg::Tree) {
    render_group(ctx, sctx, tree.root());
}

fn render_group(ctx: &mut RenderContext, sctx: &mut SVGContext, group: &usvg::Group) {
    sctx.push_transform(&convert_transform(&group.transform()));

    for child in group.children() {
        match child {
            Node::Group(g) => {
                render_group(ctx, sctx, g);
            }
            Node::Path(p) => {
                render_path(ctx, sctx, p);
            }
            Node::Image(_) => {}
            Node::Text(_) => {}
        }
    }

    sctx.pop_transform();
}

fn render_path(ctx: &mut RenderContext, sctx: &mut SVGContext, path: &usvg::Path) {
    if !path.is_visible() {
        return;
    }

    ctx.set_transform(sctx.get_transform());

    let fill = |rctx: &mut RenderContext, p: &usvg::Path| {
        if let Some(fill) = p.fill() {
            let color = match fill.paint() {
                Paint::Color(c) => {
                    AlphaColor::from_rgba8(c.red, c.green, c.blue, fill.opacity().to_u8())
                }
                _ => return,
            };

            rctx.set_fill_rule(convert_fill_rule(fill.rule()));
            rctx.set_paint(color.into());
            rctx.fill_path(&convert_path_data(p));
        }
    };

    let stroke = |rctx: &mut RenderContext, p: &usvg::Path| {
        if let Some(stroke) = p.stroke() {
            let color = match stroke.paint() {
                Paint::Color(c) => {
                    AlphaColor::from_rgba8(c.red, c.green, c.blue, stroke.opacity().to_u8())
                }
                _ => return,
            };

            let stroke = Stroke::new(stroke.width().get() as f64);

            rctx.set_stroke(stroke);
            rctx.set_paint(color.into());
            rctx.stroke_path(&convert_path_data(p));
        }
    };

    if path.paint_order() == PaintOrder::FillAndStroke {
        fill(ctx, path);
        stroke(ctx, path);
    } else {
        stroke(ctx, path);
        fill(ctx, path);
    }
}

fn convert_fill_rule(fill_rule: usvg::FillRule) -> Fill {
    match fill_rule {
        usvg::FillRule::NonZero => Fill::NonZero,
        usvg::FillRule::EvenOdd => Fill::EvenOdd,
    }
}

fn convert_transform(transform: &usvg::Transform) -> Affine {
    Affine::new([
        transform.sx as f64,
        // TODO: It's possible that it should be kx first and then ky
        transform.ky as f64,
        transform.kx as f64,
        transform.sy as f64,
        transform.tx as f64,
        transform.ty as f64,
    ])
}

fn convert_path_data(path: &usvg::Path) -> BezPath {
    let mut bez_path = BezPath::new();

    for e in path.data().segments() {
        match e {
            PathSegment::MoveTo(p) => {
                bez_path.move_to((p.x, p.y));
            }
            PathSegment::LineTo(p) => {
                bez_path.line_to((p.x, p.y));
            }
            PathSegment::QuadTo(p1, p2) => {
                bez_path.quad_to((p1.x, p1.y), (p2.x, p2.y));
            }
            PathSegment::CubicTo(p1, p2, p3) => {
                bez_path.curve_to((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y));
            }
            PathSegment::Close => {
                bez_path.close_path();
            }
        }
    }

    bez_path
}
