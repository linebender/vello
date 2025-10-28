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
use std::io::Cursor;
use std::path;
use std::path::Path;
use std::time::{Duration, Instant};
use usvg::tiny_skia_path::PathSegment;
use usvg::{Node, Paint, PaintOrder};
use vello_cpu::color::AlphaColor;
use vello_cpu::kurbo::{Affine, BezPath, Stroke};
use vello_cpu::peniko::Fill;
use vello_cpu::{Level, Pixmap, RenderContext, RenderMode, RenderSettings};

fn main() {
    let args = Args::parse();
    let scale = args.scale;
    let path = Path::new(&args.path);

    let svg = std::fs::read_to_string(path).expect("error reading SVG file");
    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    let width = (tree.size().width() * scale).ceil() as u16;
    let height = (tree.size().height() * scale).ceil() as u16;

    let mut num_iters = 0;
    let settings = RenderSettings {
        level: Level::new(),
        num_threads: args.num_threads as u16,
        render_mode: RenderMode::OptimizeSpeed,
    };
    let mut ctx = RenderContext::new_with(width, height, settings);
    let mut pixmap = Pixmap::new(width, height);
    let mut runtime = Duration::default();

    loop {
        ctx.reset();
        let mut sctx = SVGContext::new_with_scale(scale as f64);

        let start = Instant::now();

        render_tree(&mut ctx, &mut sctx, &tree);
        ctx.flush();
        ctx.render_to_pixmap(&mut pixmap);

        runtime += start.elapsed();
        num_iters += 1;

        if runtime.as_millis() > args.runtime as u128 {
            break;
        }
    }

    let avg_runtime = (runtime.as_millis() as f32) / (num_iters as f32);

    eprintln!("average runtime {avg_runtime}ms");

    write_pixmap(&mut pixmap);
}

fn write_pixmap(pixmap: &mut Pixmap) {
    let data = pixmap.clone().take_unpremultiplied();

    let mut png_data = Vec::new();
    let cursor = Cursor::new(&mut png_data);
    let encoder = PngEncoder::new(cursor);
    encoder
        .write_image(
            bytemuck::cast_slice(&data),
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
    /// For simple benchmarking, our current recommended value is 2000 (i.e. 2 seconds).
    #[arg(long, default_value_t = 0)]
    pub runtime: u32,
    /// The number of additional threads that should be used for rendering.
    #[arg(long, default_value_t = 0)]
    pub num_threads: u32,
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
    let clip_path = {
        group.clip_path().map(|p| {
            let mut path = BezPath::new();
            extract_clip_path(p.root(), &mut path);
            convert_transform(&p.transform()) * path
        })
    };

    ctx.push_layer(clip_path.as_ref(), None, None, None, None);

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

    ctx.pop_layer();
    sctx.pop_transform();
}

// This is only very crude, as in theory we also have to consider nested clip paths, fill rule,
// etc. For now, we just assume a flat hierarchy of clip paths with nonzero filling.
fn extract_clip_path(group: &usvg::Group, path: &mut BezPath) {
    for child in group.children() {
        match child {
            Node::Group(g) => {
                extract_clip_path(g, path);
            }
            Node::Path(p) => {
                path.extend(convert_path_data(p));
            }
            Node::Image(_) => {}
            Node::Text(_) => {}
        }
    }
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
            rctx.set_paint(color);
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
            rctx.set_paint(color);
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
