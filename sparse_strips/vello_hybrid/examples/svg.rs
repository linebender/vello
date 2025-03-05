// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::cast_possible_truncation, reason = "we're doing it on purpose")]

//! SVG example for hybrid renderer

use std::io::BufWriter;
use std::str::FromStr;

use roxmltree::{Document, Node};
use vello_hybrid::api::peniko::color::palette;
use vello_hybrid::api::peniko::kurbo::{Affine, BezPath, Point, Size, Stroke, Vec2};
use vello_hybrid::api::peniko::Color;
use vello_hybrid::api::RenderCtx;
use vello_hybrid::{CsRenderCtx, Pixmap};

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

/// The main function of the example. The German word for main is "Haupt".
pub fn main() {
    let mut ctx = CsRenderCtx::new(WIDTH, HEIGHT);
    let mut args = std::env::args().skip(1);
    let svg_filename = args.next().expect("svg filename is first arg");
    let out_filename = args.next().expect("png out filename is second arg");

    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
    let mut pixmap = Pixmap::new(WIDTH, HEIGHT);
    // Hacky code for crude measurements; change this to arg parsing
    for i in 0..200 {
        ctx.reset();
        let start = std::time::Instant::now();
        render_svg(&mut ctx, &parsed.items);
        let coarse_time = start.elapsed();
        ctx.render_to_pixmap(&mut pixmap);
        if i % 100 == 0 {
            println!(
                "time to coarse: {coarse_time:?}, time to fine: {:?}",
                start.elapsed()
            );
        }
    }
    pixmap.unpremultiply();
    let file = std::fs::File::create(out_filename).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixmap.data()).unwrap();
}

fn render_svg(ctx: &mut impl RenderCtx, items: &[Item]) {
    for item in items {
        match item {
            Item::Fill(fill_item) => ctx.fill(&fill_item.path, fill_item.color.into()),
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.stroke(&stroke_item.path, &style, stroke_item.color.into());
            }
            Item::Group(group_item) => {
                // TODO: apply transform from group
                render_svg(ctx, &group_item.children);
            }
        }
    }
}

// Below is copied, lightly adapted, from Vello.

pub struct PicoSvg {
    pub items: Vec<Item>,
    #[allow(unused, reason = "functionality NYI")]
    pub size: Size,
}

pub enum Item {
    Fill(FillItem),
    Stroke(StrokeItem),
    Group(GroupItem),
}

pub struct StrokeItem {
    pub width: f64,
    pub color: Color,
    pub path: vello_hybrid::api::Path,
}

pub struct FillItem {
    pub color: Color,
    pub path: vello_hybrid::api::Path,
}

pub struct GroupItem {
    #[allow(unused, reason = "functionality NYI")]
    pub affine: Affine,
    pub children: Vec<Item>,
}

struct Parser {
    scale: f64,
}

impl PicoSvg {
    pub fn load(xml_string: &str, scale: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let doc = Document::parse(xml_string)?;
        let root = doc.root_element();
        let mut parser = Parser::new(scale);
        let width = root.attribute("width").and_then(|s| f64::from_str(s).ok());
        let height = root.attribute("height").and_then(|s| f64::from_str(s).ok());
        let (origin, viewbox_size) = root
            .attribute("viewBox")
            .and_then(|vb_attr| {
                let vs: Vec<f64> = vb_attr
                    .split(' ')
                    .map(|s| f64::from_str(s).unwrap())
                    .collect();
                if let &[x, y, vb_width, vb_height] = vs.as_slice() {
                    Some((
                        Point { x, y },
                        Size {
                            width: vb_width,
                            height: vb_height,
                        },
                    ))
                } else {
                    None
                }
            })
            .unzip();

        let mut transform = if let Some(origin) = origin {
            Affine::translate(origin.to_vec2() * -1.0)
        } else {
            Affine::IDENTITY
        };

        transform *= match (width, height, viewbox_size) {
            (None, None, Some(_)) => Affine::IDENTITY,
            (Some(w), Some(h), Some(s)) => {
                Affine::scale_non_uniform(1.0 / s.width * w, 1.0 / s.height * h)
            }
            (Some(w), None, Some(s)) => Affine::scale(1.0 / s.width * w),
            (None, Some(h), Some(s)) => Affine::scale(1.0 / s.height * h),
            _ => Affine::IDENTITY,
        };

        let size = match (width, height, viewbox_size) {
            (None, None, Some(s)) => s,
            (mw, mh, None) => Size {
                width: mw.unwrap_or(300_f64),
                height: mh.unwrap_or(150_f64),
            },
            (Some(w), None, Some(s)) => Size {
                width: w,
                height: 1.0 / w * s.width * s.height,
            },
            (None, Some(h), Some(s)) => Size {
                width: 1.0 / h * s.height * s.width,
                height: h,
            },
            (Some(width), Some(height), Some(_)) => Size { width, height },
        };

        transform *= if scale >= 0.0 {
            Affine::scale(scale)
        } else {
            Affine::new([-scale, 0.0, 0.0, scale, 0.0, 0.0])
        };
        let props = RecursiveProperties {
            fill: Some(Color::BLACK),
        };
        // The root element is the svg document element, which we don't care about
        let mut items = Vec::new();
        for node in root.children() {
            parser.rec_parse(node, &props, &mut items)?;
        }
        let root_group = Item::Group(GroupItem {
            affine: transform,
            children: items,
        });
        Ok(Self {
            items: vec![root_group],
            size,
        })
    }
}

#[derive(Clone)]
struct RecursiveProperties {
    fill: Option<Color>,
}

impl Parser {
    fn new(scale: f64) -> Self {
        Self { scale }
    }

    fn rec_parse(
        &mut self,
        node: Node<'_, '_>,
        properties: &RecursiveProperties,
        items: &mut Vec<Item>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if node.is_element() {
            let mut properties = properties.clone();
            if let Some(fill_color) = node.attribute("fill") {
                if fill_color == "none" {
                    properties.fill = None;
                } else {
                    let color = parse_color(fill_color);
                    let color = modify_opacity(color, "fill-opacity", node);
                    // TODO: Handle recursive opacity properly
                    let color = modify_opacity(color, "opacity", node);
                    properties.fill = Some(color);
                }
            }
            match node.tag_name().name() {
                "g" => {
                    let mut children = Vec::new();
                    let mut affine = Affine::default();
                    if let Some(transform) = node.attribute("transform") {
                        affine = parse_transform(transform);
                    }
                    for child in node.children() {
                        self.rec_parse(child, &properties, &mut children)?;
                    }
                    items.push(Item::Group(GroupItem { affine, children }));
                }
                "path" => {
                    let d = node.attribute("d").ok_or("missing 'd' attribute")?;
                    let bp = BezPath::from_svg(d)?;
                    let path: vello_hybrid::api::Path = bp.into();
                    if let Some(color) = properties.fill {
                        items.push(Item::Fill(FillItem {
                            color,
                            path: path.clone(),
                        }));
                    }
                    if let Some(stroke_color) = node.attribute("stroke") {
                        if stroke_color != "none" {
                            let width = node
                                .attribute("stroke-width")
                                .map(|a| f64::from_str(a).unwrap_or(1.0))
                                .unwrap_or(1.0)
                                * self.scale.abs();
                            let color = parse_color(stroke_color);
                            let color = modify_opacity(color, "stroke-opacity", node);
                            // TODO: Handle recursive opacity properly
                            let color = modify_opacity(color, "opacity", node);
                            items.push(Item::Stroke(StrokeItem { width, color, path }));
                        }
                    }
                }
                other => eprintln!("Unhandled node type {other}"),
            }
        }
        Ok(())
    }
}

fn parse_transform(transform: &str) -> Affine {
    let mut nt = Affine::IDENTITY;
    for ts in transform.split(')').map(str::trim) {
        nt *= if let Some(s) = ts.strip_prefix("matrix(") {
            let vals = s
                .split([',', ' '])
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
                .expect("Could parse all values of 'matrix' as floats");
            Affine::new(
                vals.try_into()
                    .expect("Should be six arguments to `matrix`"),
            )
        } else if let Some(s) = ts.strip_prefix("translate(") {
            if let Ok(vals) = s
                .split([',', ' '])
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
            {
                match vals.as_slice() {
                    &[x, y] => Affine::translate(Vec2 { x, y }),
                    _ => Affine::IDENTITY,
                }
            } else {
                Affine::IDENTITY
            }
        } else if let Some(s) = ts.strip_prefix("scale(") {
            if let Ok(vals) = s
                .split([',', ' '])
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
            {
                match *vals.as_slice() {
                    [x, y] => Affine::scale_non_uniform(x, y),
                    [x] => Affine::scale(x),
                    _ => Affine::IDENTITY,
                }
            } else {
                Affine::IDENTITY
            }
        } else if let Some(s) = ts.strip_prefix("scaleX(") {
            s.trim()
                .parse()
                .ok()
                .map(|x| Affine::scale_non_uniform(x, 1.0))
                .unwrap_or(Affine::IDENTITY)
        } else if let Some(s) = ts.strip_prefix("scaleY(") {
            s.trim()
                .parse()
                .ok()
                .map(|y| Affine::scale_non_uniform(1.0, y))
                .unwrap_or(Affine::IDENTITY)
        } else {
            if !ts.is_empty() {
                eprintln!("Did not understand transform attribute {ts:?})");
            }
            Affine::IDENTITY
        };
    }
    nt
}

fn parse_color(color: &str) -> Color {
    let color = color.trim();
    if let Ok(c) = vello_hybrid::api::peniko::color::parse_color(color) {
        c.to_alpha_color()
    } else {
        palette::css::MAGENTA.with_alpha(0.5)
    }
}

fn modify_opacity(color: Color, attr_name: &str, node: Node<'_, '_>) -> Color {
    if let Some(opacity) = node.attribute(attr_name) {
        let alpha: f64 = if let Some(o) = opacity.strip_suffix('%') {
            let pctg = o.parse().unwrap_or(100.0);
            pctg * 0.01
        } else {
            opacity.parse().unwrap_or(1.0)
        };
        color.with_alpha(alpha as f32)
    } else {
        color
    }
}
