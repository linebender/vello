// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A loader for a tiny fragment of SVG

use std::{num::ParseFloatError, str::FromStr};

use roxmltree::{Document, Node};
use vello::{
    kurbo::{Affine, BezPath, Point, Size, Vec2},
    peniko::Color,
};

pub struct PicoSvg {
    pub items: Vec<Item>,
    pub size: Size,
}

pub enum Item {
    Fill(FillItem),
    Stroke(StrokeItem),
}

pub struct StrokeItem {
    pub width: f64,
    pub color: Color,
    pub path: BezPath,
}

pub struct FillItem {
    pub color: Color,
    pub path: BezPath,
}

struct Parser<'a> {
    scale: f64,
    items: &'a mut Vec<Item>,
}

impl PicoSvg {
    pub fn load(xml_string: &str, scale: f64) -> Result<PicoSvg, Box<dyn std::error::Error>> {
        let doc = Document::parse(xml_string)?;
        let root = doc.root_element();
        let mut items = Vec::new();
        let mut parser = Parser::new(&mut items, scale);
        let (origin, size) = if let Some(vb_attr) = root.attribute("viewBox") {
            let vs: Vec<f64> = vb_attr
                .split(' ')
                .map(|s| f64::from_str(s).unwrap())
                .collect();
            match vs.as_slice() {
                &[x, y, width, height] => (Point { x, y }, Size { width, height }),
                _ => (
                    Point::ZERO,
                    Size {
                        width: 300.0,
                        height: 150.0,
                    },
                ),
            }
        } else if let Some((width, height)) = root.attribute("width").zip(root.attribute("height"))
        {
            (
                Point::ZERO,
                Size {
                    width: f64::from_str(width).unwrap(),
                    height: f64::from_str(height).unwrap(),
                },
            )
        } else {
            (
                Point::ZERO,
                Size {
                    width: 300.0,
                    height: 150.0,
                },
            )
        };
        let transform = Affine::translate(origin.to_vec2() * -1.0)
            * if scale >= 0.0 {
                Affine::scale(scale)
            } else {
                Affine::new([-scale, 0.0, 0.0, scale, 0.0, 0.0])
            };
        let props = RecursiveProperties {
            transform,
            fill: Some(Color::BLACK),
        };
        // The root element is the svg document element, which we don't care about
        for node in root.children() {
            parser.rec_parse(node, &props)?;
        }
        Ok(PicoSvg { items, size })
    }
}

#[derive(Clone)]
struct RecursiveProperties {
    transform: Affine,
    fill: Option<Color>,
}

impl<'a> Parser<'a> {
    fn new(items: &'a mut Vec<Item>, scale: f64) -> Parser<'a> {
        Parser { scale, items }
    }

    fn rec_parse(
        &mut self,
        node: Node,
        properties: &RecursiveProperties,
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
            if let Some(transform) = node.attribute("transform") {
                properties.transform *= parse_transform(transform);
            }
            match node.tag_name().name() {
                "g" => {
                    for child in node.children() {
                        self.rec_parse(child, &properties)?;
                    }
                }
                "path" => {
                    let d = node.attribute("d").ok_or("missing 'd' attribute")?;
                    let bp = BezPath::from_svg(d)?;
                    let path = properties.transform * bp;
                    if let Some(color) = properties.fill {
                        self.items.push(Item::Fill(FillItem {
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
                            self.items
                                .push(Item::Stroke(StrokeItem { width, color, path }));
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
    let transform = transform.trim();
    if transform.starts_with("matrix(") {
        let vals = transform["matrix(".len()..transform.len() - 1]
            .split(|c| matches!(c, ',' | ' '))
            .map(str::parse)
            .collect::<Result<Vec<f64>, ParseFloatError>>()
            .expect("Could parse all values of 'matrix' as floats");
        Affine::new(
            vals.try_into()
                .expect("Should be six arguments to `matrix`"),
        )
    } else if transform.starts_with("translate(") {
        if let Ok(vals) = transform["translate(".len()..transform.len() - 1]
            .split(' ')
            .map(str::trim)
            .map(str::parse)
            .collect::<Result<Vec<f64>, ParseFloatError>>()
        {
            match vals.as_slice() {
                &[x, y] => Affine::translate(Vec2 { x, y }),
                _ => Affine::IDENTITY,
            }
        } else {
            Affine::IDENTITY
        }
    } else if transform.starts_with("scale(") {
        if let Ok(vals) = transform["scale(".len()..transform.len() - 1]
            .split(' ')
            .map(str::trim)
            .map(str::parse)
            .collect::<Result<Vec<f64>, ParseFloatError>>()
        {
            match vals.as_slice() {
                &[x, y] => Affine::scale_non_uniform(x, y),
                _ => Affine::IDENTITY,
            }
        } else {
            Affine::IDENTITY
        }
    } else {
        eprintln!("Did not understand transform attribute {transform:?}");
        Affine::IDENTITY
    }
}

fn parse_color(color: &str) -> Color {
    let color = color.trim();
    if color.as_bytes()[0] == b'#' {
        let mut hex = u32::from_str_radix(&color[1..], 16).unwrap();
        if color.len() == 4 {
            hex = (hex >> 8) * 0x110000 + ((hex >> 4) & 0xf) * 0x1100 + (hex & 0xf) * 0x11;
        }
        let rgba = (hex << 8) + 0xff;
        let (r, g, b, a) = (
            (rgba >> 24 & 255) as u8,
            ((rgba >> 16) & 255) as u8,
            ((rgba >> 8) & 255) as u8,
            (rgba & 255) as u8,
        );
        Color::rgba8(r, g, b, a)
    } else if color.starts_with("rgb(") {
        let mut iter = color[4..color.len() - 1].split(',');
        let r = u8::from_str(iter.next().unwrap()).unwrap();
        let g = u8::from_str(iter.next().unwrap()).unwrap();
        let b = u8::from_str(iter.next().unwrap()).unwrap();
        Color::rgb8(r, g, b)
    } else {
        Color::rgba8(255, 0, 255, 0x80)
    }
}

fn modify_opacity(mut color: Color, attr_name: &str, node: Node) -> Color {
    if let Some(opacity) = node.attribute(attr_name) {
        let alpha: f64 = if let Some(o) = opacity.strip_suffix('%') {
            let pctg = o.parse().unwrap_or(100.0);
            pctg * 0.01
        } else {
            opacity.parse().unwrap_or(1.0)
        };
        color.a = (alpha.min(1.0).max(0.0) * 255.0).round() as u8;
        color
    } else {
        color
    }
}
