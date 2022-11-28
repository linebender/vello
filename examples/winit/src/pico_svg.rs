//! A loader for a tiny fragment of SVG

use std::str::FromStr;

use roxmltree::{Document, Node};

use piet_scene::kurbo::{Affine, BezPath};
use piet_scene::Color;

pub struct PicoSvg {
    pub items: Vec<Item>,
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
        for node in root.children() {
            parser.rec_parse(node)?;
        }
        Ok(PicoSvg { items })
    }
}

impl<'a> Parser<'a> {
    fn new(items: &'a mut Vec<Item>, scale: f64) -> Parser<'a> {
        Parser { scale, items }
    }

    fn rec_parse(&mut self, node: Node) -> Result<(), Box<dyn std::error::Error>> {
        let transform = if self.scale >= 0.0 {
            Affine::scale(self.scale)
        } else {
            Affine::new([-self.scale, 0.0, 0.0, self.scale, 0.0, 1536.0])
        };
        if node.is_element() {
            match node.tag_name().name() {
                "g" => {
                    for child in node.children() {
                        self.rec_parse(child)?;
                    }
                }
                "path" => {
                    let d = node.attribute("d").ok_or("missing 'd' attribute")?;
                    let bp = BezPath::from_svg(d)?;
                    let path = transform * bp;
                    // TODO: default fill color is black, but this is overridden in tiger to this logic.
                    if let Some(fill_color) = node.attribute("fill") {
                        if fill_color != "none" {
                            let color = parse_color(fill_color);
                            let color = modify_opacity(color, "fill-opacity", node);
                            self.items.push(Item::Fill(FillItem {
                                color,
                                path: path.clone(),
                            }));
                        }
                    }
                    if let Some(stroke_color) = node.attribute("stroke") {
                        if stroke_color != "none" {
                            let width = self.scale.abs()
                                * f64::from_str(
                                    node.attribute("stroke-width").ok_or("missing width")?,
                                )?;
                            let color = parse_color(stroke_color);
                            let color = modify_opacity(color, "stroke-opacity", node);
                            self.items
                                .push(Item::Stroke(StrokeItem { width, color, path }));
                        }
                    }
                }
                _ => (),
            }
        }
        Ok(())
    }
}

fn parse_color(color: &str) -> Color {
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
        let alpha = if opacity.ends_with("%") {
            let pctg = opacity[..opacity.len() - 1].parse().unwrap_or(100.0);
            pctg * 0.01
        } else {
            opacity.parse().unwrap_or(1.0)
        } as f64;
        color.a = (alpha.min(1.0).max(0.0) * 255.0).round() as u8;
        color
    } else {
        color
    }
}
