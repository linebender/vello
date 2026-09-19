// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A loader for a tiny fragment of SVG

use std::str::FromStr;

use roxmltree::{Document, Node};
use vello::{
    kurbo::{Affine, BezPath, Point, Size, Vec2},
    peniko::color::{self, AlphaColor, DynamicColor, Srgb, palette},
};

pub struct PicoSvg {
    pub items: Vec<Item>,
    pub size: Size,
}

pub enum Item {
    Fill(FillItem),
    Stroke(StrokeItem),
    Group(GroupItem),
}

pub struct StrokeItem {
    pub width: f64,
    pub color: AlphaColor<Srgb>,
    pub path: BezPath,
}

pub struct FillItem {
    pub color: AlphaColor<Srgb>,
    pub path: BezPath,
}

pub struct GroupItem {
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
                if let &[x, y, width, height] = vs.as_slice() {
                    Some((Point { x, y }, Size { width, height }))
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
            fill: Some(palette::css::BLACK),
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
    fill: Option<AlphaColor<Srgb>>,
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
                    let path = bp;
                    if let Some(color) = properties.fill {
                        items.push(Item::Fill(FillItem {
                            color,
                            path: path.clone(),
                        }));
                    }
                    if let Some(stroke_color) = node.attribute("stroke")
                        && stroke_color != "none"
                    {
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

fn parse_color(color: &str) -> AlphaColor<Srgb> {
    let color = color.trim();
    color::parse_color(color.trim())
        .map(DynamicColor::to_alpha_color)
        .unwrap_or(palette::css::FUCHSIA.with_alpha(0.5))
}

fn modify_opacity(
    color: AlphaColor<Srgb>,
    attr_name: &str,
    node: Node<'_, '_>,
) -> AlphaColor<Srgb> {
    if let Some(opacity) = node.attribute(attr_name) {
        let alpha: f32 = if let Some(o) = opacity.strip_suffix('%') {
            let pctg = o.parse().unwrap_or(100.0);
            pctg * 0.01
        } else {
            opacity.parse().unwrap_or(1.0)
        };
        color.with_alpha(alpha.clamp(0., 1.))
    } else {
        color
    }
}

#[cfg(test)]
mod tests {
    use super::parse_color;
    use vello::peniko::color::{AlphaColor, Srgb, palette};

    fn assert_close_color(c1: AlphaColor<Srgb>, c2: AlphaColor<Srgb>) {
        const EPSILON: f32 = 1e-4;
        assert_eq!(c1.cs, c2.cs);
        for i in 0..4 {
            assert!((c1.components[i] - c2.components[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn parse_colors() {
        let lime = palette::css::LIME;
        let lime_a = lime.with_alpha(0.4);

        let named = parse_color("lime");
        assert_close_color(lime, named);

        let hex = parse_color("#00ff00");
        assert_close_color(lime, hex);

        let rgb = parse_color("rgb(0, 255, 0)");
        assert_close_color(lime, rgb);

        let modern = parse_color("color(srgb 0 1 0)");
        assert_close_color(lime, modern);

        let modern_a = parse_color("color(srgb 0 1 0 / 0.4)");
        assert_close_color(lime_a, modern_a);
    }
}
