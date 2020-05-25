//! A loader for a tiny fragment of SVG

use std::str::FromStr;

use roxmltree::Document;

use piet::kurbo::{Affine, BezPath};

use piet::{Color, RenderContext};

pub struct PicoSvg {
    items: Vec<Item>,
}

pub enum Item {
    Fill(FillItem),
    Stroke(StrokeItem),
}

pub struct StrokeItem {
    width: f64,
    color: Color,
    path: BezPath,
}

pub struct FillItem {
    color: Color,
    path: BezPath,
}

impl PicoSvg {
    pub fn load(xml_string: &str, scale: f64) -> Result<PicoSvg, Box<dyn std::error::Error>> {
        let doc = Document::parse(xml_string)?;
        let root = doc.root_element();
        let g = root.first_element_child().ok_or("no root element")?;
        let mut items = Vec::new();
        for el in g.children() {
            if el.is_element() {
                let d = el.attribute("d").ok_or("missing 'd' attribute")?;
                let bp = BezPath::from_svg(d)?;
                let path = Affine::scale(scale) * bp;
                if let Some(fill_color) = el.attribute("fill") {
                    let color = parse_color(fill_color);
                    items.push(Item::Fill(FillItem {
                        color,
                        path: path.clone(),
                    }));
                }
                if let Some(stroke_color) = el.attribute("stroke") {
                    let width =
                        scale * f64::from_str(el.attribute("stroke-width").ok_or("missing width")?)?;
                    let color = parse_color(stroke_color);
                    items.push(Item::Stroke(StrokeItem { width, color, path }));
                }
            }
        }
        Ok(PicoSvg { items })
    }

    pub fn render(&self, rc: &mut impl RenderContext) {
        for item in &self.items {
            match item {
                Item::Fill(fill_item) => {
                    rc.fill(&fill_item.path, &fill_item.color);
                    //rc.stroke(&fill_item.path, &fill_item.color, 1.0);
                }
                Item::Stroke(stroke_item) => {
                    rc.stroke(&stroke_item.path, &stroke_item.color, stroke_item.width);
                }
            }
        }
    }
}

fn parse_color(color: &str) -> Color {
    if color.as_bytes()[0] == b'#' {
        let mut hex = u32::from_str_radix(&color[1..], 16).unwrap();
        if color.len() == 4 {
            hex = (hex >> 8) * 0x110000 + ((hex >> 4) & 0xf) * 0x1100 + (hex & 0xf) * 0x11;
        }
        Color::from_rgba32_u32((hex << 8) + 0xff)
    } else {
        Color::from_rgba32_u32(0xff00ff80)
    }
}
