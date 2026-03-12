// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG (Ghostscript Tiger) benchmark scene.

use super::{BenchScene, Param};
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_hybrid::{Scene, WebGlRenderer};

/// Benchmark scene that renders the Ghostscript Tiger SVG.
#[derive(Debug)]
pub struct TigerScene {
    svg: PicoSvg,
}

impl Default for TigerScene {
    fn default() -> Self {
        Self::new()
    }
}

impl TigerScene {
    /// Create a new Tiger SVG benchmark scene.
    pub fn new() -> Self {
        let svg_content = include_str!("../../../../../../examples/assets/Ghostscript_Tiger.svg");
        let svg = PicoSvg::load(svg_content, 1.0).expect("Failed to parse Tiger SVG");
        Self { svg }
    }
}

/// Recursively render SVG items into a `Scene`.
fn render_svg(scene: &mut Scene, items: &[Item], transform: Affine) {
    scene.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                scene.set_paint(fill_item.color);
                scene.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                scene.set_stroke(style);
                scene.set_paint(stroke_item.color);
                scene.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg(scene, &group_item.children, transform * group_item.affine);
                scene.set_transform(transform);
            }
        }
    }
}

impl BenchScene for TigerScene {
    fn name(&self) -> &str {
        "Tiger SVG"
    }

    fn params(&self) -> Vec<Param> {
        vec![]
    }

    fn set_param(&mut self, _name: &str, _value: f64) {}

    fn render(
        &mut self,
        scene: &mut Scene,
        _renderer: &mut WebGlRenderer,
        width: u32,
        height: u32,
        _time: f64,
        view: Affine,
    ) {
        let svg_w = self.svg.size.width;
        let svg_h = self.svg.size.height;

        // Scale to fit viewport.
        let s = (width as f64 / svg_w).min(height as f64 / svg_h);

        // Center in viewport.
        let tx = (width as f64 - svg_w * s) / 2.0;
        let ty = (height as f64 - svg_h * s) / 2.0;

        let transform = view * Affine::translate((tx, ty)) * Affine::scale(s);
        render_svg(scene, &self.svg.items, transform);
        scene.reset_transform();
    }
}
