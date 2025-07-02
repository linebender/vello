// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example scene.

use std::fmt;
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_cpu::RenderContext;

use crate::scenes::ExampleScene;

/// SVG scene that renders an SVG file
pub(crate) struct SvgScene {
    transform: Affine,
    svg: PicoSvg,
}

impl fmt::Debug for SvgScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgScene")
    }
}

impl ExampleScene for SvgScene {
    fn render(&mut self, ctx: &mut RenderContext, root_transform: Affine) {
        render_svg(ctx, &self.svg.items, root_transform * self.transform);
    }
}

impl SvgScene {
    /// Create a new `SvgScene` with the Ghost Tiger SVG
    pub(crate) fn new() -> Self {
        // Load the ghost tiger SVG by default
        let svg_content = include_str!("../../../../../../examples/assets/Ghostscript_Tiger.svg");

        let svg = PicoSvg::load(svg_content, 1.0).expect("Failed to parse Ghost Tiger SVG");

        Self {
            transform: Affine::scale(3.0),
            svg,
        }
    }
}

fn render_svg(ctx: &mut RenderContext, items: &[Item], transform: Affine) {
    ctx.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color);
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color);
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg(ctx, &group_item.children, transform * group_item.affine);
                ctx.set_transform(transform);
            }
        }
    }
}
