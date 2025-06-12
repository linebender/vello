// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example scene.

use std::fmt;
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_hybrid::Scene;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use crate::ExampleScene;

/// SVG scene that renders an SVG file
pub struct SvgScene {
    transform: Affine,
    svg: PicoSvg,
}

impl fmt::Debug for SvgScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgScene")
    }
}

impl ExampleScene for SvgScene {
    fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        render_svg(scene, &self.svg.items, root_transform * self.transform);
    }
}

impl SvgScene {
    /// Create a new `SvgScene` with the Ghost Tiger SVG
    pub fn tiger() -> Self {
        // Load the ghost tiger SVG by default
        #[cfg(target_arch = "wasm32")]
        let svg_content = include_str!("../../../../../examples/assets/Ghostscript_Tiger.svg");
        #[cfg(not(target_arch = "wasm32"))]
        let svg_content = {
            let cargo_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap();
            &std::fs::read_to_string(
                cargo_dir.join("../../../../examples/assets/Ghostscript_Tiger.svg"),
            )
            .unwrap()
        };

        let svg = PicoSvg::load(svg_content, 1.0).expect("Failed to parse Ghost Tiger SVG");

        Self {
            transform: Affine::scale(3.0),
            svg,
        }
    }

    /// Create a new `SvgScene` with the content from a given file
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_svg_file(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let svg_content = std::fs::read_to_string(path)?;
        let svg = PicoSvg::load(&svg_content, 1.0)?;

        Ok(Self {
            transform: Affine::scale(3.0),
            svg,
        })
    }
}

fn render_svg(ctx: &mut Scene, items: &[Item], transform: Affine) {
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
