use std::fmt;
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_hybrid::Scene;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use crate::ExampleScene;

/// SVG scene that renders the Ghost Tiger SVG and other files
pub struct SvgScene {
    scale: f64,
    ghost_tiger_svg: PicoSvg,
    #[cfg(not(target_arch = "wasm32"))]
    custom_svg: Option<PicoSvg>,
}

impl fmt::Debug for SvgScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgScene")
    }
}

impl ExampleScene for SvgScene {
    fn new() -> Self {
        // Load the ghost tiger SVG
        #[cfg(target_arch = "wasm32")]
        let ghost_tiger = include_str!("../../../../../examples/assets/Ghostscript_Tiger.svg");
        #[cfg(not(target_arch = "wasm32"))]
        let ghost_tiger = {
            let cargo_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap();
            std::fs::read_to_string(
                cargo_dir.join("../../../../examples/assets/Ghostscript_Tiger.svg"),
            )
            .unwrap()
        };

        let ghost_tiger_svg =
            PicoSvg::load(&ghost_tiger, 1.0).expect("Failed to parse Ghost Tiger SVG");

        Self {
            scale: 3.0,
            ghost_tiger_svg,
            #[cfg(not(target_arch = "wasm32"))]
            custom_svg: None,
        }
    }

    fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        // Render the custom SVG if available on native platforms
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(svg) = &self.custom_svg {
            render_svg(scene, self.scale, &svg.items, root_transform);
            return;
        }

        // Render the ghost tiger SVG
        render_svg(
            scene,
            self.scale,
            &self.ghost_tiger_svg.items,
            root_transform,
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SvgScene {
    /// Load a custom SVG file path for display
    pub fn load_svg_file(&mut self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let svg_content = std::fs::read_to_string(path)?;
        let parsed_svg = PicoSvg::load(&svg_content, 1.0)?;
        self.custom_svg = Some(parsed_svg);
        Ok(())
    }
}

fn render_svg(ctx: &mut Scene, scale: f64, items: &[Item], root_transform: Affine) {
    fn render_svg_inner(ctx: &mut Scene, items: &[Item], transform: Affine) {
        ctx.set_transform(transform);
        for item in items {
            match item {
                Item::Fill(fill_item) => {
                    ctx.set_paint(fill_item.color.into());
                    ctx.fill_path(&fill_item.path);
                }
                Item::Stroke(stroke_item) => {
                    let style = Stroke::new(stroke_item.width);
                    ctx.set_stroke(style);
                    ctx.set_paint(stroke_item.color.into());
                    ctx.stroke_path(&stroke_item.path);
                }
                Item::Group(group_item) => {
                    render_svg_inner(ctx, &group_item.children, transform * group_item.affine);
                    ctx.set_transform(transform);
                }
            }
        }
    }

    let scene_transform = Affine::scale(scale);
    render_svg_inner(ctx, items, root_transform * scene_transform);
}
