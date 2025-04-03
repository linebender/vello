pub mod simple;
pub mod svg;
pub mod text;

use vello_common::kurbo::Affine;
use vello_hybrid::Scene;

/// Trait for scenes that maintain state between renders
pub trait ExampleScene {
    /// Render the scene using the current state
    fn render(&mut self, scene: &mut Scene, root_transform: Affine);

    /// Create a new instance of the scene with default state
    fn new() -> Self
    where
        Self: Sized;
}

/// A type-erased example scene that can be stored in collections
pub struct AnyScene {
    /// The render function that calls the wrapped scene's render method
    render_fn: Box<dyn FnMut(&mut Scene, Affine)>,
}

impl AnyScene {
    /// Create a new AnyScene from any type that implements ExampleScene
    pub fn new<T: ExampleScene + 'static>(mut scene: T) -> Self {
        Self {
            render_fn: Box::new(move |s, transform| scene.render(s, transform)),
        }
    }

    /// Render the scene
    pub fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        (self.render_fn)(scene, root_transform);
    }
}

/// Get all available example scenes
/// Unlike the Wasm version, this function allows for passing custom SVGs.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_example_scenes(svg_path: Option<&str>) -> Box<[AnyScene]> {
    let mut scenes = Vec::with_capacity(3);

    // Create SVG scene first, potentially with custom SVG
    let mut svg_scene = svg::SvgScene::new();
    if let Some(path) = svg_path {
        svg_scene.load_svg_file(path.into()).unwrap();
    }

    scenes.push(AnyScene::new(svg_scene));
    scenes.push(AnyScene::new(text::TextScene::new()));
    scenes.push(AnyScene::new(simple::SimpleScene::new()));

    scenes.into_boxed_slice()
}

/// Get all available example scenes (WASM version)
#[cfg(target_arch = "wasm32")]
pub fn get_example_scenes() -> Box<[AnyScene]> {
    vec![
        AnyScene::new(svg::SvgScene::new()),
        AnyScene::new(text::TextScene::new()),
        AnyScene::new(simple::SimpleScene::new()),
    ]
    .into_boxed_slice()
}
