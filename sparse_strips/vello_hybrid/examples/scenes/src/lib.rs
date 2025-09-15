// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example scenes for Vello Hybrid.

pub mod blend;
pub mod clip;
pub mod gradient;
pub mod image;
pub mod simple;
pub mod svg;
pub mod text;

use vello_common::kurbo::Affine;
use vello_hybrid::Scene;

/// Example scene that can maintain state between renders
pub trait ExampleScene {
    /// Render the scene using the current state
    fn render(&mut self, scene: &mut Scene, root_transform: Affine);

    /// Handle key press events (optional)
    /// Returns true if the key was handled, false otherwise
    fn handle_key(&mut self, _key: &str) -> bool {
        false
    }
}

/// A type-erased example scene
pub struct AnyScene {
    /// The render function that calls the wrapped scene's render method
    render_fn: RenderFn,
    /// The key handler function
    key_handler_fn: KeyHandlerFn,
}

/// A type-erased render function
type RenderFn = Box<dyn FnMut(&mut Scene, Affine)>;

/// A type-erased key handler function
type KeyHandlerFn = Box<dyn FnMut(&str) -> bool>;

impl std::fmt::Debug for AnyScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyScene").finish()
    }
}

impl AnyScene {
    /// Create a new `AnyScene` from any type that implements `ExampleScene`
    pub fn new<T: ExampleScene + 'static>(scene: T) -> Self {
        let scene = std::rc::Rc::new(std::cell::RefCell::new(scene));
        let scene_clone = scene.clone();

        Self {
            render_fn: Box::new(move |s, transform| scene.borrow_mut().render(s, transform)),
            key_handler_fn: Box::new(move |key| scene_clone.borrow_mut().handle_key(key)),
        }
    }

    /// Render the scene
    pub fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        (self.render_fn)(scene, root_transform);
    }

    /// Handle key press events
    /// Returns true if the key was handled, false otherwise
    pub fn handle_key(&mut self, key: &str) -> bool {
        (self.key_handler_fn)(key)
    }
}

/// Get all available example scenes
/// Unlike the Wasm version, this function allows for passing custom SVGs.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_example_scenes(svg_paths: Option<Vec<&str>>) -> Box<[AnyScene]> {
    let mut scenes = Vec::new();

    // Create SVG scenes for each provided path
    if let Some(paths) = svg_paths {
        for path in paths {
            scenes.push(AnyScene::new(
                svg::SvgScene::with_svg_file(path.into()).unwrap(),
            ));
        }
    } else {
        scenes.push(AnyScene::new(svg::SvgScene::tiger()));
    }

    scenes.push(AnyScene::new(text::TextScene::new("Hello, Vello!")));
    scenes.push(AnyScene::new(simple::SimpleScene::new()));
    scenes.push(AnyScene::new(clip::ClipScene::new()));
    scenes.push(AnyScene::new(blend::BlendScene::new()));
    scenes.push(AnyScene::new(image::ImageScene::new()));
    scenes.push(AnyScene::new(gradient::GradientExtendScene::new()));
    scenes.push(AnyScene::new(gradient::RadialScene::new()));

    scenes.into_boxed_slice()
}

/// Get all available example scenes (WASM version)
#[cfg(target_arch = "wasm32")]
pub fn get_example_scenes() -> Box<[AnyScene]> {
    vec![
        AnyScene::new(svg::SvgScene::tiger()),
        AnyScene::new(text::TextScene::new("Hello, Vello!")),
        AnyScene::new(simple::SimpleScene::new()),
        AnyScene::new(clip::ClipScene::new()),
        AnyScene::new(blend::BlendScene::new()),
        AnyScene::new(image::ImageScene::new()),
        AnyScene::new(gradient::GradientExtendScene::new()),
        AnyScene::new(gradient::RadialScene::new()),
    ]
    .into_boxed_slice()
}
