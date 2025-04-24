// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example scenes for Vello Hybrid.

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
}

/// A type-erased example scene
pub struct AnyScene {
    /// The render function that calls the wrapped scene's render method
    render_fn: RenderFn,
}

/// A type-erased render function
type RenderFn = Box<dyn FnMut(&mut Scene, Affine)>;

impl std::fmt::Debug for AnyScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyScene").finish()
    }
}

impl AnyScene {
    /// Create a new `AnyScene` from any type that implements `ExampleScene`
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
    scenes.push(AnyScene::new(image::ImageScene::new()));

    scenes.into_boxed_slice()
}

/// Get all available example scenes (WASM version)
#[cfg(target_arch = "wasm32")]
pub fn get_example_scenes() -> Box<[AnyScene]> {
    vec![
        AnyScene::new(svg::SvgScene::tiger()),
        AnyScene::new(text::TextScene::new("Hello, Vello!")),
        AnyScene::new(simple::SimpleScene::new()),
        AnyScene::new(image::ImageScene::new()),
    ]
    .into_boxed_slice()
}
