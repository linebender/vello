// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example scenes for Vello CPU.

pub(crate) mod clip;
pub(crate) mod image;
pub(crate) mod simple;
pub(crate) mod svg;
pub(crate) mod text;

use vello_common::kurbo::Affine;
use vello_cpu::RenderContext;

/// Example scene that can maintain state between renders
pub(crate) trait ExampleScene {
    /// Render the scene using the current render context
    fn render(&mut self, ctx: &mut RenderContext, root_transform: Affine);
}

/// A type-erased example scene
pub(crate) struct AnyScene {
    /// The render function that calls the wrapped scene's render method
    render_fn: RenderFn,
}

/// A type-erased render function
type RenderFn = Box<dyn FnMut(&mut RenderContext, Affine)>;

impl std::fmt::Debug for AnyScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyScene").finish()
    }
}

impl AnyScene {
    /// Create a new `AnyScene` from any type that implements `ExampleScene`
    pub(crate) fn new<T: ExampleScene + 'static>(mut scene: T) -> Self {
        Self {
            render_fn: Box::new(move |s, transform| scene.render(s, transform)),
        }
    }

    /// Render the scene
    pub(crate) fn render(&mut self, scene: &mut RenderContext, root_transform: Affine) {
        (self.render_fn)(scene, root_transform);
    }
}

/// Get all available example scenes
pub(crate) fn get_example_scenes() -> Box<[AnyScene]> {
    vec![
        AnyScene::new(svg::SvgScene::new()),
        AnyScene::new(simple::SimpleScene::new()),
        AnyScene::new(text::TextScene::new("Hello World from vello_cpu")),
        AnyScene::new(clip::ClipScene::new()),
        AnyScene::new(image::ImageScene::new()),
    ]
    .into_boxed_slice()
}
