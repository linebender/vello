// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark scene definitions.

mod rect;
mod svg;

pub use rect::RectScene;
pub use svg::TigerScene;
use vello_common::kurbo::Affine;
use vello_hybrid::{Scene, WebGlRenderer};

/// A tweakable parameter for a benchmark scene.
#[derive(Debug, Clone)]
pub struct Param {
    /// Internal name used as key.
    pub name: &'static str,
    /// Human-readable label for UI.
    pub label: &'static str,
    /// The kind of control: slider range or dropdown select.
    pub kind: ParamKind,
    /// Current value.
    pub value: f64,
}

/// Whether a parameter is a numeric slider or a dropdown select.
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// A range slider with min, max, and step.
    Slider {
        /// Minimum value.
        min: f64,
        /// Maximum value.
        max: f64,
        /// Step increment.
        step: f64,
    },
    /// A dropdown select with `(label, value)` options.
    Select(Vec<(&'static str, f64)>),
}

/// Trait for benchmark scenes with tweakable parameters.
pub trait BenchScene {
    /// Display name of this scene.
    fn name(&self) -> &str;
    /// Return the list of tweakable parameters.
    fn params(&self) -> Vec<Param>;
    /// Update a parameter by name.
    fn set_param(&mut self, name: &str, value: f64);
    /// Render one frame into the scene.
    ///
    /// `view` is a view transform (e.g. pan/zoom) applied by the interactive mode.
    /// Scenes should compose it with their own transforms.
    fn render(
        &mut self,
        scene: &mut Scene,
        renderer: &mut WebGlRenderer,
        width: u32,
        height: u32,
        time: f64,
        view: Affine,
    );
}

/// Return all available benchmark scenes.
pub fn all_scenes() -> Vec<Box<dyn BenchScene>> {
    vec![Box::new(RectScene::new()), Box::new(TigerScene::new())]
}
