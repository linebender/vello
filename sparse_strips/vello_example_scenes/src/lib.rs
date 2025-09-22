// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example scenes for Vello Sparse Strips.

pub mod blend;
pub mod clip;
pub mod gradient;
pub mod image;
pub mod path;
pub mod simple;
pub mod svg;
pub mod text;

pub use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder};
use vello_common::kurbo::Affine;
pub use vello_common::kurbo::{BezPath, Rect, Shape, Stroke};
pub use vello_common::mask::Mask;
use vello_common::paint::ImageSource;
pub use vello_common::paint::{Paint, PaintType};
pub use vello_common::peniko::{BlendMode, Fill, FontData};
use vello_common::recording::{Recordable, Recorder, Recording};
#[cfg(feature = "cpu")]
use vello_cpu::RenderContext;
use vello_hybrid::Scene;

/// A generic rendering context.
pub trait RenderingContext: Sized {
    /// The glyph renderer type.
    type GlyphRenderer: GlyphRenderer;

    /// Set the current transform.
    fn set_transform(&mut self, transform: Affine);
    /// Set the current paint transform.
    fn set_paint_transform(&mut self, transform: Affine);
    /// Set the current fill rule.
    fn set_fill_rule(&mut self, fill_rule: Fill);
    /// Set the current paint.
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    /// Set the current stroke style.
    fn set_stroke(&mut self, stroke: Stroke);
    /// Fill a path with the current paint.
    fn fill_path(&mut self, path: &BezPath);
    /// Stroke a path with the current paint and stroke style.
    fn stroke_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint.
    fn fill_rect(&mut self, rect: &Rect);
    /// Create a glyph run builder for text rendering.
    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self::GlyphRenderer>;
    /// Push a clip layer.
    fn push_clip_layer(&mut self, path: &BezPath);
    /// Push a layer with blend mode, alpha, etc.
    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
    );
    /// Pop the current layer.
    fn pop_layer(&mut self);
    /// Record rendering commands into a recording.
    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>));
    /// Generate sparse strips for a recording.
    fn prepare_recording(&mut self, recording: &mut Recording);
    /// Execute a recording directly without preparation.
    fn execute_recording(&mut self, recording: &Recording);
}

#[cfg(feature = "cpu")]
impl RenderingContext for RenderContext {
    type GlyphRenderer = Self;

    fn set_transform(&mut self, transform: Affine) {
        self.set_transform(transform);
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.set_paint(paint);
    }

    fn set_paint_transform(&mut self, transform: Affine) {
        self.set_paint_transform(transform);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.set_fill_rule(fill_rule);
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        self.set_stroke(stroke);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.fill_path(path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.stroke_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.fill_rect(rect);
    }

    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self> {
        self.glyph_run(font)
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_clip_layer(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.push_layer(clip, blend_mode, alpha, mask);
    }

    fn pop_layer(&mut self) {
        self.pop_layer();
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        Recordable::record(self, recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        Recordable::prepare_recording(self, recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        Recordable::execute_recording(self, recording);
    }
}

impl RenderingContext for Scene {
    type GlyphRenderer = Self;

    fn set_transform(&mut self, transform: Affine) {
        self.set_transform(transform);
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.set_paint(paint);
    }

    fn set_paint_transform(&mut self, transform: Affine) {
        self.set_paint_transform(transform);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.set_fill_rule(fill_rule);
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        self.set_stroke(stroke);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.fill_path(path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.stroke_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.fill_rect(rect);
    }

    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self> {
        self.glyph_run(font)
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_clip_layer(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.push_layer(clip, blend_mode, alpha, mask);
    }

    fn pop_layer(&mut self) {
        self.pop_layer();
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        Recordable::record(self, recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        Recordable::prepare_recording(self, recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        Recordable::execute_recording(self, recording);
    }
}

/// Example scene that can maintain state between renders.
pub trait ExampleScene {
    /// Render the scene using the current state.
    fn render(&mut self, target: &mut impl RenderingContext, root_transform: Affine);

    /// Handle key press events (optional).
    /// Returns true if the key was handled, false otherwise.
    fn handle_key(&mut self, _key: &str) -> bool {
        false
    }
}

/// A type-erased example scene.
pub struct AnyScene<T> {
    /// The render function that calls the wrapped scene's render method.
    render_fn: RenderFn<T>,
    /// The key handler function.
    key_handler_fn: KeyHandlerFn,
}

/// A type-erased render function.
type RenderFn<T> = Box<dyn FnMut(&mut T, Affine)>;

/// A type-erased key handler function.
type KeyHandlerFn = Box<dyn FnMut(&str) -> bool>;

impl<T> std::fmt::Debug for AnyScene<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyScene").finish()
    }
}

impl<T: RenderingContext> AnyScene<T> {
    /// Create a new `AnyScene` from any type that implements `ExampleScene`.
    pub fn new<S: ExampleScene + 'static>(scene: S) -> Self {
        let scene = std::rc::Rc::new(std::cell::RefCell::new(scene));
        let scene_clone = scene.clone();

        Self {
            render_fn: Box::new(move |s, transform| scene.borrow_mut().render(s, transform)),
            key_handler_fn: Box::new(move |key| scene_clone.borrow_mut().handle_key(key)),
        }
    }

    /// Render the scene.
    pub fn render(&mut self, target: &mut T, root_transform: Affine) {
        (self.render_fn)(target, root_transform);
    }

    /// Handle key press events.
    /// Returns true if the key was handled, false otherwise.
    pub fn handle_key(&mut self, key: &str) -> bool {
        (self.key_handler_fn)(key)
    }
}

/// Get all available example scenes.
/// Unlike the Wasm version, this function allows for passing custom SVGs.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_example_scenes<T: RenderingContext + 'static>(
    svg_paths: Option<Vec<&str>>,
    img_sources: Vec<ImageSource>,
) -> Box<[AnyScene<T>]> {
    let mut scenes = Vec::new();

    // Create SVG scenes for each provided path.
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
    scenes.push(AnyScene::new(image::ImageScene::new(img_sources)));
    scenes.push(AnyScene::new(gradient::GradientExtendScene::new()));
    scenes.push(AnyScene::new(gradient::RadialScene::new()));
    scenes.push(AnyScene::new(path::FillTypesScene::new()));
    scenes.push(AnyScene::new(path::StrokeStylesScene::new()));
    scenes.push(AnyScene::new(path::StrokeStylesScene::new_non_uniform()));
    scenes.push(AnyScene::new(path::StrokeStylesScene::new_skew()));
    scenes.push(AnyScene::new(path::TrickyStrokesScene::new()));
    scenes.push(AnyScene::new(path::FunkyPathsScene::new()));
    scenes.push(AnyScene::new(path::RobustPathsScene::new()));

    scenes.into_boxed_slice()
}

/// Get all available example scenes (WASM version).
#[cfg(target_arch = "wasm32")]
pub fn get_example_scenes<T: RenderingContext + 'static>(
    img_sources: Vec<ImageSource>,
) -> Box<[AnyScene<T>]> {
    vec![
        AnyScene::new(svg::SvgScene::tiger()),
        AnyScene::new(text::TextScene::new("Hello, Vello!")),
        AnyScene::new(simple::SimpleScene::new()),
        AnyScene::new(clip::ClipScene::new()),
        AnyScene::new(blend::BlendScene::new()),
        AnyScene::new(image::ImageScene::new(img_sources)),
        AnyScene::new(gradient::GradientExtendScene::new()),
        AnyScene::new(gradient::RadialScene::new()),
        AnyScene::new(path::FillTypesScene::new()),
        AnyScene::new(path::StrokeStylesScene::new()),
        AnyScene::new(path::StrokeStylesScene::new_non_uniform()),
        AnyScene::new(path::StrokeStylesScene::new_skew()),
        AnyScene::new(path::TrickyStrokesScene::new()),
        AnyScene::new(path::FunkyPathsScene::new()),
        AnyScene::new(path::RobustPathsScene::new()),
    ]
    .into_boxed_slice()
}
