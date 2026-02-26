// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example scenes for Vello Sparse Strips.

pub mod blend;
pub mod clip;
pub mod filter;
pub mod gradient;
pub mod image;
pub mod multi_image;
pub mod path;
pub mod simple;
pub mod svg;
pub mod text;

use core::any::Any;

use parley_draw::{Glyph, GlyphRunBuilder, ImageCache};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::WHITE;
use vello_common::filter_effects::Filter;
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
    /// Width of the render target in pixels.
    fn width(&self) -> u16;
    /// Height of the render target in pixels.
    fn height(&self) -> u16;

    /// Set the current transform.
    fn set_transform(&mut self, transform: Affine);
    /// Set the current paint transform.
    fn set_paint_transform(&mut self, transform: Affine);
    /// Set the current fill rule.
    fn set_fill_rule(&mut self, fill_rule: Fill);
    /// Set the current paint.
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    /// Set the current filter effect.
    fn set_filter_effect(&mut self, filter: Filter);
    /// Reset the current filter effect.
    fn reset_filter_effect(&mut self);
    /// Push a filter layer.
    fn push_filter_layer(&mut self, filter: Filter);
    /// Set the current stroke style.
    fn set_stroke(&mut self, stroke: Stroke);
    /// Fill a path with the current paint.
    fn fill_path(&mut self, path: &BezPath);
    /// Stroke a path with the current paint and stroke style.
    fn stroke_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint.
    fn fill_rect(&mut self, rect: &Rect);
    /// Push a clip layer.
    fn push_clip_layer(&mut self, path: &BezPath);
    /// Push a clip path.
    fn push_clip_path(&mut self, path: &BezPath);
    /// Push a layer with blend mode, alpha, etc.
    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    );
    /// Pop the current layer.
    fn pop_layer(&mut self);
    /// Pop the last clip path.
    fn pop_clip_path(&mut self);
    /// Record rendering commands into a recording.
    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>));
    /// Generate sparse strips for a recording.
    fn prepare_recording(&mut self, recording: &mut Recording);
    /// Execute a recording directly without preparation.
    fn execute_recording(&mut self, recording: &Recording);

    /// Get the current transform.
    fn transform(&self) -> Affine;

    /// Create a new set of glyph caches for this renderer backend.
    ///
    /// Returns a type-erased cache that must be passed to [`Self::fill_glyphs`].
    fn create_glyph_caches(&self) -> Box<dyn Any>;

    /// Fill glyphs using the renderer's glyph pipeline.
    ///
    /// `glyph_caches` must be the value returned by [`Self::create_glyph_caches`].
    fn fill_glyphs(
        &mut self,
        font: &FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &[i16],
        glyphs: impl Iterator<Item = Glyph>,
        glyph_caches: &mut dyn Any,
    );
}

#[cfg(feature = "cpu")]
impl RenderingContext for RenderContext {
    fn width(&self) -> u16 {
        self.width()
    }

    fn height(&self) -> u16 {
        self.height()
    }

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

    fn set_filter_effect(&mut self, filter: Filter) {
        self.set_filter_effect(filter);
    }

    fn reset_filter_effect(&mut self) {
        self.reset_filter_effect();
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.push_filter_layer(filter);
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

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_clip_layer(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.push_layer(clip, blend_mode, alpha, mask, filter);
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

    fn push_clip_path(&mut self, path: &BezPath) {
        Self::push_clip_path(self, path);
    }

    fn pop_clip_path(&mut self) {
        Self::pop_clip_path(self);
    }

    fn transform(&self) -> Affine {
        *self.transform()
    }

    fn create_glyph_caches(&self) -> Box<dyn Any> {
        Box::new(CpuGlyphState {
            glyph_caches: parley_draw::CpuGlyphCaches::new(512, 512),
            image_cache: ImageCache::new_with_config(parley_draw::AtlasConfig::default()),
        })
    }

    fn fill_glyphs(
        &mut self,
        font: &FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &[i16],
        glyphs: impl Iterator<Item = Glyph>,
        glyph_caches: &mut dyn Any,
    ) {
        let state = glyph_caches
            .downcast_mut::<CpuGlyphState>()
            .expect("wrong glyph cache type for CPU renderer");
        let transform = *self.transform();
        GlyphRunBuilder::new(font.clone(), transform, self)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(bytemuck::cast_slice(normalized_coords))
            .fill_glyphs(glyphs, &mut state.glyph_caches, &mut state.image_cache);
    }
}

/// Glyph caches for the CPU renderer backend.
#[cfg(feature = "cpu")]
struct CpuGlyphState {
    glyph_caches: parley_draw::CpuGlyphCaches,
    image_cache: ImageCache,
}

impl RenderingContext for Scene {
    fn width(&self) -> u16 {
        self.width()
    }

    fn height(&self) -> u16 {
        self.height()
    }

    fn set_transform(&mut self, transform: Affine) {
        self.set_transform(transform);
    }

    fn set_filter_effect(&mut self, filter: Filter) {
        self.set_filter_effect(filter);
    }

    fn reset_filter_effect(&mut self) {
        self.reset_filter_effect();
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.push_filter_layer(filter);
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

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_clip_layer(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        alpha: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.push_layer(clip, blend_mode, alpha, mask, filter);
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

    fn push_clip_path(&mut self, path: &BezPath) {
        Self::push_clip_path(self, path);
    }

    fn pop_clip_path(&mut self) {
        Self::pop_clip_path(self);
    }

    fn transform(&self) -> Affine {
        *self.transform()
    }

    fn create_glyph_caches(&self) -> Box<dyn Any> {
        Box::new(GpuGlyphState {
            glyph_caches: parley_draw::GpuGlyphCaches::with_config(
                parley_draw::GlyphCacheConfig::default(),
            ),
            image_cache: ImageCache::new_with_config(parley_draw::AtlasConfig::default()),
        })
    }

    fn fill_glyphs(
        &mut self,
        font: &FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &[i16],
        glyphs: impl Iterator<Item = Glyph>,
        glyph_caches: &mut dyn Any,
    ) {
        let state = glyph_caches
            .downcast_mut::<GpuGlyphState>()
            .expect("wrong glyph cache type for hybrid renderer");
        let transform = *self.transform();
        GlyphRunBuilder::new(font.clone(), transform, self)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(bytemuck::cast_slice(normalized_coords))
            .fill_glyphs(glyphs, &mut state.glyph_caches, &mut state.image_cache);
    }
}

/// Glyph caches for the hybrid (GPU) renderer backend.
struct GpuGlyphState {
    glyph_caches: parley_draw::GpuGlyphCaches,
    image_cache: ImageCache,
}

/// Example scene that can maintain state between renders.
pub trait ExampleScene {
    /// Render the scene using the current state.
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine);

    /// Handle key press events (optional).
    /// Returns true if the key was handled, false otherwise.
    fn handle_key(&mut self, _key: &str) -> bool {
        false
    }

    /// Optional status string shown in the window title (e.g. element count).
    fn status(&self) -> Option<String> {
        None
    }
}

/// A type-erased example scene.
pub struct AnyScene<T> {
    /// The render function that calls the wrapped scene's render method.
    render_fn: RenderFn<T>,
    /// The key handler function.
    key_handler_fn: KeyHandlerFn,
    /// The status query function.
    status_fn: StatusFn,
    /// Whether to show the wide tile columns overlay.
    show_widetile_columns: bool,
}

/// A type-erased render function.
type RenderFn<T> = Box<dyn FnMut(&mut T, Affine)>;

/// A type-erased key handler function.
type KeyHandlerFn = Box<dyn FnMut(&str) -> bool>;

/// A type-erased status function.
type StatusFn = Box<dyn Fn() -> Option<String>>;

impl<T> std::fmt::Debug for AnyScene<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyScene")
            .field("show_tile_grid", &self.show_widetile_columns)
            .finish_non_exhaustive()
    }
}

impl<T: RenderingContext> AnyScene<T> {
    /// Create a new `AnyScene` from any type that implements `ExampleScene`.
    pub fn new<S: ExampleScene + 'static>(scene: S) -> Self {
        let scene = std::rc::Rc::new(std::cell::RefCell::new(scene));
        let scene_clone = scene.clone();
        let scene_status = scene.clone();

        Self {
            render_fn: Box::new(move |s, transform| scene.borrow_mut().render(s, transform)),
            key_handler_fn: Box::new(move |key| scene_clone.borrow_mut().handle_key(key)),
            status_fn: Box::new(move || scene_status.borrow().status()),
            show_widetile_columns: false,
        }
    }

    /// Render the scene.
    pub fn render(&mut self, ctx: &mut T, root_transform: Affine) {
        // Render the actual scene content
        (self.render_fn)(ctx, root_transform);

        // Draw tile grid overlay if enabled
        if self.show_widetile_columns {
            self.draw_widetile_columns(ctx);
        }
    }

    /// Handle key press events.
    /// Returns true if the key was handled, false otherwise.
    pub fn handle_key(&mut self, key: &str) -> bool {
        // First check for global shortcuts
        match key {
            "t" | "T" => {
                self.toggle_tile_grid();
                return true;
            }
            _ => {}
        }

        // Then delegate to the scene-specific handler
        (self.key_handler_fn)(key)
    }

    /// Get an optional status string from the scene.
    pub fn status(&self) -> Option<String> {
        (self.status_fn)()
    }

    /// Toggle the tile grid overlay.
    pub fn toggle_tile_grid(&mut self) {
        self.show_widetile_columns = !self.show_widetile_columns;
    }

    /// Draw the tile grid overlay.
    ///
    /// Note: We don't restore transform/paint since this runs at the end of `render()`.
    fn draw_widetile_columns(&self, ctx: &mut T) {
        ctx.set_transform(Affine::IDENTITY);
        ctx.set_paint(WHITE);

        let vw = ctx.width() as f64;
        let vh = ctx.height() as f64;

        let mut tile_x = 0.0;
        let line_width = 1.0;
        while tile_x <= vw {
            ctx.fill_rect(&Rect::from_points((tile_x, 0.0), (tile_x + line_width, vh)));
            tile_x += WideTile::WIDTH as f64;
        }
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

    scenes.push(AnyScene::new(text::TextScene::default()));
    scenes.push(AnyScene::new(simple::SimpleScene::new()));
    scenes.push(AnyScene::new(clip::ClipScene::new()));
    #[cfg(feature = "cpu")]
    scenes.push(AnyScene::new(filter::FilterScene::new()));
    scenes.push(AnyScene::new(blend::BlendScene::new()));
    let flower_source = img_sources[0].clone();
    scenes.push(AnyScene::new(image::ImageScene::new(img_sources)));
    scenes.push(AnyScene::new(multi_image::MultiImageScene::new(
        flower_source,
    )));
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
    let scenes = vec![
        AnyScene::new(svg::SvgScene::tiger()),
        AnyScene::new(simple::SimpleScene::new()),
        #[cfg(feature = "cpu")]
        AnyScene::new(filter::FilterScene::new()),
        AnyScene::new(clip::ClipScene::new()),
        AnyScene::new(blend::BlendScene::new()),
        AnyScene::new(image::ImageScene::new(img_sources.clone())),
        AnyScene::new(multi_image::MultiImageScene::new(img_sources[0].clone())),
        AnyScene::new(gradient::GradientExtendScene::new()),
        AnyScene::new(gradient::RadialScene::new()),
        AnyScene::new(path::FillTypesScene::new()),
        AnyScene::new(path::StrokeStylesScene::new()),
        AnyScene::new(path::StrokeStylesScene::new_non_uniform()),
        AnyScene::new(path::StrokeStylesScene::new_skew()),
        AnyScene::new(path::TrickyStrokesScene::new()),
        AnyScene::new(path::FunkyPathsScene::new()),
        AnyScene::new(path::RobustPathsScene::new()),
    ];
    scenes.into_boxed_slice()
}
