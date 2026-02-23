// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::clip::ClipContext;
use vello_common::coarse::{MODE_HYBRID, Wide};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter_effects::Filter;
use vello_common::glyph::{GlyphCaches, GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{ImageId, ImageSource, Paint, PaintType};
use vello_common::peniko::Extend;
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::recording::{PushLayerCommand, Recordable, Recorder, Recording, RenderCommand};
use vello_common::render_graph::RenderGraph;
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};

use crate::AtlasConfig;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// Hints provided to the renderer to optimize performance.
#[derive(Copy, Clone, Debug)]
pub struct RenderHints(u32);

impl Default for RenderHints {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderHints {
    const DEFAULT_BLENDING_ONLY: u32 = 1 << 0;

    /// Create a new set of render hints.
    #[inline(always)]
    pub fn new() -> Self {
        Self(0)
    }

    /// Caller guarantees that the scene will only use the default blend mode
    /// (normal, source-over). This enables the blit rect fast path.
    ///
    /// # Panics
    ///
    /// The renderer will panic if a non-default blend mode is used.
    #[inline(always)]
    pub fn expect_only_default_blending(self) -> Self {
        Self(self.0 | Self::DEFAULT_BLENDING_ONLY)
    }

    #[inline(always)]
    fn blit_rect_pipeline_enabled(&self) -> bool {
        (self.0 & Self::DEFAULT_BLENDING_ONLY) != 0
    }
}

/// A blit rect for the blit rect pipeline.
///
/// At scene-encode time we do not have enough information to fully resolve a
/// blit rect (the [`ImageResource`](crate::image_cache::ImageResource) with
/// atlas coordinates is only available at render time). This struct captures
/// everything known at scene time; the renderer resolves it later.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlitRect {
    /// Screen-space center of the quad (after geometry transform): [x, y].
    pub center: [f32; 2],
    /// First column vector of the rect-to-screen transform.
    pub col0: [f32; 2],
    /// Second column vector of the rect-to-screen transform.
    pub col1: [f32; 2],
    /// Pre-transform rect dimensions in geometry space: [w, h].
    pub rect_wh: [u16; 2],
    /// Source image reference (resolved to atlas coords at render time).
    pub image_id: ImageId,
    /// Image-space origin: where the rect's top-left maps to in image coords.
    pub img_origin: [f32; 2],
}

/// A single command in the fast path buffer.
#[derive(Debug)]
pub(crate) enum FastPathCommand {
    /// A regular path fill (strip-based).
    Path(BufferedPath),
    /// A blit rect (image-based, bypasses strips entirely).
    Blit(BlitRect),
}

/// Metadata for a single buffered path in the fast path.
#[derive(Debug)]
pub(crate) struct BufferedPath {
    pub(crate) strip_start: usize,
    pub(crate) strip_end: usize,
    pub(crate) paint: Paint,
}

/// Accumulates strips across path draws when no layers are active.
///
/// When a scene consists entirely of simple path fills with no layers, clips, or blending,
/// this buffer collects strips directly and converts them to `GpuStrip`s at render time,
/// bypassing both coarse rasterization and scheduling.
#[derive(Debug, Default)]
pub(crate) struct FastPathBuffer {
    pub(crate) strips: Vec<Strip>,
    pub(crate) commands: Vec<FastPathCommand>,
}

impl FastPathBuffer {
    fn clear(&mut self) {
        self.strips.clear();
        self.commands.clear();
    }
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The configuration for the texture atlas.
    ///
    /// This controls how images are managed in GPU memory through texture atlases.
    /// The atlas system packs multiple images into larger textures to reduce the
    /// number of GPU texture bindings. This config allows customizing atlas parameters such as:
    /// - The number and size of atlases
    /// - How images are allocated across multiple atlases
    /// - Whether new atlases are automatically created when needed
    ///
    /// Adjusting these settings can affect memory usage and rendering performance
    /// depending on your application's image usage patterns.
    pub atlas_config: AtlasConfig,
    /// Render hints for the renderer.
    pub render_hints: RenderHints,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::fallback()),
            atlas_config: AtlasConfig::default(),
            render_hints: RenderHints::new(),
        }
    }
}

/// A render state which contains the style properties for path rendering and
/// the current transform.
///
/// This is used to save and restore rendering state during recording operations.
#[derive(Debug)]
struct RenderState {
    /// The paint type (solid color, gradient, or image).
    pub(crate) paint: PaintType,
    /// Transform applied to the paint coordinates.
    pub(crate) paint_transform: Affine,
    /// Stroke style for path stroking operations.
    pub(crate) stroke: Stroke,
    /// Transform applied to geometry.
    pub(crate) transform: Affine,
    /// Fill rule for path filling operations.
    pub(crate) fill_rule: Fill,
    /// Blend mode for compositing.
    pub(crate) blend_mode: BlendMode,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    /// Render hints for the renderer.
    render_hints: RenderHints,
    /// Width of the rendering surface in pixels.
    pub(crate) width: u16,
    /// Height of the rendering surface in pixels.
    pub(crate) height: u16,
    /// Wide coarse rasterizer for generating binned draw commands.
    pub(crate) wide: Wide<MODE_HYBRID>,
    clip_context: ClipContext,
    pub(crate) paint: PaintType,
    /// Transform applied to paint coordinates.
    pub(crate) paint_transform: Affine,
    pub(crate) aliasing_threshold: Option<u8>,
    /// Storage for encoded gradient and image paint data.
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Current stroke style for path stroking operations.
    pub(crate) stroke: Stroke,
    /// Current transform applied to geometry.
    pub(crate) transform: Affine,
    /// Current fill rule for path filling operations.
    pub(crate) fill_rule: Fill,
    /// Current blend mode for compositing.
    pub(crate) blend_mode: BlendMode,
    /// Generator for converting paths to strips.
    pub(crate) strip_generator: StripGenerator,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Cache for rasterized glyphs to improve text rendering performance.
    pub(crate) glyph_caches: Option<GlyphCaches>,
    /// Dependency graph for managing layer rendering order and filter effects.
    pub(crate) render_graph: RenderGraph,
    /// Fast path buffer for simple scenes without layers.
    ///
    /// Strips are accumulated here instead of going through coarse rasterization.
    /// The buffer is retained across frames to avoid reallocations.
    pub(crate) fast_path: FastPathBuffer,
    /// Whether the fast path is active for the current scene.
    ///
    /// Set to `true` on reset. Set to `false` once `push_layer` is called,
    /// disabling the fast path for the rest of the scene.
    pub(crate) fast_path_active: bool,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    const DEFAULT_BLEND_MODE: BlendMode = BlendMode::new(Mix::Normal, Compose::SrcOver);

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        let render_state = Self::default_render_state();
        let render_graph = RenderGraph::new();
        Self {
            render_hints: settings.render_hints,
            width,
            height,
            wide: Wide::<MODE_HYBRID>::new(width, height),
            clip_context: ClipContext::new(),
            aliasing_threshold: None,
            paint: render_state.paint,
            paint_transform: render_state.paint_transform,
            encoded_paints: vec![],
            paint_visible: true,
            stroke: render_state.stroke,
            strip_generator: StripGenerator::new(width, height, settings.level),
            strip_storage: RefCell::new(StripStorage::default()),
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
            glyph_caches: Some(GlyphCaches::default()),
            render_graph,
            fast_path: FastPathBuffer::default(),
            fast_path_active: true,
        }
    }

    /// Create default rendering state.
    fn default_render_state() -> RenderState {
        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
        let paint_transform = Affine::IDENTITY;
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        RenderState {
            transform,
            fill_rule,
            paint,
            paint_transform,
            stroke,
            blend_mode: Self::DEFAULT_BLEND_MODE,
        }
    }

    /// Encode the current paint into a `Paint` that can be used for rendering.
    ///
    /// For solid colors, this is a simple conversion. For gradients and images,
    /// this encodes the paint data into the `encoded_paints` buffer and returns
    /// a `Paint` that references that data. The combined transform (geometry + paint)
    /// is applied during encoding.
    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => g.encode_into(
                &mut self.encoded_paints,
                self.transform * self.paint_transform,
            ),
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints,
                self.transform * self.paint_transform,
            ),
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        let paint = self.encode_current_paint();
        self.fill_path_with(
            path,
            self.transform,
            self.fill_rule,
            paint,
            self.aliasing_threshold,
        );
    }

    /// Build strips for a filled path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a path
    /// and submits them to the coarse rasterizer. The path is first converted
    /// to strips by the strip generator, then the strips are processed by the
    /// wide coarse rasterizer to generate binned draw commands.
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let strip_storage = &mut self.strip_storage.borrow_mut();
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            strip_storage,
            self.clip_context.get(),
        );
        if self.fast_path_active {
            let strip_start = self.fast_path.strips.len();
            self.fast_path
                .strips
                .extend_from_slice(&strip_storage.strips);
            let strip_end = self.fast_path.strips.len();
            self.fast_path
                .commands
                .push(FastPathCommand::Path(BufferedPath {
                    strip_start,
                    strip_end,
                    paint,
                }));
        } else {
            self.wide.generate(
                &strip_storage.strips,
                paint,
                self.blend_mode,
                0,
                None,
                &self.encoded_paints,
            );
        }
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            self.fill_rule,
            self.transform,
            self.aliasing_threshold,
        );
    }

    /// Pop a clip path from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clip paths before finishing the rendering operation.
    pub fn pop_clip_path(&mut self) {
        self.clip_context.pop_clip();
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        let paint = self.encode_current_paint();
        self.stroke_path_with(path, self.transform, paint, self.aliasing_threshold);
    }

    /// Build strips for a stroked path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a stroked path
    /// and submits them to the coarse rasterizer. The path is first stroked and
    /// converted to strips by the strip generator, then the strips are processed by
    /// the wide coarse rasterizer to generate binned draw commands.
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let strip_storage = &mut self.strip_storage.borrow_mut();

        self.strip_generator.generate_stroked_path(
            path,
            &self.stroke,
            transform,
            aliasing_threshold,
            strip_storage,
            self.clip_context.get(),
        );

        if self.fast_path_active {
            let strip_start = self.fast_path.strips.len();
            self.fast_path
                .strips
                .extend_from_slice(&strip_storage.strips);
            let strip_end = self.fast_path.strips.len();
            self.fast_path
                .commands
                .push(FastPathCommand::Path(BufferedPath {
                    strip_start,
                    strip_end,
                    paint,
                }));
        } else {
            self.wide.generate(
                &strip_storage.strips,
                paint,
                self.blend_mode,
                0,
                None,
                &self.encoded_paints,
            );
        }
    }

    /// Set the aliasing threshold.
    ///
    /// If set to `None` (which is the recommended option in nearly all cases),
    /// anti-aliasing will be applied.
    ///
    /// If instead set to some value, then a pixel will be fully painted if
    /// the coverage is bigger than the threshold (between 0 and 255), otherwise
    /// it will not be painted at all.
    ///
    /// Note that there is no performance benefit to disabling anti-aliasing and
    /// this functionality is simply provided for compatibility.
    pub fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.aliasing_threshold = aliasing_threshold;
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        if self.try_blit_rect(rect) {
            return;
        }
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Attempt the fast-path blit rect pipeline. Returns `true` if the rect was
    /// handled by the blit pipeline, `false` if it should fall through to the
    /// strip pipeline.
    #[inline(always)]
    fn try_blit_rect(&mut self, rect: &Rect) -> bool {
        if !self.render_hints.blit_rect_pipeline_enabled() {
            return false;
        }

        if !self.fast_path_active {
            return false;
        }

        if !self.paint_visible {
            return true;
        }

        if self.wide.has_layers() {
            return false;
        }

        if self.clip_context.get().is_some() {
            return false;
        }

        let image_id = match &self.paint {
            PaintType::Image(img) => {
                if img.sampler.x_extend != Extend::Pad
                    || img.sampler.y_extend != Extend::Pad
                    || img.sampler.alpha != 1.0
                {
                    return false;
                }
                match &img.image {
                    ImageSource::OpaqueId(id) => *id,
                    _ => return false,
                }
            }
            _ => return false,
        };

        if self.blend_mode != Self::DEFAULT_BLEND_MODE {
            return false;
        }

        let geo_coeffs = self.transform.as_coeffs();
        if geo_coeffs[0] * geo_coeffs[3] - geo_coeffs[1] * geo_coeffs[2] <= 0.0 {
            return false;
        }

        let paint_coeffs = self.paint_transform.as_coeffs();
        if (paint_coeffs[1] as f32).abs() > f32::EPSILON
            || (paint_coeffs[2] as f32).abs() > f32::EPSILON
        {
            return false;
        }
        if ((paint_coeffs[0] - 1.0) as f32).abs() > f32::EPSILON
            || ((paint_coeffs[3] - 1.0) as f32).abs() > f32::EPSILON
        {
            return false;
        }

        let (ptx, pty) = (paint_coeffs[4], paint_coeffs[5]);
        let img_origin_xy = [(rect.x0 - ptx) as f32, (rect.y0 - pty) as f32];

        if img_origin_xy[0] < 0.0 || img_origin_xy[1] < 0.0 {
            return false;
        }

        let rect_wh = [
            pixel_snap(rect.x1 - rect.x0).max(0.0) as u16,
            pixel_snap(rect.y1 - rect.y0).max(0.0) as u16,
        ];

        if rect_wh[0] == 0 || rect_wh[1] == 0 {
            return true;
        }

        let [a, b, c, d, tx, ty] = geo_coeffs.map(|x| x as f32);
        let w = rect_wh[0] as f32;
        let h = rect_wh[1] as f32;

        let col0 = [a * w, b * w];
        let col1 = [c * h, d * h];

        let cx = (rect.x0 + rect.x1) as f32 * 0.5;
        let cy = (rect.y0 + rect.y1) as f32 * 0.5;
        let center_xy = [a * cx + c * cy + tx, b * cx + d * cy + ty];

        self.fast_path
            .commands
            .push(FastPathCommand::Blit(BlitRect {
                center: center_xy,
                col0,
                col1,
                rect_wh,
                image_id,
                img_origin: img_origin_xy,
            }));

        true
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Flush the fast path buffer through the normal coarse rasterization pipeline.
    ///
    /// This is called when `push_layer` is invoked, retroactively processing all buffered
    /// strips through `Wide::generate`. After flushing, `fast_path_active` is set to `false`,
    /// disabling the fast path for the rest of the scene.
    fn flush_fast_path(&mut self) {
        if !self.fast_path_active {
            return;
        }
        for i in 0..self.fast_path.commands.len() {
            match &self.fast_path.commands[i] {
                FastPathCommand::Path(buffered) => {
                    let start = buffered.strip_start;
                    let end = buffered.strip_end;
                    let paint = buffered.paint.clone();
                    self.wide.generate(
                        &self.fast_path.strips[start..end],
                        paint,
                        BlendMode::default(),
                        0,
                        None,
                        &self.encoded_paints,
                    );
                }
                FastPathCommand::Blit(_blit) => {
                    // Blit rects fall back to `fill_rect` through the normal pipeline
                    // when the fast path is flushed. Since we don't have the original
                    // rect, we skip them here — the caller should not have mixed blits
                    // with layers (the blit conditions check for no layers).
                }
            }
        }
        self.fast_path_active = false;
    }

    /// Push a new layer with the given properties.
    ///
    /// Only `clip_path` is supported for now.
    // TODO: Implement filter integration.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        let blend_mode_val = blend_mode.unwrap_or(Self::DEFAULT_BLEND_MODE);
        if self.render_hints.blit_rect_pipeline_enabled() {
            assert!(
                blend_mode_val == Self::DEFAULT_BLEND_MODE,
                "blit rect pipeline only supports default blending"
            );
        }

        self.flush_fast_path();

        if filter.is_some() {
            unimplemented!("Filter effects are not yet supported in vello_hybrid");
        }

        let mut strip_storage = self.strip_storage.borrow_mut();

        let clip = if let Some(c) = clip_path {
            self.strip_generator.generate_filled_path(
                c,
                self.fill_rule,
                self.transform,
                self.aliasing_threshold,
                &mut strip_storage,
                self.clip_context.get(),
            );

            Some(strip_storage.strips.as_slice())
        } else {
            None
        };

        // Mask is unsupported. Blend is partially supported.
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            0,
            clip,
            blend_mode_val,
            None,
            opacity.unwrap_or(1.),
            None,
            self.transform,
            &mut self.render_graph,
            0,
        );
    }

    /// Push a new clip layer.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_path`.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None, None);
    }

    /// Push a new blend layer.
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }

    /// Push a new opacity layer.
    pub fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(None, None, Some(opacity), None, None);
    }

    /// Push a new mask layer.
    ///
    /// Note that masks are not yet supported in `vello_hybrid`.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a new filter layer.
    ///
    /// Note that filters are currently ignored in `vello_hybrid`.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        self.wide.pop_layer(&mut self.render_graph);
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        if self.render_hints.blit_rect_pipeline_enabled() {
            assert!(
                blend_mode == Self::DEFAULT_BLEND_MODE,
                "blit rect pipeline only supports default blending"
            );
        }
        self.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the paint for subsequent rendering operations.
    // TODO: This API is not final. Supporting images from a pixmap is explicitly out of scope.
    //       Instead images should be passed via a backend-agnostic opaque id, and be hydrated at
    //       render time into a texture usable by the renderer backend.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
        self.paint_visible = match &self.paint {
            PaintType::Solid(color) => color.components[3] != 0.0,
            _ => true,
        };
    }

    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.paint_transform = Affine::IDENTITY;
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Apply filter to the current paint (affects next drawn element)
    pub fn set_filter_effect(&mut self, _filter: Filter) {
        unimplemented!("Filter effects integration with Scene")
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        unimplemented!("Filter effects integration with Scene")
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator.reset();
        self.clip_context.reset();
        self.strip_storage.borrow_mut().clear();
        self.encoded_paints.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.paint_transform = render_state.paint_transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;

        self.glyph_caches.as_mut().unwrap().maintain();
        self.fast_path.clear();
        self.fast_path_active = true;
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }
}

impl GlyphRenderer for Scene {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                let paint = self.encode_current_paint();
                self.fill_path_with(
                    glyph.path,
                    prepared_glyph.transform,
                    Fill::NonZero,
                    paint,
                    self.aliasing_threshold,
                );
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn stroke_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                let paint = self.encode_current_paint();
                self.stroke_path_with(
                    glyph.path,
                    prepared_glyph.transform,
                    paint,
                    self.aliasing_threshold,
                );
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn take_glyph_caches(&mut self) -> GlyphCaches {
        self.glyph_caches.take().unwrap_or_default()
    }

    fn restore_glyph_caches(&mut self, cache: GlyphCaches) {
        self.glyph_caches = Some(cache);
    }
}

impl Recordable for Scene {
    fn record<F>(&mut self, recording: &mut Recording, f: F)
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recorder = Recorder::new(recording, self.transform, self.take_glyph_caches());
        f(&mut recorder);
        self.glyph_caches = Some(recorder.take_glyph_caches());
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        let buffers = recording.take_cached_strips();
        let (strip_storage, strip_start_indices) =
            self.generate_strips_from_commands(recording.commands(), buffers);
        recording.set_cached_strips(strip_storage, strip_start_indices);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        let (cached_strips, cached_alphas) = recording.get_cached_strips();
        let adjusted_strips = self.prepare_cached_strips(cached_strips, cached_alphas);

        // Use pre-calculated strip start indices from when we generated the cache
        let strip_start_indices = recording.get_strip_start_indices();
        let mut range_index = 0;

        // Replay commands in order, using cached strips for geometry
        for command in recording.commands() {
            match command {
                RenderCommand::FillPath(_)
                | RenderCommand::StrokePath(_)
                | RenderCommand::FillRect(_)
                | RenderCommand::StrokeRect(_)
                | RenderCommand::FillOutlineGlyph(_)
                | RenderCommand::StrokeOutlineGlyph(_) => {
                    self.process_geometry_command(
                        strip_start_indices,
                        range_index,
                        &adjusted_strips,
                    );
                    range_index += 1;
                }
                RenderCommand::SetPaint(paint) => {
                    self.set_paint(paint.clone());
                }
                RenderCommand::SetPaintTransform(transform) => {
                    self.set_paint_transform(*transform);
                }
                RenderCommand::ResetPaintTransform => {
                    self.reset_paint_transform();
                }
                RenderCommand::SetTransform(transform) => {
                    self.set_transform(*transform);
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.set_fill_rule(*fill_rule);
                }
                RenderCommand::SetStroke(stroke) => {
                    self.set_stroke(stroke.clone());
                }
                RenderCommand::SetFilterEffect(filter) => {
                    self.set_filter_effect(filter.clone());
                }
                RenderCommand::ResetFilterEffect => {
                    self.reset_filter_effect();
                }
                RenderCommand::PushLayer(PushLayerCommand {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    filter,
                }) => {
                    self.push_layer(
                        clip_path.as_ref(),
                        *blend_mode,
                        *opacity,
                        mask.clone(),
                        filter.clone(),
                    );
                }
                RenderCommand::PopLayer => {
                    self.pop_layer();
                }
            }
        }
    }
}

/// Recording management implementation.
impl Scene {
    /// Generate strips from strip commands and capture ranges.
    ///
    /// Returns:
    /// - `collected_strips`: The generated strips.
    /// - `collected_alphas`: The generated alphas.
    /// - `strip_start_indices`: The start indices of strips for each geometry command.
    fn generate_strips_from_commands(
        &mut self,
        commands: &[RenderCommand],
        buffers: (StripStorage, Vec<usize>),
    ) -> (StripStorage, Vec<usize>) {
        let (mut strip_storage, mut strip_start_indices) = buffers;
        strip_storage.clear();
        strip_storage.set_generation_mode(GenerationMode::Append);
        strip_start_indices.clear();

        let saved_state = self.take_current_state();

        for command in commands {
            let start_index = strip_storage.strips.len();

            match command {
                RenderCommand::FillPath(path) => {
                    self.strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokePath(path) => {
                    self.strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillRect(rect) => {
                    self.strip_generator.generate_filled_path(
                        rect.to_path(DEFAULT_TOLERANCE),
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeRect(rect) => {
                    self.strip_generator.generate_stroked_path(
                        rect.to_path(DEFAULT_TOLERANCE),
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillOutlineGlyph((path, glyph_transform)) => {
                    self.strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeOutlineGlyph((path, glyph_transform)) => {
                    self.strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::SetTransform(transform) => {
                    self.transform = *transform;
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.fill_rule = *fill_rule;
                }
                RenderCommand::SetStroke(stroke) => {
                    self.stroke = stroke.clone();
                }

                _ => {}
            }
        }

        self.restore_state(saved_state);

        (strip_storage, strip_start_indices)
    }

    fn process_geometry_command(
        &mut self,
        strip_start_indices: &[usize],
        range_index: usize,
        adjusted_strips: &[Strip],
    ) {
        assert!(
            range_index < strip_start_indices.len(),
            "Strip range index out of bounds: range_index={}, strip_start_indices.len()={}",
            range_index,
            strip_start_indices.len()
        );
        let start = strip_start_indices[range_index];
        let end = strip_start_indices
            .get(range_index + 1)
            .copied()
            .unwrap_or(adjusted_strips.len());
        let count = end - start;
        if count == 0 {
            // There are no strips to generate.
            return;
        }
        assert!(
            start < adjusted_strips.len() && count > 0,
            "Invalid strip range: start={start}, end={end}, count={count}"
        );
        let paint = self.encode_current_paint();
        let strips = &adjusted_strips[start..end];
        if self.fast_path_active {
            let strip_start = self.fast_path.strips.len();
            self.fast_path.strips.extend_from_slice(strips);
            let strip_end = self.fast_path.strips.len();
            self.fast_path
                .commands
                .push(FastPathCommand::Path(BufferedPath {
                    strip_start,
                    strip_end,
                    paint,
                }));
        } else {
            self.wide.generate(
                strips,
                paint,
                self.blend_mode,
                0,
                None,
                &self.encoded_paints,
            );
        }
    }

    /// Prepare cached strips for rendering by adjusting alpha indices and extending alpha buffer.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Alphas length conversion is safe in this case"
    )]
    fn prepare_cached_strips(
        &mut self,
        cached_strips: &[Strip],
        cached_alphas: &[u8],
    ) -> Vec<Strip> {
        let mut strip_storage = self.strip_storage.borrow_mut();
        // Calculate offset for alpha indices based on current buffer size.
        let alpha_offset = strip_storage.alphas.len() as u32;
        // Extend current alpha buffer with cached alphas.
        strip_storage.alphas.extend(cached_alphas);
        // Create adjusted strips with corrected alpha indices
        cached_strips
            .iter()
            .map(move |strip| {
                let mut adjusted_strip = *strip;
                adjusted_strip.set_alpha_idx(adjusted_strip.alpha_idx() + alpha_offset);
                adjusted_strip
            })
            .collect()
    }

    /// Save current rendering state.
    fn take_current_state(&mut self) -> RenderState {
        RenderState {
            paint: self.paint.clone(),
            paint_transform: self.paint_transform,
            transform: self.transform,
            fill_rule: self.fill_rule,
            blend_mode: self.blend_mode,
            stroke: core::mem::take(&mut self.stroke),
        }
    }

    /// Restore rendering state.
    fn restore_state(&mut self, state: RenderState) {
        self.paint = state.paint;
        self.paint_transform = state.paint_transform;
        self.stroke = state.stroke;
        self.transform = state.transform;
        self.fill_rule = state.fill_rule;
        self.blend_mode = state.blend_mode;
    }
}

/// Round to the nearest integer for pixel snapping.
///
/// On `wasm32-unknown-unknown`, [`f64::round`] compiles to a software `libm`
/// call. `(x + 0.5).floor()` maps directly to the native `f64.floor`
/// instruction while giving identical results for non-negative values.
#[inline(always)]
fn pixel_snap(x: f64) -> f64 {
    (x + 0.5).floor()
}
