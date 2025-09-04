// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use alloc::vec;
use alloc::vec::Vec;
use vello_common::coarse::{MODE_HYBRID, Wide};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::recording::{PushLayerCommand, Recordable, Recording, RenderCommand};
use vello_common::strip::Strip;
use vello_common::strip_generator::StripGenerator;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render state which contains the style properties for path rendering and
/// the current transform.
#[derive(Debug)]
struct RenderState {
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
    pub(crate) alphas: Vec<u8>,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide<MODE_HYBRID>,
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) aliasing_threshold: Option<u8>,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    paint_visible: bool,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
    pub(crate) strip_generator: StripGenerator,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let render_state = Self::default_render_state();
        Self {
            width,
            height,
            wide: Wide::<MODE_HYBRID>::new(width, height),
            aliasing_threshold: None,
            paint: render_state.paint,
            paint_transform: render_state.paint_transform,
            encoded_paints: vec![],
            paint_visible: true,
            stroke: render_state.stroke,
            strip_generator: StripGenerator::new(
                width,
                height,
                Level::try_detect().unwrap_or(Level::fallback()),
            ),
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
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
        let blend_mode = BlendMode::new(Mix::Normal, Compose::SrcOver);
        RenderState {
            transform,
            fill_rule,
            paint,
            paint_transform,
            stroke,
            blend_mode,
            alphas: vec![],
        }
    }

    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(_) => {
                unimplemented!("Gradient not implemented")
            }
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
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;
        let func = |strips| wide.generate(strips, fill_rule, paint, 0);
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            func,
        );
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
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;
        let func = |strips| wide.generate(strips, Fill::NonZero, paint, 0);
        self.strip_generator.generate_stroked_path(
            path,
            &self.stroke,
            transform,
            aliasing_threshold,
            func,
        );
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
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Push a new layer with the given properties.
    ///
    /// Only `clip_path` is supported for now.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        let clip = if let Some(c) = clip_path {
            let mut strip_buf = &[][..];

            self.strip_generator.generate_filled_path(
                c,
                self.fill_rule,
                self.transform,
                self.aliasing_threshold,
                |strips| strip_buf = strips,
            );

            Some((strip_buf, self.fill_rule))
        } else {
            None
        };

        // Mask is unsupported. Blend is partially supported.
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            clip,
            blend_mode.unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
            None,
            opacity.unwrap_or(1.),
            0,
        );
    }

    /// Push a new clip layer.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None);
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        self.wide.pop_layer();
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
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

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator.reset();
        self.encoded_paints.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.paint_transform = render_state.paint_transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;
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
}

impl Recordable for Scene {
    fn prepare_recording(&mut self, recording: &mut Recording) {
        let buffers = recording.take_cached_strips();
        let (strips, alphas, strip_start_indices) =
            self.generate_strips_from_commands(recording.commands(), buffers);
        recording.set_cached_strips(strips, alphas, strip_start_indices);
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
                        command,
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
                RenderCommand::PushLayer(PushLayerCommand {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                }) => {
                    self.push_layer(clip_path.as_ref(), *blend_mode, *opacity, mask.clone());
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
        buffers: (Vec<Strip>, Vec<u8>, Vec<usize>),
    ) -> (Vec<Strip>, Vec<u8>, Vec<usize>) {
        let (mut collected_strips, mut cached_alphas, mut strip_start_indices) = buffers;
        collected_strips.clear();
        cached_alphas.clear();
        strip_start_indices.clear();

        let saved_state = self.take_current_state(cached_alphas);

        for command in commands {
            let start_index = collected_strips.len();

            match command {
                RenderCommand::FillPath(path) => {
                    self.generate_fill_strips(path, &mut collected_strips, self.transform);
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokePath(path) => {
                    self.generate_stroke_strips(path, &mut collected_strips, self.transform);
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillRect(rect) => {
                    let path = rect.to_path(DEFAULT_TOLERANCE);
                    self.generate_fill_strips(&path, &mut collected_strips, self.transform);
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeRect(rect) => {
                    let path = rect.to_path(DEFAULT_TOLERANCE);
                    self.generate_stroke_strips(&path, &mut collected_strips, self.transform);
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillOutlineGlyph((path, transform)) => {
                    let glyph_transform = self.transform * *transform;
                    self.generate_fill_strips(path, &mut collected_strips, glyph_transform);
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeOutlineGlyph((path, transform)) => {
                    let glyph_transform = self.transform * *transform;
                    self.generate_stroke_strips(path, &mut collected_strips, glyph_transform);
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

        let collected_alphas = self.strip_generator.take_alpha_buf();
        self.restore_state(saved_state);

        (collected_strips, collected_alphas, strip_start_indices)
    }

    fn process_geometry_command(
        &mut self,
        command: &RenderCommand,
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
        assert!(
            start < adjusted_strips.len() && count > 0,
            "Invalid strip range: start={start}, end={end}, count={count}"
        );
        let paint = self.encode_current_paint();
        let fill_rule = match command {
            RenderCommand::FillPath(_) | RenderCommand::FillRect(_) => self.fill_rule,
            RenderCommand::StrokePath(_) | RenderCommand::StrokeRect(_) => Fill::NonZero,
            _ => Fill::NonZero,
        };
        self.wide
            .generate(&adjusted_strips[start..end], fill_rule, paint, 0);
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
        // Calculate offset for alpha indices based on current buffer size.
        let alpha_offset = self.strip_generator.alpha_buf().len() as u32;
        // Extend current alpha buffer with cached alphas.
        self.strip_generator.extend_alpha_buf(cached_alphas);
        // Create adjusted strips with corrected alpha indices
        cached_strips
            .iter()
            .map(move |strip| {
                let mut adjusted_strip = *strip;
                adjusted_strip.alpha_idx += alpha_offset;
                adjusted_strip
            })
            .collect()
    }

    /// Generate strips for a filled path.
    fn generate_fill_strips(&mut self, path: &BezPath, strips: &mut Vec<Strip>, transform: Affine) {
        self.strip_generator.generate_filled_path(
            path,
            self.fill_rule,
            transform,
            self.aliasing_threshold,
            |generated_strips| {
                strips.extend_from_slice(generated_strips);
            },
        );
    }

    /// Generate strips for a stroked path.
    fn generate_stroke_strips(
        &mut self,
        path: &BezPath,
        strips: &mut Vec<Strip>,
        transform: Affine,
    ) {
        self.strip_generator.generate_stroked_path(
            path,
            &self.stroke,
            transform,
            self.aliasing_threshold,
            |generated_strips| {
                strips.extend_from_slice(generated_strips);
            },
        );
    }

    /// Save current rendering state.
    fn take_current_state(&mut self, cached_alphas: Vec<u8>) -> RenderState {
        RenderState {
            paint: self.paint.clone(),
            paint_transform: self.paint_transform,
            transform: self.transform,
            fill_rule: self.fill_rule,
            blend_mode: self.blend_mode,
            stroke: core::mem::take(&mut self.stroke),
            alphas: self.strip_generator.replace_alpha_buf(cached_alphas),
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
        self.strip_generator.set_alpha_buf(state.alphas);
    }
}
