// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use alloc::vec;
use alloc::vec::Vec;
use vello_common::coarse::Wide;
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, Line};
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::recording::{PushLayerCommand, Recordable, Recording, RenderCommand};
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

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
    pub(crate) strip_buf: Vec<Strip>,
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
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: PaintType,
    pub(crate) paint_transform: Affine,
    pub(crate) anti_alias: bool,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    paint_visible: bool,
    level: Level,
    flatten_ctx: FlattenCtx,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let render_state = Self::default_render_state();
        Self {
            width,
            height,
            wide: Wide::new(width, height),
            anti_alias: true,
            alphas: vec![],
            level: Level::fallback(),
            line_buf: vec![],
            tiles: Tiles::new(),
            strip_buf: vec![],
            paint: render_state.paint,
            paint_transform: render_state.paint_transform,
            encoded_paints: vec![],
            paint_visible: true,
            stroke: render_state.stroke,
            flatten_ctx: FlattenCtx::default(),
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
            strip_buf: vec![],
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
        flatten::fill(
            self.level,
            path,
            self.transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        let paint = self.encode_current_paint();
        self.render_path(self.fill_rule, paint);
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }
        flatten::stroke(
            self.level,
            path,
            &self.stroke,
            self.transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        let paint = self.encode_current_paint();
        self.render_path(Fill::NonZero, paint);
    }

    /// Set whether to enable anti-aliasing.
    pub fn set_anti_aliasing(&mut self, value: bool) {
        self.anti_alias = value;
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
            flatten::fill(
                self.level,
                c,
                self.transform,
                &mut self.line_buf,
                &mut self.flatten_ctx,
            );
            self.make_strips(self.fill_rule);
            Some((self.strip_buf.as_slice(), self.fill_rule))
        } else {
            None
        };

        // Blend mode, opacity, and mask are not supported yet.
        if blend_mode.is_some() {
            unimplemented!()
        }
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            clip,
            BlendMode::new(Mix::Normal, Compose::SrcOver),
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
        self.alphas.clear();
        self.line_buf.clear();
        self.tiles.reset();
        self.strip_buf.clear();
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

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.make_strips(fill_rule);
        self.wide.generate(&self.strip_buf, fill_rule, paint, 0);
    }

    fn make_strips(&mut self, fill_rule: Fill) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();
        strip::render(
            Level::fallback(),
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            self.anti_alias,
            &self.line_buf,
        );
    }
}

impl GlyphRenderer for Scene {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::fill(
                    self.level,
                    glyph.path,
                    prepared_glyph.transform,
                    &mut self.line_buf,
                    &mut self.flatten_ctx,
                );
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn stroke_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                flatten::stroke(
                    self.level,
                    glyph.path,
                    &self.stroke,
                    prepared_glyph.transform,
                    &mut self.line_buf,
                    &mut self.flatten_ctx,
                );
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
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

        let collected_alphas = core::mem::take(&mut self.alphas);
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
        let alpha_offset = self.alphas.len() as u32;
        // Extend current alpha buffer with cached alphas.
        self.alphas.extend_from_slice(cached_alphas);
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
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips_into_buffers(self.fill_rule, strips);
    }

    /// Generate strips for a stroked path.
    fn generate_stroke_strips(
        &mut self,
        path: &BezPath,
        strips: &mut Vec<Strip>,
        transform: Affine,
    ) {
        flatten::stroke(
            self.level,
            path,
            &self.stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips_into_buffers(Fill::NonZero, strips);
    }

    /// Generate strips and append to provided buffers.
    fn make_strips_into_buffers(&mut self, fill_rule: Fill, strips: &mut Vec<Strip>) {
        self.make_strips(fill_rule);
        strips.append(&mut self.strip_buf);
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
            strip_buf: core::mem::take(&mut self.strip_buf),
            alphas: core::mem::replace(&mut self.alphas, cached_alphas),
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
        self.strip_buf = state.strip_buf;
        self.alphas = state.alphas;
    }
}
