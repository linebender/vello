// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recording API for caching sparse strips

#[cfg(feature = "text")]
use crate::glyph::{GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::mask::Mask;
use crate::paint::PaintType;
#[cfg(feature = "text")]
use crate::peniko::Font;
use crate::peniko::{BlendMode, Fill};
use crate::strip::Strip;
use alloc::vec::Vec;

/// Cached sparse strip data.
#[derive(Debug, Default)]
pub struct CachedStrips {
    /// The cached sparse strips.
    strips: Vec<Strip>,
    /// The alpha buffer data.
    alphas: Vec<u8>,
    /// Strip start indices for each geometry command.
    strip_start_indices: Vec<usize>,
}

impl CachedStrips {
    /// Create a new cached strips instance.
    pub fn new(strips: Vec<Strip>, alphas: Vec<u8>, strip_start_indices: Vec<usize>) -> Self {
        Self {
            strips,
            alphas,
            strip_start_indices,
        }
    }

    /// Clear the contents.
    pub fn clear(&mut self) {
        self.strips.clear();
        self.alphas.clear();
        self.strip_start_indices.clear();
    }

    /// Check if this cached strips is empty.
    pub fn is_empty(&self) -> bool {
        self.strips.is_empty() && self.alphas.is_empty() && self.strip_start_indices.is_empty()
    }

    /// Get the number of strips.
    pub fn strip_count(&self) -> usize {
        self.strips.len()
    }

    /// Get the number of alpha bytes.
    pub fn alpha_count(&self) -> usize {
        self.alphas.len()
    }

    /// Get strips as slice.
    pub fn strips(&self) -> &[Strip] {
        &self.strips
    }

    /// Get alphas as slice
    pub fn alphas(&self) -> &[u8] {
        &self.alphas
    }

    /// Get strip start indices.
    pub fn strip_start_indices(&self) -> &[usize] {
        &self.strip_start_indices
    }

    /// Takes ownership of all buffers.
    pub fn take(&mut self) -> (Vec<Strip>, Vec<u8>, Vec<usize>) {
        let strips = core::mem::take(&mut self.strips);
        let alphas = core::mem::take(&mut self.alphas);
        let strip_start_indices = core::mem::take(&mut self.strip_start_indices);
        (strips, alphas, strip_start_indices)
    }
}

/// A recording of rendering commands that can cache generated strips.
#[derive(Debug)]
pub struct Recording {
    /// Recorded commands.
    commands: Vec<RenderCommand>,
    /// Cached sparse strips.
    cached_strips: CachedStrips,
    /// Last recorded transform.
    transform: Affine,
}

/// Command for pushing a new layer.
#[derive(Debug, Clone)]
pub struct PushLayerCommand {
    /// Clip path.
    pub clip_path: Option<BezPath>,
    /// Blend mode.
    pub blend_mode: Option<BlendMode>,
    /// Opacity.
    pub opacity: Option<f32>,
    /// Mask.
    pub mask: Option<Mask>,
}

/// Individual rendering commands that can be recorded.
#[derive(Debug)]
pub enum RenderCommand {
    /// Fill a path.
    FillPath(BezPath),
    /// Stroke a path.
    StrokePath(BezPath),
    /// Fill a rectangle.
    FillRect(Rect),
    /// Stroke a rectangle.
    StrokeRect(Rect),
    /// Set the current transform.
    SetTransform(Affine),
    /// Set the fill rule.
    SetFillRule(Fill),
    /// Set the stroke parameters.
    SetStroke(Stroke),
    /// Push a new layer with optional clipping and effects.
    PushLayer(PushLayerCommand),
    /// Pop the current layer.
    PopLayer,
    /// Set the current paint.
    SetPaint(PaintType),
    /// Set the paint transform.
    SetPaintTransform(Affine),
    /// Reset the paint transform.
    ResetPaintTransform,
    /// Render a fill outline glyph.
    #[cfg(feature = "text")]
    FillOutlineGlyph((BezPath, Affine)),
    /// Render a stroke outline glyph.
    #[cfg(feature = "text")]
    StrokeOutlineGlyph((BezPath, Affine)),
}

impl Recording {
    /// Create a new empty recording.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            cached_strips: CachedStrips::default(),
            transform: Affine::IDENTITY,
        }
    }

    /// Set the transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Get the current transform.
    pub fn transform(&self) -> Affine {
        self.transform
    }

    /// Get commands as a slice.
    pub fn commands(&self) -> &[RenderCommand] {
        &self.commands
    }

    /// Get the number of commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Check if recording has cached strips.
    pub fn has_cached_strips(&self) -> bool {
        !self.cached_strips.is_empty()
    }

    /// Get the number of cached strips.
    pub fn strip_count(&self) -> usize {
        self.cached_strips.strip_count()
    }

    /// Get the number of cached alpha bytes.
    pub fn alpha_count(&self) -> usize {
        self.cached_strips.alpha_count()
    }

    /// Get cached strips.
    pub fn get_cached_strips(&self) -> (&[Strip], &[u8]) {
        (self.cached_strips.strips(), self.cached_strips.alphas())
    }

    /// Takes cached strip buffers.
    pub fn take_cached_strips(&mut self) -> (Vec<Strip>, Vec<u8>, Vec<usize>) {
        self.cached_strips.take()
    }

    /// Get strip start indices.
    pub fn get_strip_start_indices(&self) -> &[usize] {
        self.cached_strips.strip_start_indices()
    }

    /// Clear the recording contents.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.cached_strips.clear();
        self.transform = Affine::IDENTITY;
    }

    /// Add a command to the recording.
    pub(crate) fn add_command(&mut self, command: RenderCommand) {
        self.commands.push(command);
    }

    /// Set cached strips.
    pub fn set_cached_strips(
        &mut self,
        strips: Vec<Strip>,
        alphas: Vec<u8>,
        strip_start_indices: Vec<usize>,
    ) {
        self.cached_strips = CachedStrips::new(strips, alphas, strip_start_indices);
    }
}

impl Default for Recording {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for rendering contexts that support recording and replaying operations.
///
/// # State Modification During Replay
///
/// **Important:** When replaying recordings using methods like `execute_recording()`,
/// the renderer's state (transform, paint, fill rule, stroke settings, etc.) will be
/// modified to match the state changes captured in the recording. The renderer will
/// be left in the final state after all commands have been executed.
///
/// # Multithreading Limitation
///
/// **Note:** Recording and replay functionality is not currently implemented for
/// `vello_cpu` when multithreading is enabled. This limitation only affects
/// `vello_cpu` in multithreaded mode; single-threaded `vello_cpu` and `vello_hybrid`
/// work correctly with recordings.
///
/// # Usage Pattern
///
/// A consumer needs to do the following to render:
/// ```ignore
/// let mut recording = Recording::new();
/// scene.record(&mut recording, |ctx| { ... });
/// scene.prepare_recording(&mut recording);
/// scene.execute_recording(&recording);
/// ```
///
/// Or to prepare for later rendering:
/// ```ignore
/// let mut recording = Recording::new();
/// scene.record(&mut recording, |ctx| { ... });
/// scene.prepare_recording(&mut recording);
///
/// // sometime later
/// scene.execute_recording(&recording);
/// ```
pub trait Recordable {
    /// Record rendering commands into a recording.
    ///
    /// This method allows you to capture a sequence of rendering operations
    /// in a `Recording` that can be cached and replayed later.
    ///
    /// # Example
    /// ```ignore
    /// let mut recording = Recording::new();
    /// scene.record(&mut recording, |ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ///     ctx.set_paint(Color::RED);
    ///     ctx.stroke_path(&some_path);
    /// });
    /// ```
    fn record<F>(&mut self, recording: &mut Recording, f: F)
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recorder = Recorder::new(recording);
        f(&mut recorder);
    }

    /// Generate sparse strips for a recording.
    ///
    /// This method processes the recorded commands and generates cached sparse strips
    /// without executing the rendering. This allows you to pre-generate strips for
    /// better control over when the expensive computation happens.
    ///
    /// # Example
    /// ```ignore
    /// let mut recording = Recording::new();
    /// scene.record(&mut recording, |ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    /// });
    ///
    /// // Generate strips explicitly
    /// scene.prepare_recording(&mut recording);
    /// ```
    fn prepare_recording(&mut self, recording: &mut Recording);

    /// Execute a recording directly without preparation.
    ///
    /// This method executes the rendering commands from a recording, using any
    /// cached sparse strips that have been previously generated. If the recording
    /// has not been prepared (no cached strips), this will result in empty rendering.
    ///
    /// Use this method when you have a recording that has already been prepared
    /// via `prepare_recording()`, or when you want to execute commands immediately
    /// without explicit preparation.
    ///
    /// # Example
    /// ```ignore
    /// let mut recording = Recording::new();
    /// scene.record(&mut recording, |ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    /// });
    ///
    /// // Prepare strips first
    /// scene.prepare_recording(&mut recording);
    ///
    /// // Then execute with cached strips
    /// scene.execute_recording(&recording);
    /// ```
    fn execute_recording(&mut self, recording: &Recording);
}

/// Recorder context that captures commands.
#[derive(Debug)]
pub struct Recorder<'a> {
    /// The recording to capture commands into.
    recording: &'a mut Recording,
}

impl<'a> Recorder<'a> {
    /// Create a new recorder for the given recording.
    pub fn new(recording: &'a mut Recording) -> Self {
        Self { recording }
    }

    /// Fill a path with current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        self.recording
            .add_command(RenderCommand::FillPath(path.clone()));
    }

    /// Stroke a path with current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        self.recording
            .add_command(RenderCommand::StrokePath(path.clone()));
    }

    /// Fill a rectangle with current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.recording.add_command(RenderCommand::FillRect(*rect));
    }

    /// Stroke a rectangle with current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.recording.add_command(RenderCommand::StrokeRect(*rect));
    }

    /// Set the transform for subsequent operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.recording
            .add_command(RenderCommand::SetTransform(transform));
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.recording
            .add_command(RenderCommand::SetFillRule(fill_rule));
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.recording.add_command(RenderCommand::SetStroke(stroke));
    }

    /// Set the paint for subsequent rendering operations.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.recording
            .add_command(RenderCommand::SetPaint(paint.into()));
    }

    /// Set the current paint transform.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.recording
            .add_command(RenderCommand::SetPaintTransform(paint_transform));
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.recording
            .add_command(RenderCommand::ResetPaintTransform);
    }

    /// Push a new layer with the given properties.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.recording
            .add_command(RenderCommand::PushLayer(PushLayerCommand {
                clip_path: clip_path.cloned(),
                blend_mode,
                opacity,
                mask,
            }));
    }

    /// Push a new clip layer.
    pub fn push_clip_layer(&mut self, clip_path: &BezPath) {
        self.push_layer(Some(clip_path), None, None, None);
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        self.recording.add_command(RenderCommand::PopLayer);
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.recording.transform, self)
    }
}

#[cfg(feature = "text")]
impl<'a> GlyphRenderer for Recorder<'a> {
    fn fill_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph.glyph_type {
            GlyphType::Outline(outline_glyph) => {
                if !outline_glyph.path.is_empty() {
                    self.recording.add_command(RenderCommand::FillOutlineGlyph((
                        outline_glyph.path.clone(),
                        glyph.transform,
                    )));
                }
            }

            _ => {
                unimplemented!("Recording glyphs of type {:?}", glyph.glyph_type);
            }
        }
    }

    fn stroke_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph.glyph_type {
            GlyphType::Outline(outline_glyph) => {
                if !outline_glyph.path.is_empty() {
                    self.recording
                        .add_command(RenderCommand::StrokeOutlineGlyph((
                            outline_glyph.path.clone(),
                            glyph.transform,
                        )));
                }
            }
            _ => {
                unimplemented!("Recording glyphs of type {:?}", glyph.glyph_type);
            }
        }
    }

    fn restore_hinting_cache(&mut self, cache: crate::glyph::HintCache) {}

    fn take_hinting_cache(&mut self) -> crate::glyph::HintCache {
        Default::default()
    }

    fn take_glyph_cache(&mut self) -> crate::glyph::GlyphCache {
        Default::default()
    }

    fn restore_glyph_cache(&mut self, cache: crate::glyph::GlyphCache) {}
}
