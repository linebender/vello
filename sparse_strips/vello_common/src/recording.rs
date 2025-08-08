// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recording API for caching sparse strips

use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::mask::Mask;
use crate::paint::PaintType;
use crate::peniko::{BlendMode, Fill};
use crate::strip::Strip;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// Cached sparse strip data
#[derive(Debug)]
pub struct CachedStrips {
    /// The cached sparse strips
    pub strips: Box<[Strip]>,
    /// The alpha buffer data
    pub alphas: Box<[u8]>,
}

impl CachedStrips {
    /// Create a new cached strips instance
    pub fn new(strips: Box<[Strip]>, alphas: Box<[u8]>) -> Self {
        Self { strips, alphas }
    }

    /// Check if this cached strips is empty
    pub fn is_empty(&self) -> bool {
        self.strips.is_empty()
    }

    /// Get the number of strips
    pub fn strip_count(&self) -> usize {
        self.strips.len()
    }

    /// Get the number of alpha bytes
    pub fn alpha_count(&self) -> usize {
        self.alphas.len()
    }

    /// Get strips as slice
    pub fn strips(&self) -> &[Strip] {
        &self.strips
    }

    /// Get alphas as slice
    pub fn alphas(&self) -> &[u8] {
        &self.alphas
    }
}

/// A recording of rendering commands that can cache generated strips
#[derive(Debug)]
pub struct Recording {
    /// Recorded commands
    pub commands: Vec<RenderCommand>,
    /// Cached sparse strips
    pub cached_strips: Option<CachedStrips>,
    /// Strip ranges for each geometry command: `(start_index, count)`
    pub strip_ranges: Option<Vec<(usize, usize)>>,
    /// Last recorded transform
    pub transform: Affine,
}

/// Individual rendering commands that can be recorded
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Fill a path
    FillPath(BezPath),
    /// Stroke a path
    StrokePath(BezPath),
    /// Fill a rectangle
    FillRect(Rect),
    /// Stroke a rectangle
    StrokeRect(Rect),
    /// Set the current transform
    SetTransform(Affine),
    /// Set the fill rule
    SetFillRule(Fill),
    /// Set the stroke parameters
    SetStroke(Stroke),
    /// Push a new layer with optional clipping and effects
    PushLayer {
        /// Optional clipping path
        clip_path: Option<BezPath>,
        /// Optional blend mode
        blend_mode: Option<BlendMode>,
        /// Optional opacity
        opacity: Option<f32>,
        /// Optional mask
        mask: Option<Mask>,
    },
    /// Pop the current layer
    PopLayer,
    /// Set the current paint
    SetPaint(PaintType),
    /// Set the paint transform
    SetPaintTransform(Affine),
    /// Reset the paint transform
    ResetPaintTransform,
}

impl Recording {
    /// Create a new empty recording
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            cached_strips: None,
            strip_ranges: None,
            transform: Affine::IDENTITY,
        }
    }

    /// Set the transform
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Check if recording has cached strips
    pub fn has_cached_strips(&self) -> bool {
        self.cached_strips.is_some()
    }

    /// Get the number of cached strips
    pub fn strip_count(&self) -> usize {
        self.cached_strips
            .as_ref()
            .map_or(0, |cached| cached.strip_count())
    }

    /// Get the number of cached alpha bytes
    pub fn alpha_count(&self) -> usize {
        self.cached_strips
            .as_ref()
            .map_or(0, |cached| cached.alpha_count())
    }

    /// Set cached strips
    pub fn set_cached_strips(&mut self, strips: Box<[Strip]>, alphas: Box<[u8]>) {
        self.cached_strips = Some(CachedStrips::new(strips, alphas));
    }

    /// Get cached strips
    pub fn get_cached_strips(&self) -> Option<(&[Strip], &[u8])> {
        self.cached_strips
            .as_ref()
            .map(|cached| (cached.strips(), cached.alphas()))
    }

    /// Set strip ranges
    pub fn set_strip_ranges(&mut self, ranges: Vec<(usize, usize)>) {
        self.strip_ranges = Some(ranges);
    }

    /// Get strip ranges
    pub fn get_strip_ranges(&self) -> &[(usize, usize)] {
        self.strip_ranges
            .as_ref()
            .map_or(&[], |ranges| ranges.as_slice())
    }
}

impl Default for Recording {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for rendering contexts that support recording and replaying operations
pub trait Recordable {
    /// Record rendering commands and return recording
    ///
    /// This method allows you to capture a sequence of rendering operations
    /// in a `Recording` that can be cached and replayed later.
    ///
    /// # Example
    /// ```ignore
    /// let recording = scene.record(|ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ///     ctx.set_paint(Color::RED);
    ///     ctx.stroke_path(&some_path);
    /// });
    /// ```
    fn record<F>(&mut self, f: F) -> Recording
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recording = Recording::new();
        let mut recorder = Recorder::new(&mut recording);
        f(&mut recorder);
        recording
    }

    /// Generate sparse strips for a recording
    ///
    /// This method processes the recorded commands and generates cached sparse strips
    /// without executing the rendering. This allows you to pre-generate strips for
    /// better control over when the expensive computation happens.
    ///
    /// # Example
    /// ```ignore
    /// let mut recording = scene.record(|ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    /// });
    ///
    /// // Generate strips explicitly
    /// scene.prepare_recording(&mut recording);
    ///
    /// // Now render using pre-generated strips
    /// scene.render_recording(&mut recording);
    /// ```
    fn prepare_recording(&mut self, recording: &mut Recording) {
        if !recording.has_cached_strips() {
            let (strips, alphas, strip_ranges) =
                self.generate_strips_from_commands(&recording.commands);
            std::println!("Prepare recording: Alpha count: {}", alphas.len());
            recording.set_cached_strips(strips.into_boxed_slice(), alphas.into_boxed_slice());
            recording.set_strip_ranges(strip_ranges);
        }
    }

    /// Render using a recording (caches strips on first use)
    ///
    /// This method executes a previously recorded sequence of operations.
    /// On first use, it will generate and cache the necessary rendering data.
    /// Subsequent calls will reuse the cached data for better performance.
    fn render_recording(&mut self, recording: &mut Recording) {
        self.prepare_recording(recording);
        self.execute_recording(recording);
    }

    /// Record and prepare strips immediately
    ///
    /// This is a convenience method that combines `record` and `prepare_recording`.
    /// Use this when you want to pre-generate strips without rendering yet.
    ///
    /// # Example
    /// ```ignore
    /// let recording = scene.record_and_prepare(|ctx| {
    ///     ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    /// });
    /// // Strips are now generated and cached
    /// ```
    fn record_and_prepare<F>(&mut self, f: F) -> Recording
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recording = self.record(f);
        self.prepare_recording(&mut recording);
        recording
    }

    /// Record and render immediately
    ///
    /// This is a convenience method that combines `record` and `render_recording`.
    /// Use this when you want to execute commands immediately but also keep
    /// the recording for potential reuse.
    fn record_and_render<F>(&mut self, f: F) -> Recording
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recording = self.record(f);
        self.render_recording(&mut recording);
        recording
    }

    /// Render recording if available, otherwise record and render
    ///
    /// This method provides a convenient way to handle optional recordings.
    /// If a recording is provided, it will be rendered. Otherwise, it will
    /// record and render the provided commands.
    fn render_or_record<F>(&mut self, recording: Option<Recording>, f: F) -> Recording
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        if let Some(mut existing) = recording {
            self.render_recording(&mut existing);
            existing
        } else {
            self.record_and_render(f)
        }
    }

    /// Execute a recording
    ///
    /// This method executes a previously recorded sequence of operations.
    /// It will generate and cache the necessary rendering data if it hasn't been done yet.
    /// Subsequent calls will reuse the cached data for better performance.
    fn execute_recording(&mut self, recording: &Recording);

    /// Generate strips from strip commands and capture ranges
    /// Returns:
    /// - `collected_strips`: The generated strips
    /// - `collected_alphas`: The generated alphas
    /// - `strip_ranges`: The ranges of strips in the generated strips
    fn generate_strips_from_commands(
        &mut self,
        commands: &[RenderCommand],
    ) -> (Vec<Strip>, Vec<u8>, Vec<(usize, usize)>);
}

/// Recorder context that captures commands
#[derive(Debug)]
pub struct Recorder<'a> {
    /// The recording to capture commands into
    recording: &'a mut Recording,
}

impl<'a> Recorder<'a> {
    /// Create a new recorder for the given recording
    pub fn new(recording: &'a mut Recording) -> Self {
        Self { recording }
    }

    /// Fill a path with current paint and fill rule
    pub fn fill_path(&mut self, path: &BezPath) {
        self.recording
            .commands
            .push(RenderCommand::FillPath(path.clone()));
    }

    /// Stroke a path with current paint and stroke settings
    pub fn stroke_path(&mut self, path: &BezPath) {
        self.recording
            .commands
            .push(RenderCommand::StrokePath(path.clone()));
    }

    /// Fill a rectangle with current paint and fill rule
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.recording.commands.push(RenderCommand::FillRect(*rect));
    }

    /// Stroke a rectangle with current paint and stroke settings
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.recording
            .commands
            .push(RenderCommand::StrokeRect(*rect));
    }

    /// Set the transform for subsequent operations
    pub fn set_transform(&mut self, transform: Affine) {
        self.recording
            .commands
            .push(RenderCommand::SetTransform(transform));
    }

    /// Set the fill rule for subsequent fill operations
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.recording
            .commands
            .push(RenderCommand::SetFillRule(fill_rule));
    }

    /// Set the stroke settings for subsequent stroke operations
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.recording
            .commands
            .push(RenderCommand::SetStroke(stroke));
    }

    /// Set the paint for subsequent rendering operations
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.recording
            .commands
            .push(RenderCommand::SetPaint(paint.into()));
    }

    /// Set the current paint transform
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.recording
            .commands
            .push(RenderCommand::SetPaintTransform(paint_transform));
    }

    /// Reset the current paint transform
    pub fn reset_paint_transform(&mut self) {
        self.recording
            .commands
            .push(RenderCommand::ResetPaintTransform);
    }

    /// Push a new layer with the given properties
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.recording.commands.push(RenderCommand::PushLayer {
            clip_path: clip_path.cloned(),
            blend_mode,
            opacity,
            mask,
        });
    }

    /// Push a new clip layer
    pub fn push_clip_layer(&mut self, clip_path: &BezPath) {
        self.push_layer(Some(clip_path), None, None, None);
    }

    /// Pop the last pushed layer
    pub fn pop_layer(&mut self) {
        self.recording.commands.push(RenderCommand::PopLayer);
    }
}
