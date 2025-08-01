// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recording API for caching sparse strips

use alloc::boxed::Box;
use alloc::vec::Vec;
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::PaintType;
use vello_common::peniko::{BlendMode, Fill, Font};
use vello_common::strip::Strip;

/// Cached sparse strip data
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Recording {
    /// Recorded commands
    pub commands: Vec<RenderCommand>,
    /// Cached sparse strips
    pub cached_strips: Option<CachedStrips>,
    /// Strip ranges for each geometry command: (start_index, count)
    strip_ranges: Option<Vec<(usize, usize)>>,
    /// Last recorded transform
    pub transform: Affine,
}

/// Individual rendering commands that can be recorded
#[derive(Debug, Clone)]
pub enum RenderCommand {
    // Geometry commands (affect strip generation)
    FillPath(BezPath),
    StrokePath(BezPath),
    FillRect(Rect),
    StrokeRect(Rect),

    // State commands that affect strip generation
    SetTransform(Affine),
    SetFillRule(Fill),
    SetStroke(Stroke),

    // Layer commands (clips affect strip generation)
    PushLayer {
        clip_path: Option<BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    },
    PopLayer,

    // Paint commands (don't affect strip generation - applied at render time)
    SetPaint(PaintType),
    SetPaintTransform(Affine),
    ResetPaintTransform,
    SetBlendMode(BlendMode),
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

    /// Apply translation to cached strips
    /// This allows reusing cached strips at different positions without re-recording
    /// The translation values will be rounded to integers
    /// TODO: Negative coordinates are not supported, should we regenerate the strips?
    pub fn translate(&mut self, dx: i32, dy: i32) {
        if let Some(ref mut cached_strips) = self.cached_strips {
            for strip in cached_strips.strips.iter_mut() {
                // Skip sentinel values
                if strip.x == u16::MAX || strip.y == u16::MAX {
                    continue;
                }

                let new_x = strip.x as i32 + dx;
                let new_y = strip.y as i32 + dy;

                // Check for overflow/underflow
                if new_x < 0 || new_x >= u16::MAX as i32 || new_y < 0 || new_y >= u16::MAX as i32 {
                    panic!("Translation would cause overflow");
                }

                strip.x = new_x as u16;
                strip.y = new_y as u16;
            }
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
    pub(crate) fn set_cached_strips(&mut self, strips: Box<[Strip]>, alphas: Box<[u8]>) {
        self.cached_strips = Some(CachedStrips::new(strips, alphas));
    }

    /// Get cached strips
    pub(crate) fn get_cached_strips(&self) -> Option<(&[Strip], &[u8])> {
        self.cached_strips
            .as_ref()
            .map(|cached| (cached.strips(), cached.alphas()))
    }

    /// Set strip ranges
    pub(crate) fn set_strip_ranges(&mut self, ranges: Vec<(usize, usize)>) {
        self.strip_ranges = Some(ranges);
    }

    /// Get strip ranges
    pub(crate) fn get_strip_ranges(&self) -> &[(usize, usize)] {
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

/// Recorder context that captures commands
pub struct Recorder<'a> {
    /// The recording to capture commands into
    recording: &'a mut Recording,
}

impl<'a> Recorder<'a> {
    pub(crate) fn new(recording: &'a mut Recording) -> Self {
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

    /// Set the blend mode for subsequent rendering operations
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.recording
            .commands
            .push(RenderCommand::SetBlendMode(blend_mode));
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
