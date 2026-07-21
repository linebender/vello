// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared viewport state.

use crate::clip::{ClipState, PathDataRef};
use crate::filter::FilterData;
use crate::kurbo::{Affine, BezPath};
use crate::strip_generator::StripGenerator;
use alloc::vec::Vec;
use fearless_simd::Level;
use peniko::Fill;

/// Viewport state storing information about clip state and active strip generators in
/// currently active root viewports.
#[derive(Debug)]
pub struct ViewportState {
    clip_state: ClipState,
    strip_generator: StripGenerator,
    strip_generator_stack: Vec<StripGenerator>,
    level: Level,
}

impl ViewportState {
    /// Create a new viewport state.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            clip_state: ClipState::new(),
            strip_generator: StripGenerator::new(width, height, level),
            strip_generator_stack: Vec::new(),
            level,
        }
    }

    /// Width of the active viewport.
    pub fn width(&self) -> u16 {
        self.strip_generator.width()
    }

    /// Height of the active viewport.
    pub fn height(&self) -> u16 {
        self.strip_generator.height()
    }

    /// Return the current clip path.
    pub fn clip(&self) -> Option<PathDataRef<'_>> {
        self.clip_state.get()
    }

    /// Whether any root viewports are currently pushed.
    pub fn has_root_viewports(&self) -> bool {
        !self.strip_generator_stack.is_empty()
    }

    /// Use the active strip generator together with the current clip.
    pub fn with_generator_and_clip<R>(
        &mut self,
        f: impl FnOnce(&mut StripGenerator, Option<PathDataRef<'_>>) -> R,
    ) -> R {
        let clip = self.clip_state.get();

        f(&mut self.strip_generator, clip)
    }

    /// Push a new clip path.
    pub fn push_clip(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.clip_state.push_clip(
            path,
            &mut self.strip_generator,
            fill_rule,
            transform,
            aliasing_threshold,
        );
    }

    /// Pop the last clip path.
    pub fn pop_clip(&mut self) {
        self.clip_state.pop_clip();
    }

    /// Push a new root viewport.
    pub fn push_root_viewport(&mut self, filter_data: &FilterData) {
        let padding = filter_data.source_padding;
        let width = self
            .strip_generator
            .width()
            .saturating_add(padding.left)
            .saturating_add(padding.right);
        let height = self
            .strip_generator
            .height()
            .saturating_add(padding.top)
            .saturating_add(padding.bottom);
        // TODO: Use a pool of strip generators.
        let filter_generator = StripGenerator::new(width, height, self.level);
        let parent_generator = core::mem::replace(&mut self.strip_generator, filter_generator);
        self.strip_generator_stack.push(parent_generator);

        self.clip_state
            .push_root_viewport(filter_data.source_shift(), &mut self.strip_generator);
    }

    /// Pop the last root viewport.
    pub fn pop_root_viewport(&mut self) {
        self.strip_generator = self
            .strip_generator_stack
            .pop()
            .expect("root viewport stack underflow");

        self.clip_state.pop_root_viewport(&mut self.strip_generator);
    }

    /// Reset strip generation and clipping for a new viewport.
    pub fn reset(&mut self, width: u16, height: u16) {
        self.clip_state.reset();
        self.strip_generator_stack.clear();
        self.strip_generator.reset(width, height);
    }
}
