// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Additional utilities for clipping that complement the [`vello_common::clip`] module.
//!
//! This module implements an additional piece of logic to make non-isolated clips work properly
//! with filter layers. The root of all "evil" that requires us to implement this wrapper around
//! [`vello_common::clip::ClipContext`] is that, as the user pushes new filter layers into the
//! render context, we eagerly apply a shift to all subsequently rendered contents to ensure that
//! everything necessary for correct filter rendering is guaranteed to be visible. However, since
//! the clip stack eagerly generates strips for each clip path that is clipped to the original
//! viewport, those generated clip paths cannot just be translated on demand to account for the
//! source shift of the filter layer. Therefore, every time a new filter layer is pushed, we need
//! to regenerate the clip context for that specific layer to ensure clips are applied correctly.

// TODO: Perhaps this will have to be moved to vello_common, in case vello_hybrid needs a
// similar workaround.

use crate::kurbo::{Affine, BezPath, PathEl};
use crate::peniko::Fill;
use crate::util::{Clear, Pool};
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::clip::{ClipContext, PathDataRef};
use vello_common::strip_generator::StripGenerator;

/// Raw data of a previously pushed clip path.
#[derive(Debug)]
struct RawClip {
    /// The range of commands in [`ClipState::path_elements`] belonging to this clip path.
    path: Range<usize>,
    fill_rule: Fill,
    transform: Affine,
    aliasing_threshold: Option<u8>,
}

/// A frame containing clipping-relevant state for the root layer or a filter layer.
#[derive(Debug)]
struct ClipFrame {
    /// The clip context of the parent.
    parent_context: ClipContext,
    /// The accumulated source shift of the current layer.
    source_shift: Affine,
    /// The current revision of the clipping state.
    clip_revision: u64,
}

/// State for managing clip paths across multiple viewports.
#[derive(Debug)]
pub(crate) struct ClipState {
    /// The currently active clip context.
    context: ClipContext,
    /// A pool of reusable clip contexts.
    context_pool: Pool<ClipContext>,
    /// A flat factor of path elements storing the original path data of clip paths.
    path_elements: Vec<PathEl>,
    /// Raw data of the currently active stack of clip paths
    raw_clips: Vec<RawClip>,
    /// Stack of pushed clip frames.
    frames: Vec<ClipFrame>,
    /// The current revision.
    revision: u64,
}

impl Clear for ClipContext {
    fn clear(&mut self) {
        self.reset();
    }
}

impl ClipState {
    pub(crate) fn new() -> Self {
        Self {
            context: ClipContext::new(),
            context_pool: Pool::default(),
            path_elements: Vec::new(),
            raw_clips: Vec::new(),
            frames: Vec::new(),
            revision: 0,
        }
    }

    pub(crate) fn get(&self) -> Option<PathDataRef<'_>> {
        self.context.get()
    }

    pub(crate) fn push_filter_surface(
        &mut self,
        source_shift: (u16, u16),
        strip_generator: &mut StripGenerator,
    ) {
        let parent_context = core::mem::replace(&mut self.context, self.context_pool.take());
        let source_shift =
            Affine::translate((f64::from(source_shift.0), f64::from(source_shift.1)))
                * self.active_shift();
        self.frames.push(ClipFrame {
            parent_context,
            source_shift,
            clip_revision: self.revision,
        });
        self.rebuild_context(strip_generator);
    }

    pub(crate) fn pop_filter_surface(&mut self, strip_generator: &mut StripGenerator) {
        let frame = self.frames.pop().expect("filter clip stack underflow");
        let filter_context = core::mem::replace(&mut self.context, frame.parent_context);
        self.context_pool.submit(filter_context);
        if self.revision == frame.clip_revision {
            // No new clip paths have been pushed or popped since then, so we don't have to rebuild it.
        } else {
            self.rebuild_context(strip_generator);
        }
    }

    pub(crate) fn push_clip(
        &mut self,
        path: &BezPath,
        strip_generator: &mut StripGenerator,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        let path_start = self.path_elements.len();
        self.path_elements.extend(path.iter());
        let path = path_start..self.path_elements.len();
        let clip_transform = self.active_shift() * transform;

        self.context.push_clip(
            self.path_elements[path.clone()].iter().copied(),
            strip_generator,
            fill_rule,
            clip_transform,
            aliasing_threshold,
        );
        self.raw_clips.push(RawClip {
            path,
            fill_rule,
            transform,
            aliasing_threshold,
        });
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn pop_clip(&mut self) {
        let raw_clip = self.raw_clips.pop().expect("clip stack underflowed");
        self.path_elements.truncate(raw_clip.path.start);
        self.context.pop_clip();
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn reset(&mut self) {
        self.context.reset();
        for frame in self.frames.drain(..) {
            self.context_pool.submit(frame.parent_context);
        }
        self.path_elements.clear();
        self.raw_clips.clear();
        self.revision = 0;
    }

    fn active_shift(&self) -> Affine {
        self.frames
            .last()
            .map_or(Affine::IDENTITY, |frame| frame.source_shift)
    }

    fn rebuild_context(&mut self, strip_generator: &mut StripGenerator) {
        self.context.reset();
        let active_shift = self.active_shift();
        for raw_clip in &self.raw_clips {
            self.context.push_clip(
                self.path_elements[raw_clip.path.clone()].iter().copied(),
                strip_generator,
                raw_clip.fill_rule,
                active_shift * raw_clip.transform,
                raw_clip.aliasing_threshold,
            );
        }
    }
}
