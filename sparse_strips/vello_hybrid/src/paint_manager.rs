// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Unified paint index management for scene and filter paints.

use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;

/// Manages scene paints (borrowed) and filter paints (owned), providing
/// unified global indexing across both collections.
pub(crate) struct PaintManager<'a> {
    scene_paints: &'a [EncodedPaint],
    filter_paints: Vec<EncodedPaint>,
}

impl<'a> PaintManager<'a> {
    /// Create a new `PaintManager` with borrowed scene paints and an owned
    /// (pre-cleared) filter paints vec.
    pub(crate) fn new(scene_paints: &'a [EncodedPaint]) -> Self {
        Self {
            scene_paints,
            // TODO: Reuse allocation? Unfortunately a bit tricky because this struct
            // has a lifetime and therefore can't be stored in the main renderer. Workarounds
            // are possible, but not sure if worth it.
            filter_paints: Vec::new(),
        }
    }

    /// Push a filter paint and return its **global** index.
    pub(crate) fn push(&mut self, paint: EncodedPaint) -> usize {
        let idx = self.len();
        self.filter_paints.push(paint);
        idx
    }

    /// Unified lookup across scene then filter paints.
    pub(crate) fn get(&self, index: usize) -> Option<&EncodedPaint> {
        let scene_len = self.scene_paints.len();

        self.scene_paints
            .get(index)
            .or_else(|| self.filter_paints.get(index - scene_len))
    }

    /// Iterator chaining scene paints then filter paints.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &EncodedPaint> {
        self.scene_paints.iter().chain(self.filter_paints.iter())
    }

    /// Total number of paints (scene + filter).
    pub(crate) fn len(&self) -> usize {
        self.scene_paints.len() + self.filter_paints.len()
    }
}
