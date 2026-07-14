// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared structure for holding different transforms.

use crate::filter::FilterData;
use crate::kurbo::Affine;
use smallvec::{SmallVec, smallvec};

/// Context for holding different transforms.
#[derive(Debug, Clone)]
pub struct Transforms {
    root_transforms: SmallVec<[Affine; 3]>,
    transform: Affine,
    paint_transform: Affine,
}

impl Default for Transforms {
    fn default() -> Self {
        Self::new()
    }
}

impl Transforms {
    /// Create a new transforms context.
    pub fn new() -> Self {
        Self {
            root_transforms: smallvec![Affine::IDENTITY],
            transform: Affine::IDENTITY,
            paint_transform: Affine::IDENTITY,
        }
    }

    /// Return the currently active root transform.
    pub fn root_transform(&self) -> Affine {
        *self
            .root_transforms
            .last()
            .expect("root transform stack should never be empty")
    }

    /// Return the current scene transform.
    pub fn transform(&self) -> &Affine {
        &self.transform
    }

    /// Set the current scene transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the current scene transform.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Return the current paint transform.
    pub fn paint_transform(&self) -> &Affine {
        &self.paint_transform
    }

    /// Set the current paint transform.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.paint_transform = Affine::IDENTITY;
    }

    /// Return the transform used for rendering paths.
    pub fn effective_path_transform(&self) -> Affine {
        self.root_transform() * self.transform
    }

    // Unlike [`Self::effective_path_transform`], this intentionally does not apply
    // the root transform because clipping handles filter viewport shifts separately.
    /// Return the transform used for non-isolated clip paths.
    pub fn clip_path_transform(&self) -> Affine {
        self.transform
    }

    /// Return the transform used for rendering paints.
    pub fn effective_paint_transform(&self) -> Affine {
        self.effective_path_transform() * self.paint_transform
    }

    /// Push a new root layer.
    pub fn push_root(&mut self, filter_data: Option<&FilterData>) {
        // The important part! Let's say we have an element placed in a way such that
        // its drop shadow starts at (0, 0). In order for it to render correctly, we would
        // have to render parts of the shape that at negative viewport coordinates, which is
        // not supported. Therefore, we instead shift everything down such that we can assume
        // everything left/above (0, 0) is not needed for correct rendering, and simply
        // shift everything back when actually compositing the rendered filter layer.
        let relative_transform = filter_data.map_or(Affine::IDENTITY, |filter_data| {
            let (shift_x, shift_y) = filter_data.source_shift();

            Affine::translate((f64::from(shift_x), f64::from(shift_y)))
        });

        self.root_transforms
            .push(relative_transform * self.root_transform());
    }

    /// Pop the last root layer.
    pub fn pop_root(&mut self) {
        self.root_transforms.pop();
    }
}
