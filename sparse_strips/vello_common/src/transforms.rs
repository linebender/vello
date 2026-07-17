// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared structures for holding different transforms.

use crate::kurbo::Affine;
use smallvec::{SmallVec, smallvec};

/// Context for holding transforms for paths and paints.
#[derive(Debug, Clone)]
pub struct Transforms {
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
            transform: Affine::IDENTITY,
            paint_transform: Affine::IDENTITY,
        }
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

    // Unlike [`Self::effective_path_transform`], this intentionally does not apply
    // the root transform because clipping handles filter viewport shifts separately.
    /// Return the transform used for non-isolated clip paths.
    pub fn clip_path_transform(&self) -> Affine {
        self.transform
    }
}

/// Stack of root transforms.
#[derive(Debug)]
pub struct RootTransforms {
    transforms: SmallVec<[Affine; 3]>,
}

impl Default for RootTransforms {
    fn default() -> Self {
        Self::new()
    }
}

impl RootTransforms {
    /// Create a new root transform stack.
    pub fn new() -> Self {
        Self {
            transforms: smallvec![Affine::IDENTITY],
        }
    }

    /// Return the currently active root transform.
    pub fn root_transform(&self) -> Affine {
        *self
            .transforms
            .last()
            .expect("root transform stack should never be empty")
    }

    /// Return the transform used for rendering paths.
    pub fn effective_path_transform(&self, transforms: &Transforms) -> Affine {
        self.root_transform() * transforms.transform
    }

    /// Return the transform used for rendering paints.
    pub fn effective_paint_transform(&self, transforms: &Transforms) -> Affine {
        self.effective_path_transform(transforms) * transforms.paint_transform
    }

    /// Push a new root transform relative to the currently active root transform.
    pub fn push_root(&mut self, relative_transform: Affine) {
        self.transforms
            .push(relative_transform * self.root_transform());
    }

    /// Pop the last root layer.
    pub fn pop_root(&mut self) {
        self.transforms.pop();
    }

    /// Reset the root transform stack.
    pub fn reset(&mut self) {
        self.transforms.clear();
        self.transforms.push(Affine::IDENTITY);
    }
}
