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

    /// Return the transform used for rendering scene geometry.
    pub fn scene_transform(&self) -> Affine {
        self.transform
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

    /// Return the transform used for rendering scene paints.
    pub fn scene_paint_transform(&self) -> Affine {
        self.scene_transform() * self.paint_transform
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

    /// Return the root transform of the currently active filter viewport, including
    /// shifts inherited from nested filters.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_transforms_accumulate_relative_transforms() {
        let mut roots = RootTransforms::new();
        let parent = Affine::translate((10.0, 20.0));
        let child = Affine::scale(2.0);

        roots.push_root(parent);
        roots.push_root(child);

        assert_eq!(roots.root_transform(), child * parent);
    }

    #[test]
    fn pop_root_restores_parent_transform() {
        let mut roots = RootTransforms::new();
        let parent = Affine::translate((10.0, 20.0));

        roots.push_root(parent);
        roots.push_root(Affine::scale(2.0));
        roots.pop_root();

        assert_eq!(roots.root_transform(), parent);
    }

    #[test]
    fn effective_transforms_include_root_scene_and_paint_transforms() {
        let mut roots = RootTransforms::new();
        let root = Affine::translate((10.0, 20.0));
        let scene = Affine::scale(2.0);
        let paint = Affine::translate((3.0, 4.0));
        let mut transforms = Transforms::new();
        transforms.set_transform(scene);
        transforms.set_paint_transform(paint);
        roots.push_root(root);

        assert_eq!(roots.effective_path_transform(&transforms), root * scene);
        assert_eq!(
            roots.effective_paint_transform(&transforms),
            root * scene * paint
        );
    }
}
