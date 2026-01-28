// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A collection of vector graphics paths, for use in 2d rendering.
//!
//! This is an internal implementation detail of [`Scene`](crate::Scene).
//! If you're a consumer of Vello API writing an application, you do not need to use this API.
//! Instead, this is exposed for use by renderers.
//!
//! This design has been made with "path caching" in mind, which will allow using rasterised forms of
//! paths (such as glyphs) directly in the renderer, instead of rasterising from scratch each frame.
//! This can massively improve performance and efficiency on subsequent frames.
//! This is however not implemented by any backend, so isn't provided for in Vello API.
//! It also ensures that there are very few per-frame allocations (i.e. avoids allocating
//! for each path).

use alloc::vec::Vec;
use peniko::{Style, kurbo::PathEl};

use crate::exact::ExactPathElements;

/// The id for a single path within a given [`PathSet`].
/// This is an index into the [`meta`](PathSet::meta) field.
#[derive(Debug, Clone, Copy)]
// In a future world with path caching, this would be paired with a path group id.
// For "scene-local" paths, you would then use a marker "local" path group id.
pub struct PathId(pub u32);

/// A collection of filled or stroked paths, each associated with an id.
///
/// As noted in the [module level documentation](crate::paths), this type is an implementation
/// detail of [`Scene`](crate::Scene).
/// As such, the fields are public, allowing implementations of Vello API to read the contained paths.
///
/// This representation of paths is not simply the path points (as in, for example, an svg "path" attribute).
/// Instead, this also contains the attributes which describe the shape for which the
/// path points provide the outline.
/// That is, the elements of this type are either a filled shape, or a stroked path,
/// without any brush information.
/// Each `Scene` stores a sequence of these, with the associated brush, to create a 2d scene.
///
/// This separation is designed for a future path caching mechanism, where the rasterised geometry
/// of a path can be computed once, then re-used with multiple brushed, to increase efficiency.
/// Note however that this plan is not proven in the current version of Vello API.
#[derive(Debug)]
// The same "reason about visibility" comment applies as in `Scene`
pub struct PathSet {
    // There are arguments for a "dynamic length" encoding here, as PathEl is sized for 6 f64s (plus a disciminant)
    // It depends somewhat on what proportion of the elements are a CurveTo
    /// The elements of the contained paths.
    pub elements: Vec<PathEl>,
    /// The metadata about each path.
    pub meta: Vec<PathMeta>,
}

impl PathSet {
    /// Create a new, empty path collection.
    ///
    /// This method doesn't allocate.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            meta: Vec::new(),
        }
    }

    /// Clears the path set, removing all values.
    ///
    /// This does not free the underlying allocations.
    pub fn clear(&mut self) {
        self.elements.clear();
        self.meta.clear();
    }
}

impl Default for PathSet {
    fn default() -> Self {
        Self::new()
    }
}

impl PathSet {
    /// Prepare an outline for drawing as a shape with the given style.
    ///
    /// This returns the id of this path in this `PathSet`.
    /// See the docs on [`append`](PathSet::append) for how this changes when path sets are combined.
    ///
    /// This method is generally only expected to be used by [`Scene`](crate::Scene).
    pub fn prepare_shape(
        &mut self,
        shape: &impl ExactPathElements,
        style: impl Into<Style>,
    ) -> PathId {
        let start_index = self.elements.len();
        self.elements.extend(shape.exact_path_elements());
        let meta_index = self.meta.len();
        self.meta.push(PathMeta {
            start_index,
            operation: style.into(),
        });

        // TODO: Better error handling here?
        PathId(meta_index.try_into().unwrap())
    }

    /// Append the shapes in `other` to this pathset.
    ///
    /// The return value should be added to the field of [`PathId`]s from `other`
    /// for use in the combined pathset (i.e. the new value of `self`).
    ///
    /// This method is expected to be used to implement [`PaintScene::append`](crate::PaintScene::append).
    #[must_use]
    pub fn append(&mut self, other: &Self) -> u32 {
        let external_correction_factor = self.meta.len().try_into().unwrap();
        let internal_correction_factor = self.elements.len();
        self.elements.extend(&other.elements);
        self.meta.extend(other.meta.iter().cloned().map(|mut it| {
            it.start_index += internal_correction_factor;
            it
        }));

        external_correction_factor
    }
}

/// Metadata about a single path in a [`PathSet`].
#[derive(Debug, Clone)]
pub struct PathMeta {
    // Would u32 work here?
    /// The index in [`PathSet::elements`] from which this path's elements starts.
    ///
    /// The path ends at the start of the next `PathMeta`, and so for the final path
    /// the elements are the remaining elements.
    pub start_index: usize,
    /// How the path will be rendered.
    ///
    /// There are reasonable arguments for moving this away from the path set,
    /// to allow it to e.g. share the segments for filled and stroked versions of a path.
    /// However, in the current draft, we make this a key property of the path.
    /// This would make future caching work easier (as `Stroke` is an extremely unwieldy type to key off).
    ///
    /// There are arguments for splitting again, into "paths" and "styled paths" or similar, but
    /// that piles on complexity; and realistically how many people will use that?
    /// Alternatively, if we made `PathMeta` store a range instead of a single index,
    /// that makes reusing segments much easier.
    ///
    /// As stroke expansion is going to be happening on the CPU anyway,
    /// we could expand strokes extremely eagerly/require the user to perform stroke expansion.
    /// The reason not to is that it's potentially expensive (?), and so should be scheduled to a
    /// background thread.
    pub operation: Style,
}
