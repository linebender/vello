// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A collection of vector graphics paths, for use in.
//!
//! This is an internal implementation detail of [`Scene`](crate::Scene).
//! If you're a consumer of Vello API writing an application, you do not need to use this API.
//! Instead, this is exposed for use by [`Renderer`](crate::Renderer)s.
//!
//! This design has been made with "path caching" in mind, which will allow using rasterised forms of
//! paths (such as glyphs) directly in the renderer, instead of rasterising from scratch each frame.
//! This can massively improve performance and efficiency on subsequent frames.
//! This is however not implemented by any backend, so isn't provided for in Vello API.
//! It also ensures that there are very few per-frame allocations (i.e. avoids allocating
//! for each path).

use alloc::vec::Vec;
use peniko::{
    Fill,
    kurbo::{PathEl, Shape, Stroke},
};

/// The id for a single path within a given [`PathSet`].
/// This is an index into the [`meta`](PathSet::meta) field.
#[derive(Debug, Clone, Copy)]
// In a future world with path caching, this would be paired with a path group id.
// For "scene-local" paths, you would then use a marker "local" path group id.
pub struct PathId(pub u32);

#[derive(Debug)]
pub struct PathSet {
    // There are arguments for a "dynamic length" encoding here, as PathEl is sized for 6 f64s (plus a disciminant)
    // It depends somewhat on what proportion of the elements are a CurveTo
    pub elements: Vec<PathEl>,
    pub meta: Vec<PathMeta>,
}

impl PathSet {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            meta: Vec::new(),
        }
    }

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
    pub fn prepare_fill(&mut self, fill_rule: Fill, shape: &impl Shape) -> PathId {
        let start_index = self.elements.len();
        // TODO: We hard-code this tolerance to be 0.1, as every other call does so.
        // We should maybe change that at some point?
        // https://xi.zulipchat.com/#narrow/channel/197075-vello/topic/Determining.20correct.20.60Shape.60.20tolerance/with/565793178
        // If you need a different tolerance, you should pass in a `BezPath` (i.e. using Shape::to_path with the tolerance you require)
        self.elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(PathMeta {
            start_index,
            operation: Operation::Fill(fill_rule),
        });

        // TODO: Better error handling here?
        PathId(meta_index.try_into().unwrap())
    }

    // TODO: Maybe require stroke expansion happen *prior* to this stage?
    pub fn prepare_stroke(&mut self, stroke_rule: Stroke, shape: &impl Shape) -> PathId {
        let start_index = self.elements.len();
        // TODO: Perform dash expansion now, even if not stroke expansion?
        // See https://xi.zulipchat.com/#narrow/channel/260979-kurbo/topic/Removing.20dash_pattern.20from.20Stroke/with/561141820
        self.elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(PathMeta {
            start_index,
            operation: Operation::Stroke(stroke_rule),
        });

        PathId(meta_index.try_into().unwrap())
    }

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

/// How a path will be rendered.
///
/// There are reasonable arguments for moving this away from the path set,
/// to allow it to e.g. share the segments for filled and stroked versions of a path.
///
/// However, in the current draft, we make this a key property of the path.
/// This would make future caching work easier (as `Stroke` is an extremely unwieldy type to key off).
///
/// There are arguments for splitting again, into "paths" and "styled paths" or similar, but
/// that piles on complexity; and realistically how many people will use that?
///
/// Alternatively of course, as stroke expansion is going to be happening on the CPU anyway,
/// we could expand strokes extremely eagerly/require the user to perform stroke expansion.
///
/// The reason not to is that it's potentially expensive (?), and so should be scheduled to a
/// background thread.
#[derive(Debug, Clone)]
pub enum Operation {
    Stroke(Stroke),
    Fill(Fill),
}

#[derive(Debug, Clone)]
pub struct PathMeta {
    // Would u32 work here?
    pub start_index: usize,
    pub operation: Operation,
}
