// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use peniko::{
    Fill,
    kurbo::{PathEl, Shape, Stroke},
};

#[derive(Debug, Clone, Copy)]
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
        // TODO: Maybe determine the optimal tolerance from the transform.
        // If this can be post-transformed, that doesn't actually meaningfully apply?
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
        // TODO: Determine the optimal tolerance from the transform.
        // TODO: Perform dash expansion now, even if not stroke expansion?
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
