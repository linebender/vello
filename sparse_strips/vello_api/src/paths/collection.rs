// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use peniko::{
    Fill,
    kurbo::{Affine, PathEl, Shape, Stroke},
};

#[derive(Debug)]
pub struct PathSet {
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
    pub fn prepare_fill(
        &mut self,
        fill_rule: Fill,
        meta: PreparedPathMeta,
        shape: &impl Shape,
    ) -> usize {
        let start_index = self.elements.len();
        // TODO: Determine the optimal tolerance from the transform.
        self.elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(PathMeta {
            transform: meta.transform,
            start_index,
            operation: Operation::Fill(fill_rule),
        });

        meta_index
    }

    // TODO: Maybe require stroke expansion happen *prior* to this stage?
    pub fn prepare_stroke(
        &mut self,
        stroke_rule: Stroke,
        meta: PreparedPathMeta,
        shape: &impl Shape,
    ) -> usize {
        let start_index = self.elements.len();
        // TODO: Determine the optimal tolerance from the transform.
        self.elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(PathMeta {
            transform: meta.transform,
            start_index,
            operation: Operation::Stroke(stroke_rule),
        });

        meta_index
    }
}

#[derive(Debug)]
pub enum Operation {
    Stroke(Stroke),
    Fill(Fill),
}

#[derive(Debug)]
pub struct PathMeta {
    pub transform: Affine,
    pub start_index: usize,
    pub operation: Operation,
}

#[derive(Debug)]
pub struct PreparedPathMeta {
    pub transform: Affine,
    // TODO: Do we need these properties?
    pub width: u16,
    pub height: u16,
    pub x_offset: i32,
    pub y_offset: i32,
}
