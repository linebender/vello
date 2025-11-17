// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Versions of the APIs in Vello API which are portable to all rendering APIs.

use alloc::vec::Vec;
use peniko::{
    Fill,
    kurbo::{Affine, PathEl, Shape, Stroke},
};

use crate::{
    PaintScene,
    prepared::{PreparePaths, PreparedPathIndex, PreparedPathMeta},
};
/// The implementation of [`PreparePaths`] which only stores all the relevant path
/// segments, to be applied later.

#[derive(Debug)]
pub struct BaselinePreparePaths {
    // TODO: How valuable would a rope-like structure to avoid realloc be?
    path_elements: Vec<PathEl>,
    meta: Vec<BaselinePathMeta>,
}

impl BaselinePreparePaths {
    pub fn new() -> Self {
        Self {
            path_elements: Vec::new(),
            meta: Vec::new(),
        }
    }
    pub fn clear(&mut self) {
        self.path_elements.clear();
        self.meta.clear();
    }

    pub fn apply<P: PaintScene>(&mut self, painter: P, path: PreparedPathIndex) {
        todo!()
    }
}

impl Default for BaselinePreparePaths {
    fn default() -> Self {
        Self::new()
    }
}

impl PreparePaths for BaselinePreparePaths {
    fn prepare_fill(
        &mut self,
        fill_rule: Fill,
        meta: PreparedPathMeta,
        shape: &impl Shape,
    ) -> PreparedPathIndex {
        let start_index = self.path_elements.len();
        // TODO: Determine the optimal tolerance from the transform.
        self.path_elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(BaselinePathMeta {
            transform: meta.transform,
            start_index,
            operation: Operation::Fill(fill_rule),
        });
        PreparedPathIndex(
            meta_index
                .try_into()
                .expect("Fewer than 2^64 prepared paths."),
        )
    }

    fn prepare_stroke(
        &mut self,
        stroke_rule: Stroke,
        meta: PreparedPathMeta,
        shape: &impl Shape,
    ) -> PreparedPathIndex {
        let start_index = self.path_elements.len();
        // TODO: Determine the optimal tolerance from the transform.
        self.path_elements.extend(shape.path_elements(0.1));
        let meta_index = self.meta.len();
        self.meta.push(BaselinePathMeta {
            transform: meta.transform,
            start_index,
            operation: Operation::Stroke(stroke_rule),
        });
        PreparedPathIndex(
            meta_index
                .try_into()
                .expect("Fewer than 2^64 prepared paths."),
        )
    }
}

#[derive(Debug)]
enum Operation {
    Stroke(Stroke),
    Fill(Fill),
}

#[derive(Debug)]
struct BaselinePathMeta {
    transform: Affine,
    start_index: usize,
    operation: Operation,
}
