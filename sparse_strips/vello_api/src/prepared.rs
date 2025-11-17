// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{sync::Arc, vec::Vec};
use core::{any::Any, fmt::Debug};
use peniko::{
    Fill,
    kurbo::{Affine, PathEl, Shape, Stroke},
};

pub struct PreparedPaths {
    value: Arc<dyn InnerPathAccessor>,
}

impl PreparedPaths {
    fn create(value: Arc<dyn InnerPathAccessor>) -> Self {
        Self { value }
    }
}

pub trait InnerPathAccessor: Any + Send + Sync + Debug {
    fn path_count(&self) -> Option<u16>;
}

// We require the "Any" supertrait to allow end-users to downcast to a specific implementation.
pub trait PreparePaths: Any {
    fn prepare_fill(
        &mut self,
        fill_rule: Fill,
        meta: PreparedPathMeta,
        shape: &impl Shape,
        // TODO: Result? For example, if we have a fixed storage size
    ) -> PreparedPathIndex;
    fn prepare_stroke(
        &mut self,
        stroke_rule: Stroke,
        meta: PreparedPathMeta,
        stroke: &impl Shape,
    ) -> PreparedPathIndex;
}

pub struct PreparedPathMeta {
    pub transform: Affine,
    pub width: u16,
    pub height: u16,
}

pub struct PreparedPathIndex(pub u64);

#[derive(Debug)]
pub struct BaselinePreparePaths {
    // TODO: How valuable would a rope-like structure to avoid realloc be?
    path_elements: Vec<PathEl>,
    meta: Vec<BaselinePathMeta>,
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
