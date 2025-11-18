// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Prepared paths are versions of.

use alloc::sync::Arc;
use core::{any::Any, fmt::Debug};
use peniko::{
    Fill,
    kurbo::{Affine, Shape, Stroke},
};

use crate::PaintScene;

pub struct PreparedPaths {
    value: Arc<dyn InnerPathAccessor>,
}

impl PreparedPaths {
    pub fn create(value: Arc<dyn InnerPathAccessor>) -> Self {
        Self { value }
    }
    pub fn handle(&self) -> &Arc<dyn InnerPathAccessor> {
        &self.value
    }
}

pub trait InnerPathAccessor: Any + Send + Sync + Debug {
    // TODO: What is the use case for this?
    fn path_count(&self) -> Option<u16>;
}

#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct PreparedPathIndex(pub u64);

// We require the "Any" supertrait to allow end-users to downcast to a specific implementation.
pub trait PreparePaths<Scene: PaintScene>: Any {
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
    fn draw_into(
        &self,
        scene: &mut Scene,
        index: PreparedPathIndex,
        x_offset: i32,
        y_offset: i32,
    ) -> Result<(), ()>;
}

pub struct PreparedPathMeta {
    pub transform: Affine,
    pub width: u16,
    pub height: u16,
}
