// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Prepared paths are acceleration structures for drawing paths.
//!
//! This is designed for efficient glyph rendering.

use core::{any::Any, fmt::Debug};
use peniko::{
    Fill,
    kurbo::{Affine, Shape, Stroke},
};

use crate::PaintScene;

#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct PreparedPathIndex(pub u64);

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

pub trait PreparePathsDirect<Scene: PaintScene>: PreparePaths {
    fn draw_into(
        &self,
        scene: &mut Scene,
        index: PreparedPathIndex,
        x_offset: i32,
        y_offset: i32,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>;
}

pub trait TransformablePreparedPaths<Scene: PaintScene>: PreparePathsDirect<Scene> {
    fn draw_into_transformed(
        &self,
        scene: &mut Scene,
        index: PreparedPathIndex,
        transform: Affine,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>;
}

#[derive(Debug)]
pub struct PreparedPathMeta {
    pub transform: Affine,
    pub width: u16,
    pub height: u16,
    pub x_offset: i32,
    pub y_offset: i32,
}
