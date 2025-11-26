// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use peniko::{
    Fill,
    kurbo::{Affine, PathEl, Shape, Stroke},
};

use crate::{
    PaintScene,
    prepared::{
        PreparePaths, PreparePathsDirect, PreparedPathIndex, PreparedPathMeta,
        TransformablePreparedPaths,
    },
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
impl<P: PaintScene> PreparePathsDirect<P> for BaselinePreparePaths {
    // TODO: Proper error kind
    fn draw_into(
        &self,
        scene: &mut P,
        path: PreparedPathIndex,
        x_offset: i32,
        y_offset: i32,
    ) -> Result<(), ()> {
        self.draw_into_transformed(
            scene,
            path,
            Affine::translate((x_offset as f64, y_offset as f64)),
        )
    }
}

impl<P: PaintScene> TransformablePreparedPaths<P> for BaselinePreparePaths {
    fn draw_into_transformed(
        &self,
        scene: &mut P,
        index: PreparedPathIndex,
        transform: Affine,
    ) -> Result<(), ()> {
        let idx = usize::try_from(index.0).map_err(drop)?;
        let path_details = self.meta.get(idx).ok_or(())?;
        // Overflow: Impossible if previous line succeeded, as overflow would require entire memory to be full!
        let next_path_details = self.meta.get(idx + 1);
        let path_range_end =
            next_path_details.map_or(self.path_elements.len(), |it| it.start_index);

        let shape = &self.path_elements[path_details.start_index..path_range_end];
        let transform = transform * path_details.transform;
        match &path_details.operation {
            Operation::Stroke(stroke) => {
                scene.stroke_path(transform, &stroke.clone(), shape);
            }
            Operation::Fill(fill_rule) => {
                scene.fill_path(transform, *fill_rule, shape);
            }
        }
        Ok(())
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
