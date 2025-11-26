// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use peniko::kurbo::Affine;

use crate::{
    PaintScene,
    baseline::BaselinePreparePaths,
    prepared::{PreparePaths, PreparePathsDirect, TransformablePreparedPaths},
    recording::{RecordScene, TransformedRecording},
};

type RenderCommand = ();

#[derive(Debug)]
pub struct BaselinePainter<P> {
    prepared_paths: P,
    commands: Vec<RenderCommand>,
    width: u16,
    height: u16,
    origin_x_offset: i32,
    origin_y_offset: i32,
}

impl<P> BaselinePainter<P> {
    pub fn new(paths: P) -> Self {
        Self {
            prepared_paths: paths,
            commands: Vec::new(),
            height: 0,
            width: 0,
            origin_x_offset: 0,
            origin_y_offset: 0,
        }
    }
    pub fn set_dimensions(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }

    pub fn set_origin_offset(&mut self, origin_x_offset: i32, origin_y_offset: i32) {
        self.origin_x_offset = origin_x_offset;
        self.origin_y_offset = origin_y_offset;
    }
    pub fn into_inner(self) -> P {
        self.prepared_paths
    }
}
impl BaselinePainter<BaselinePreparePaths> {
    pub fn clear(&mut self) {
        self.commands.clear();
        self.prepared_paths.clear();
    }
}

impl Default for BaselinePainter<BaselinePreparePaths> {
    fn default() -> Self {
        Self::new(BaselinePreparePaths::default())
    }
}

#[expect(
    unused_variables,
    clippy::todo,
    // There aren't any huge unknowns with writing this implementation, so it's fine to leave incomplete.
    reason = "Incomplete implementation, needs to exist for Vellos API and Hybrid to type check."
)]
impl<P: PreparePaths> PaintScene for BaselinePainter<P> {
    fn width(&self) -> u16 {
        todo!()
    }
    fn height(&self) -> u16 {
        todo!()
    }
    fn fill_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        fill_rule: peniko::Fill,
        path: impl peniko::kurbo::Shape,
    ) {
        todo!()
    }
    fn stroke_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        stroke_params: &peniko::kurbo::Stroke,
        path: impl peniko::kurbo::Shape,
    ) {
        todo!()
    }
    fn set_brush(
        &mut self,
        brush: impl Into<peniko::Brush<peniko::ImageBrush<crate::texture::TextureId>>>,
        // transform: peniko::kurbo::Affine,
        paint_transform: peniko::kurbo::Affine,
    ) {
        todo!()
    }
    fn set_blurred_rounded_rect_brush(
        &mut self,
        paint_transform: peniko::kurbo::Affine,
        color: peniko::Color,
        rect: &peniko::kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        todo!()
    }
    fn set_blend_mode(&mut self, blend_mode: peniko::BlendMode) {
        todo!()
    }
    fn push_clip_path(&mut self, path: &peniko::kurbo::BezPath) {
        todo!()
    }
    fn pop_clip_path(&mut self) {
        todo!()
    }
    fn push_layer(
        &mut self,
        clip_transform: peniko::kurbo::Affine,
        clip_path: Option<impl peniko::kurbo::Shape>,
        blend_mode: Option<peniko::BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        todo!()
    }
    fn push_clip_layer(
        &mut self,
        clip_transform: peniko::kurbo::Affine,
        path: impl peniko::kurbo::Shape,
    ) {
        todo!()
    }
    fn pop_layer(&mut self) {
        todo!()
    }
}

#[expect(
    unused_variables,
    clippy::todo,
    // There aren't any huge unknowns with writing this implementation, so it's fine to leave incomplete.
    reason = "Incomplete implementation, needs to exist for Vellos API and Hybrid to type check."
)]
impl<P: PreparePathsDirect<Target>, Target: PaintScene> RecordScene<Target> for BaselinePainter<P> {
    fn draw_into(
        &self,
        scene: &mut Target,
        x_offset: i32,
        y_offset: i32,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()> {
        // TODO: Reason about whether this needs to be - instead of +
        let x_offset = x_offset + self.origin_x_offset;
        let y_offset = y_offset + self.origin_y_offset;
        todo!();
    }
}

#[expect(
    unused_variables,
    clippy::todo,
    // There aren't any huge unknowns with writing this implementation, so it's fine to leave incomplete.
    reason = "Incomplete implementation, needs to exist for Vellos API and Hybrid to type check."
)]
impl<P: TransformablePreparedPaths<Target>, Target: PaintScene> TransformedRecording<Target>
    for BaselinePainter<P>
{
    fn draw_into(
        &self,
        scene: &mut Target,
        transform: Affine,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()> {
        // TODO: Reason about whether this needs to be - the offset.
        let transform = transform
            .pre_translate((self.origin_x_offset as f64, self.origin_y_offset as f64).into());
        todo!()
    }
}
