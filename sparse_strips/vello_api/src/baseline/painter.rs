// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use peniko::{
    BlendMode, Brush, Color, Fill, ImageBrush,
    kurbo::{self, Affine, Shape},
};

use crate::{
    PaintScene,
    baseline::BaselinePreparePaths,
    prepared::{PreparePaths, PreparePathsDirect, TransformablePreparedPaths},
    recording::{RecordScene, TransformedRecording},
    texture::TextureId,
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

impl<P: PreparePaths> PaintScene for BaselinePainter<P> {
    fn width(&self) -> u16 {
        self.width
    }

    fn height(&self) -> u16 {
        self.height
    }

    fn fill_path_new(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape) {
        todo!()
    }

    fn stroke_path_new(
        &mut self,
        transform: Affine,
        stroke_params: &kurbo::Stroke,
        path: impl Shape,
    ) {
        todo!()
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        transform: Affine,
        paint_transform: Affine,
    ) {
        todo!()
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        transform: Affine,
        paint_transform: Affine,
        color: Color,
        rect: &kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        todo!()
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        todo!()
    }

    fn push_layer_new(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        todo!()
    }

    fn push_clip_layer_new(&mut self, clip_transform: Affine, path: impl Shape) {
        todo!()
    }

    fn pop_layer(&mut self) {
        todo!()
    }

    fn read_stateful_transform(&self) -> Affine {
        todo!()
    }

    fn read_stateful_paint_transform(&self) -> Affine {
        todo!()
    }

    fn read_stateful_fill_rule(&self) -> Fill {
        todo!()
    }

    fn read_stateful_stroke(&self) -> kurbo::Stroke {
        todo!()
    }

    fn set_stroke(&mut self, stroke: kurbo::Stroke) {
        todo!()
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        todo!()
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        todo!()
    }

    fn set_transform(&mut self, transform: Affine) {
        todo!()
    }

    fn fill_blurred_rounded_rect(&mut self, rect: &kurbo::Rect, radius: f32, std_dev: f32) {
        todo!()
    }

    fn push_clip_path(&mut self, path: &kurbo::BezPath) {
        todo!()
    }

    fn pop_clip_path(&mut self) {
        todo!()
    }
}

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
        // TODO: Reason about whether this needs to be -
        let x_offset = x_offset + self.origin_x_offset;
        let y_offset = y_offset + self.origin_y_offset;
        todo!()
    }
}

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
