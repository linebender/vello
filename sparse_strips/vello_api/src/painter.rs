// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use peniko::kurbo::{self, Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, Brush, Color, Fill, ImageBrush};

use crate::texture::TextureId;

pub trait PaintScene: Any {
    fn width(&self) -> u16;
    fn height(&self) -> u16;

    // Legacy methods, which we don't expect to keep around long-term
    // However, these are needed to make porting vello_sparse_tests tractable

    fn read_stateful_transform(&self) -> Affine;
    fn read_stateful_paint_transform(&self) -> Affine;
    fn read_stateful_fill_rule(&self) -> Fill;
    fn read_stateful_stroke(&self) -> Stroke;

    fn set_stroke(&mut self, stroke: kurbo::Stroke);
    fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);

    fn fill_path(&mut self, path: &kurbo::BezPath) {
        self.fill_path_new(
            self.read_stateful_transform(),
            self.read_stateful_fill_rule(),
            path,
        );
    }

    fn stroke_path(&mut self, path: &kurbo::BezPath) {
        self.stroke_path_new(
            self.read_stateful_transform(),
            &self.read_stateful_stroke(),
            path,
        );
    }

    fn fill_rect(&mut self, rect: &kurbo::Rect) {
        self.fill_path_new(
            self.read_stateful_transform(),
            self.read_stateful_fill_rule(),
            rect,
        );
    }

    fn fill_blurred_rounded_rect(&mut self, rect: &kurbo::Rect, radius: f32, std_dev: f32);

    fn stroke_rect(&mut self, rect: &kurbo::Rect) {
        self.stroke_path_new(
            self.read_stateful_transform(),
            &self.read_stateful_stroke(),
            rect,
        );
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&kurbo::BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<()>,
    ) {
        if mask.is_some() {
            unimplemented!();
        }
        self.push_layer_new(
            self.read_stateful_transform(),
            clip_path,
            blend_mode,
            opacity,
        );
    }

    fn push_clip_layer(&mut self, path: &kurbo::BezPath) {
        self.push_clip_layer_new(self.read_stateful_transform(), path);
    }

    fn set_paint(&mut self, paint: impl Into<Brush<ImageBrush<TextureId>>>) {
        self.set_brush(
            paint,
            self.read_stateful_transform(),
            self.read_stateful_paint_transform(),
        );
    }

    // ---------------
    // The actual expected API (less any _new) suffices
    fn fill_path_new(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape);
    fn stroke_path_new(&mut self, transform: Affine, stroke_params: &Stroke, path: impl Shape);

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        transform: Affine,
        paint_transform: Affine,
    );
    fn set_blurred_rounded_rect_brush(
        &mut self,
        transform: Affine,
        paint_transform: Affine,
        color: Color,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    );
    fn set_solid_brush(&mut self, color: Color) {
        // The transform doesn't matter for a solid color brush.
        self.set_brush(Brush::Solid(color), Affine::IDENTITY, Affine::IDENTITY);
    }

    fn fill_blurred_rounded_rect_new(
        &mut self,
        transform: Affine,
        color: Color,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.set_blurred_rounded_rect_brush(
            transform,
            Affine::IDENTITY,
            color,
            rect,
            radius,
            std_dev,
        );
        // The impulse response of a gaussian filter is infinite.
        // For performance reason we cut off the filter at some extent where the response is close to zero.
        let kernel_size = (2.5 * std_dev) as f64;

        let shape: Rect = rect.inflate(kernel_size, kernel_size);
        self.fill_path_new(transform, Fill::EvenOdd, shape);
    }

    // This can be accessed through downcasting:
    // fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    // TODO: What does this mean?
    fn set_blend_mode(&mut self, blend_mode: BlendMode);

    // TODO: Do we want this exposed?
    // TODO: &impl Shape?
    fn push_clip_path(&mut self, path: &kurbo::BezPath);
    fn pop_clip_path(&mut self);

    fn push_layer_new(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    );
    // TODO: Why are there so many kinds of layers?
    fn push_clip_layer_new(&mut self, clip_transform: Affine, path: impl Shape);

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer_new(
            // Ok, this shows a real footgun with the proped "push_layer" API here.
            self.read_stateful_transform(),
            None::<BezPath>,
            Some(blend_mode),
            None,
        );
    }
    fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer_new(
            self.read_stateful_transform(),
            None::<BezPath>,
            None,
            Some(opacity),
        );
    }

    fn pop_layer(&mut self);
}

#[derive(Debug)]
pub struct SceneOptions {
    /// The target area within the Texture to render to.
    ///
    /// If `None`, will render to the whole texture.
    /// Format: (x0, y0, width, height).
    // TODO: This is currently extremely underutilised - it's only used for
    // resizing the canvas in webgl.
    pub target: Option<(u16, u16, u16, u16)>,
    /// The color which the texture will be cleared to before drawing.
    ///
    /// If this is `None`, the previous content will be retained.
    /// This is useful for use cases such as the web's `CanvasRenderingContext2D`,
    /// which doesn't have automatic clearing.
    pub clear_color: Option<Color>,
}

impl SceneOptions {
    pub fn size(&self) -> Option<(u16, u16)> {
        self.target.map(|(_, _, width, height)| (width, height))
    }
}
