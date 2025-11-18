// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use peniko::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, Brush, Color, Fill, ImageBrush};

use crate::texture::Texture;

#[derive(Debug)]
pub struct SceneOptions {
    /// The target area within the Texture to render to.
    ///
    /// If `None`, will render to the whole texture.
    /// Format: (x0, y0, width, height).
    pub target: Option<(u16, u16, u16, u16)>,
    /// The color which the texture will be cleared to before drawing.
    ///
    /// If this is `None`, the previous content will be retained.
    /// This is useful for use cases such as the web's `CanvasRenderingContext2D`,
    /// which doesn't have automatic clearing.
    pub clear_color: Option<Color>,
}

pub trait PaintScene: Any {
    fn width(&self) -> u16;
    fn height(&self) -> u16;

    // Copied without analysis from Vello Sparse Tests.
    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape);
    fn stroke_path(&mut self, transform: Affine, stroke_params: &Stroke, path: impl Shape);

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<Texture>>>,
        transform: Affine,
        paint_transform: Affine,
    );
    fn set_blurred_rounded_rect_brush(
        &mut self,
        transform: Affine,
        paint_transform: Affine,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    );
    fn set_solid_brush(&mut self, color: Color) {
        self.set_brush(Brush::Solid(color), Affine::IDENTITY, Affine::IDENTITY);
    }

    fn fill_blurred_rounded_rect(
        &mut self,
        transform: Affine,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.set_blurred_rounded_rect_brush(transform, Affine::IDENTITY, rect, radius, std_dev);
        // The impulse response of a gaussian filter is infinite.
        // For performance reason we cut off the filter at some extent where the response is close to zero.
        let kernel_size = (2.5 * std_dev) as f64;

        let shape: Rect = rect.inflate(kernel_size, kernel_size);
        self.fill_path(transform, Fill::EvenOdd, shape);
    }

    // This can be accessed through downcasting:
    // fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    // TODO: What does this mean?
    fn set_blend_mode(&mut self, blend_mode: BlendMode);

    fn push_layer(
        &mut self,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    );
    // TODO: Why are there so many kinds of layers?
    fn push_clip_layer(&mut self, path: &BezPath);
    // fn push_blend_layer(&mut self, blend_mode: BlendMode);
    // fn push_opacity_layer(&mut self, opacity: f32);
    // fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);
}
