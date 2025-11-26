// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "This code is incomplete.")]
#![expect(clippy::result_unit_err, reason = "This code is incomplete.")]

use vello_api::{PaintScene, texture::TextureId};
use vello_common::{
    kurbo::{self, Affine, Shape},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush},
};

use crate::Scene;

#[cfg(feature = "wgpu")]
mod wgpu;
#[cfg(feature = "wgpu")]
pub use wgpu::VelloHybrid;

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl::VelloHybridWebgl;

#[derive(Debug)]
pub struct HybridScenePainter {
    scene: Scene,
    target: TextureId,
}

impl PaintScene for HybridScenePainter {
    fn width(&self) -> u16 {
        self.scene.width()
    }

    fn height(&self) -> u16 {
        self.scene.height()
    }

    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill_rule);
        // TODO: Tweak inner `fill_path` API to either take a `Shape` or an &[PathEl]
        self.scene.fill_path(&path.to_path(0.1));
    }

    fn stroke_path(&mut self, transform: Affine, stroke_params: &kurbo::Stroke, path: impl Shape) {
        self.scene.set_transform(transform);
        self.scene.set_stroke(stroke_params.clone());
        self.scene.stroke_path(&path.to_path(0.1));
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        paint_transform: Affine,
    ) {
        self.scene.set_paint_transform(paint_transform);
        let brush = match brush.into() {
            Brush::Solid(alpha_color) => Brush::Solid(alpha_color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient),
            Brush::Image(brush) => {
                // TODO: Make this read more easily.
                let image_index = brush.image.to_raw().try_into().expect("Handle this.");
                Brush::Image(ImageBrush {
                    image: ImageSource::OpaqueId(ImageId::new(image_index)),
                    sampler: brush.sampler,
                })
            }
        };
        self.scene.set_paint(brush);
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        _paint_transform: Affine,
        _color: Color,
        _rect: &kurbo::Rect,
        _radius: f32,
        _std_dev: f32,
    ) {
        unimplemented!("Vello Hybrid doesn't expose drawing blurred rounded rectangles yet.")
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.scene.set_blend_mode(blend_mode);
    }

    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        self.scene.set_transform(clip_transform);
        self.scene.push_layer(
            clip_path.map(|it| it.to_path(0.1)).as_ref(),
            blend_mode,
            opacity,
            None,
            None,
        );
    }

    fn push_clip_layer(&mut self, clip_transform: Affine, path: impl Shape) {
        self.scene.set_transform(clip_transform);
        self.scene.push_clip_layer(
            // TODO: Not allocate
            &path.to_path(0.1),
        );
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn push_clip_path(&mut self, path: &kurbo::BezPath) {
        self.scene.push_clip_path(path);
    }

    fn pop_clip_path(&mut self) {
        self.scene.pop_clip_path();
    }
}
