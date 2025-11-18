// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_api::{PaintScene, Renderer, baseline::BaselinePreparePaths, texture::Texture};
use vello_common::{
    kurbo::{self, Affine, Shape},
    peniko::{BlendMode, Brush, Fill, ImageBrush, ImageData},
};

use crate::RenderContext;

pub struct VelloCPU;

impl Renderer for VelloCPU {
    type ScenePainter = CPUScenePainter;

    type PathPreparer = BaselinePreparePaths;

    fn create_texture(descriptor: vello_api::texture::TextureDescriptor) -> Texture {
        todo!()
    }

    fn create_scene(
        &mut self,
        to: &Texture,
        options: vello_api::SceneOptions,
    ) -> Self::ScenePainter {
        let (width, height) = if let Some(size) = options.size() {
            size
        } else {
            (to.descriptor().width, to.descriptor().height)
        };

        // TODO: Cache the contexts internally.
        let context = RenderContext::new(width, height);
        CPUScenePainter {
            render_context: context,
            target: to.clone(),
        }
    }

    fn queue_render(&mut self, mut from: Self::ScenePainter) {
        from.render_context.flush();
    }

    fn queue_download(&mut self, texture: &Texture) -> vello_api::DownloadId {
        todo!()
    }

    fn upload_image(
        to: &Texture,
        data: ImageData,
        region: Option<(u16, u16, u16, u16)>,
    ) -> Result<(), ()> {
        todo!()
    }

    fn create_path_cache(&mut self) -> Self::PathPreparer {
        BaselinePreparePaths::new()
    }
}

pub struct CPUScenePainter {
    render_context: RenderContext,
    target: Texture,
}

impl PaintScene for CPUScenePainter {
    fn width(&self) -> u16 {
        self.render_context.width()
    }

    fn height(&self) -> u16 {
        self.render_context.height()
    }

    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape) {
        self.render_context.set_transform(transform);
        self.render_context.set_fill_rule(fill_rule);
        // TODO: Tweak inner `fill_path` API to either take a `Shape` or an &[PathEl]
        self.render_context.fill_path(&path.to_path(0.1));
    }

    fn stroke_path(&mut self, transform: Affine, stroke_params: &kurbo::Stroke, path: impl Shape) {
        self.render_context.set_transform(transform);
        self.render_context.set_stroke(stroke_params.clone());
        self.render_context.stroke_path(&path.to_path(0.1));
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<Texture>>>,
        // The transform should be the same as for the following path.
        _: Affine,
        paint_transform: Affine,
    ) {
        // self.render_context.set_transform(transform);
        self.render_context.set_paint_transform(paint_transform);
        let brush = match brush.into() {
            Brush::Solid(alpha_color) => Brush::Solid(alpha_color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient),
            Brush::Image(_) => todo!(),
        };
        self.render_context.set_paint(brush);
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        transform: Affine,
        paint_transform: Affine,
        rect: &kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        unimplemented!()
    }

    fn fill_blurred_rounded_rect(
        &mut self,
        transform: Affine,
        rect: &kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.render_context.set_transform(transform);
        self.render_context
            .fill_blurred_rounded_rect(rect, radius, std_dev);
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.render_context.set_blend_mode(blend_mode);
    }

    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        self.render_context.set_transform(clip_transform);
        self.render_context.push_layer(
            clip_path.map(|it| it.to_path(0.1)).as_ref(),
            blend_mode,
            opacity,
            None,
        );
    }

    fn push_clip_layer(&mut self, clip_transform: Affine, path: impl Shape) {
        self.render_context.set_transform(clip_transform);
        self.render_context.push_clip_layer(
            // TODO: Not allocate
            &path.to_path(0.1),
        );
    }

    fn pop_layer(&mut self) {
        self.render_context.pop_layer();
    }
}
