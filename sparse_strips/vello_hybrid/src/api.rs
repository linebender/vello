// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "This code is incomplete.")]
#![expect(clippy::result_unit_err, reason = "This code is incomplete.")]

use core::ptr;

use alloc::{sync::Arc, vec::Vec};
use vello_api::{
    PaintScene, Renderer, Scene,
    peniko::Style,
    scene::{RenderCommand, extract_integer_translation},
    texture::{TextureHandle, TextureId},
};
use vello_common::{
    kurbo::{self, Affine, BezPath, Shape},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush},
};

use crate::Scene as HybridScene;

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
    scene: HybridScene,
    target: TextureId,
    renderer: Arc<dyn Renderer>,
    textures: Vec<TextureHandle>,
}

impl PaintScene for HybridScenePainter {
    fn append(
        &mut self,
        mut scene_transform: Affine,
        Scene {
            // Make sure we consider all the fields of Scene by destructuring
            paths: input_paths,
            commands: input_commands,
            renderer: input_renderer,
            hinted: input_hinted,
            textures: input_textures,
        }: &Scene,
    ) -> Result<(), ()> {
        // Ideally, we'd use `ptr_eq`, but self.renderer
        if !ptr::addr_eq(Arc::as_ptr(input_renderer), Arc::as_ptr(&self.renderer)) {
            // Mismatched Renderers
            return Err(());
        }

        if *input_hinted {
            if let Some((dx, dy)) = extract_integer_translation(scene_transform) {
                // Update the transform to be a pure integer translation.
                // This is valid as the scene is hinted, so we know it won't be later scaled.
                // As such, a displacement of up to 1/100 of a pixel is inperceptible, but it
                // makes our reasoning about this easier.
                scene_transform = Affine::translate((dx, dy));
            } else {
                // Translation not hinting compatible.
                return Err(());
            }
        }
        for command in input_commands {
            match command {
                RenderCommand::DrawPath(affine, path_id) => {
                    self.scene.set_transform(scene_transform * *affine);
                    let path = &input_paths.meta[usize::try_from(path_id.0).unwrap()];
                    let path_end = &input_paths
                        .meta
                        .get(usize::try_from(path_id.0).unwrap() + 1)
                        .map_or(input_paths.elements.len(), |it| it.start_index);
                    let segments = &input_paths.elements[path.start_index..*path_end];
                    // Obviously, ideally we'd not be allocating here. This is forced by the current public API of Vello CPU.
                    let bezpath = BezPath::from_iter(segments.iter().cloned());
                    match &path.operation {
                        Style::Stroke(stroke) => {
                            self.scene.set_stroke(stroke.clone());
                            self.scene.stroke_path(&bezpath);
                        }
                        Style::Fill(fill) => {
                            self.scene.set_fill_rule(*fill);
                            self.scene.fill_path(&bezpath);
                        }
                    }
                }
                RenderCommand::PushLayer(push_layer_command) => {
                    self.scene.set_transform(push_layer_command.clip_transform);
                    let clip_path = if let Some(path_id) = push_layer_command.clip_path {
                        let path = &input_paths.meta[usize::try_from(path_id.0).unwrap()];
                        let path_end = &input_paths
                            .meta
                            .get(usize::try_from(path_id.0).unwrap() + 1)
                            .map_or(input_paths.elements.len(), |it| it.start_index);
                        let segments = &input_paths.elements[path.start_index..*path_end];
                        // Obviously, ideally we'd not be allocating here. This is forced by the current public API of Vello CPU.
                        let bezpath = BezPath::from_iter(segments.iter().cloned());
                        Some(bezpath)
                    } else {
                        None
                    };
                    self.scene.push_layer(
                        clip_path.as_ref(),
                        push_layer_command.blend_mode,
                        push_layer_command.opacity,
                        None,
                        None,
                    );
                }
                RenderCommand::PopLayer => self.scene.pop_layer(),
                RenderCommand::SetPaint(paint_transform, brush) => {
                    self.scene.set_paint_transform(*paint_transform);
                    let brush = match brush {
                        Brush::Solid(alpha_color) => Brush::Solid(*alpha_color),
                        Brush::Gradient(gradient) => Brush::Gradient(gradient.clone()),
                        Brush::Image(brush) => {
                            let image_index =
                                brush.image.to_raw().try_into().expect("Handle this.");
                            Brush::Image(ImageBrush {
                                image: ImageSource::OpaqueId(ImageId::new(image_index)),
                                sampler: brush.sampler,
                            })
                        }
                    };
                    self.scene.set_paint(brush);
                }
                RenderCommand::BlurredRoundedRectPaint(_) => {
                    unimplemented!(
                        "Vello Hybrid doesn't support drawing blurred rounded rectangles yet."
                    )
                }
            }
        }

        // We avoid duplicating stored handles.
        // It is likely that there's a better data structure for this (a `HashSet`?)
        // but there isn't one provided in core/alloc.
        // We expect the number of textures to be relatively small, so this being
        // O(N^2) isn't an immediate optimisation target.
        for texture in input_textures {
            if !self.textures.contains(texture) {
                self.textures.push(texture.clone());
            }
        }
        Ok(())
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
        brush: impl Into<Brush<ImageBrush<TextureHandle>>>,
        paint_transform: Affine,
    ) {
        self.scene.set_paint_transform(paint_transform);
        let brush = match brush.into() {
            Brush::Solid(alpha_color) => Brush::Solid(alpha_color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient),
            Brush::Image(brush) => {
                // TODO: Make this read more easily.
                let image_index = brush.image.id().to_raw().try_into().expect("Handle this.");
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
        unimplemented!("Vello Hybrid doesn't support drawing blurred rounded rectangles yet.")
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
}
