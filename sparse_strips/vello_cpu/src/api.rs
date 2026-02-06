// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration of the experimental Vello API preview into Vello CPU.
//!
//! <div class="warning">
//!
//! Vello API is currently in an experimental phase, released only as a preview, and has no stability guarantees.
//! See [its documentation](vello_api) for more details.
//!
//! </div>

use vello_api::{
    PaintScene, Scene,
    exact::ExactPathElements,
    peniko::Style,
    scene::{RenderCommand, extract_integer_translation},
    texture::TextureId,
};
use vello_common::{
    kurbo::{self, Affine, BezPath},
    paint::{ImageId, ImageSource},
    peniko::{BlendMode, Brush, Color, Fill, ImageBrush},
};

use crate::RenderContext;

/// An adapter to implement [`PaintScene`] for Vello CPU's ['`Scene`'][RenderContext] type.
///
/// This type exists to avoid breaking the other APIs in this crate whilst we land/stabilise Vello API.
#[derive(Debug)]
pub struct CPUScenePainter {
    /// The underlying render context. This is public on an interim basis, whilst we decide how
    /// Vello API will develop further.
    pub render_context: RenderContext,
}

impl PaintScene for CPUScenePainter {
    fn append(
        &mut self,
        mut scene_transform: Affine,
        Scene {
            // Make sure we consider all the fields of Scene by destructuring
            paths: input_paths,
            commands: input_commands,

            hinted: input_hinted,
        }: &Scene,
    ) -> Result<(), ()> {
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
                    self.render_context.set_transform(scene_transform * *affine);
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
                            self.render_context.set_stroke(stroke.clone());
                            self.render_context.stroke_path(&bezpath);
                        }
                        Style::Fill(fill) => {
                            self.render_context.set_fill_rule(*fill);
                            self.render_context.fill_path(&bezpath);
                        }
                    }
                }
                RenderCommand::PushLayer(push_layer_command) => {
                    self.render_context
                        .set_transform(push_layer_command.clip_transform);
                    let clip_path = if let Some(path_id) = push_layer_command.clip_path {
                        let path = &input_paths.meta[usize::try_from(path_id.0).unwrap()];
                        // TODO: Also correctly support the case where the meta has a `Style::Stroke`
                        let path_end = &input_paths
                            .meta
                            .get(usize::try_from(path_id.0).unwrap() + 1)
                            .map_or(input_paths.elements.len(), |it| it.start_index);
                        let segments = &input_paths.elements[path.start_index..*path_end];
                        // TODO: Obviously, ideally we'd not be allocating here. This is forced by the current public API of Vello CPU.
                        let bezpath = BezPath::from_iter(segments.iter().cloned());
                        Some(bezpath)
                    } else {
                        None
                    };
                    self.render_context.push_layer(
                        clip_path.as_ref(),
                        push_layer_command.blend_mode,
                        push_layer_command.opacity,
                        None,
                        None,
                    );
                }
                RenderCommand::PopLayer => self.render_context.pop_layer(),
                RenderCommand::SetPaint(paint_transform, brush) => {
                    self.render_context.set_paint_transform(*paint_transform);
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
                    self.render_context.set_paint(brush);
                }
                RenderCommand::BlurredRoundedRectPaint(_) => {
                    unimplemented!(
                        "Vello CPU doesn't expose drawing a blurred rounded rectangle in custom shapes yet."
                    )
                }
            }
        }

        Ok(())
    }

    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: &impl ExactPathElements) {
        self.render_context.set_transform(transform);
        self.render_context.set_fill_rule(fill_rule);
        // However, using `to_path` avoids allocation in some cases.
        // TODO: Tweak inner API to accept an `ExactPathElements` (or at least, the resultant iterator)
        // That would avoid the superfluous allocation here.
        self.render_context
            .fill_path(&path.exact_path_elements().collect());
    }

    fn stroke_path(
        &mut self,
        transform: Affine,
        stroke_params: &kurbo::Stroke,
        path: &impl ExactPathElements,
    ) {
        self.render_context.set_transform(transform);
        self.render_context.set_stroke(stroke_params.clone());
        // TODO: As in `fill_path`
        self.render_context
            .stroke_path(&path.exact_path_elements().collect());
    }

    fn set_brush(
        &mut self,
        brush: impl Into<Brush<ImageBrush<TextureId>>>,
        paint_transform: Affine,
    ) {
        self.render_context.set_paint_transform(paint_transform);
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
        self.render_context.set_paint(brush);
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        _paint_transform: Affine,
        _color: Color,
        _rect: &kurbo::Rect,
        _radius: f32,
        _std_dev: f32,
    ) {
        // This is "trivially" fixable.
        unimplemented!(
            "Vello CPU doesn't expose drawing a blurred rounded rectangle in custom shapes yet."
        )
    }

    fn fill_blurred_rounded_rect(
        &mut self,
        transform: Affine,
        color: Color,
        rect: &kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.render_context.set_paint(color);
        self.render_context.set_transform(transform);
        self.render_context
            .fill_blurred_rounded_rect(rect, radius, std_dev);
    }

    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<&impl ExactPathElements>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        // We set the fill rule to nonzero for the clip path as a reasonable default.
        // We should make it user provided in the future
        self.render_context.set_fill_rule(Fill::NonZero);
        self.render_context.set_transform(clip_transform);
        self.render_context.push_layer(
            // TODO: As in `fill_path`
            clip_path
                .map(|it| it.exact_path_elements().collect())
                .as_ref(),
            blend_mode,
            opacity,
            None,
            None,
        );
    }

    fn push_clip_layer(&mut self, clip_transform: Affine, path: &impl ExactPathElements) {
        self.render_context.set_fill_rule(Fill::NonZero);
        self.render_context.set_transform(clip_transform);
        // TODO: As in `fill_path`
        self.render_context
            .push_clip_layer(&path.exact_path_elements().collect());
    }

    fn pop_layer(&mut self) {
        self.render_context.pop_layer();
    }
}
