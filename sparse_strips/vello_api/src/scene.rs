// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{sync::Arc, vec::Vec};

use peniko::{BlendMode, ImageBrush, kurbo::Affine};

use crate::{
    PaintScene, Renderer,
    paths::{PathId, PathSet},
    texture::{TextureHandle, TextureId},
};

#[derive(Debug)]
pub enum RenderCommand {
    /// Draw a path with the current brush.
    DrawPath(Affine, PathId),
    /// Push a new layer with optional clipping and effects.
    PushLayer(PushLayerCommand),
    /// Pop the current layer.
    PopLayer,
    /// Set the current paint.
    ///
    /// The affine is currently path local for future drawing operations.
    /// That is something I expect *could* change in the future/want to change.
    ///
    /// This doesn't use [`OurBrush`] because we want to limit the
    /// number of textures stored.
    SetPaint(Affine, peniko::Brush<peniko::ImageBrush<TextureId>>),
    /// Set the paint to be a blurred rounded rectangle.
    ///
    /// This is useful for box shadows.
    BlurredRoundedRectPaint(BlurredRoundedRectBrush),
}

/// Command for pushing a new layer.
#[derive(Debug, Clone)]
pub struct PushLayerCommand {
    pub clip_transform: Affine,
    /// Clip path.
    pub clip_path: Option<PathId>,
    /// Blend mode.
    pub blend_mode: Option<BlendMode>,
    /// Opacity.
    pub opacity: Option<f32>,
    // /// Mask.
    // pub mask: Option<crate::renderer::Mask>,
    // /// Filter.
    // pub filter: Option<Filter>,
}

#[derive(Debug, Clone)]
pub struct BlurredRoundedRectBrush {
    pub paint_transform: peniko::kurbo::Affine,
    pub color: peniko::Color,
    pub rect: peniko::kurbo::Rect,
    pub radius: f32,
    pub std_dev: f32,
}

#[derive(Debug)]
pub struct Scene {
    paths: PathSet,
    commands: Vec<RenderCommand>,
    renderer: Arc<dyn Renderer>,
    hinted: bool,
    // TODO: HashSet? We'd need to bring in hashbrown, but that's probably fine
    textures: Vec<TextureHandle>,
}

impl Scene {
    pub fn new(renderer: Arc<dyn Renderer>, hinted: bool) -> Self {
        Self {
            renderer,
            paths: PathSet::new(),
            commands: Vec::new(),
            hinted,
            textures: Vec::new(),
        }
    }
}

impl Scene {
    pub fn clear(&mut self) {
        self.commands.clear();
        self.paths.clear();
    }
    pub fn hinted(&self) -> bool {
        self.hinted
    }
}

// TODO: Change module and give a better name.
pub type OurBrush = peniko::Brush<peniko::ImageBrush<TextureHandle>>;

pub fn extract_integer_translation(transform: Affine) -> Option<(f64, f64)> {
    fn is_nearly(a: f64, b: f64) -> bool {
        // TODO: This is a very arbitrary threshold.
        (a - b).abs() < 0.01
    }
    let [a, b, c, d, dx, dy] = transform.as_coeffs();
    // If there's a skew, rotation or scale, then the transform is not compatible with hinting.
    if !(is_nearly(a, 1.0) && is_nearly(b, 0.0) && is_nearly(c, 1.0) && is_nearly(d, 0.0)) {
        return None;
    }

    // TODO: Is `round` or `round_ties_even` more performant?
    let round_x = dx.round();
    let round_y = dy.round();
    if is_nearly(dx, round_x) && is_nearly(dy, round_y) {
        Some((round_x, round_y))
    } else {
        None
    }
}

impl PaintScene for Scene {
    fn append(
        &mut self,
        mut scene_transform: Affine,
        Self {
            // Make sure we consider all the fields of Scene by destructuring
            // (I wonder if there should be an opt in restriction clippy lint?)
            paths: other_paths,
            commands: other_commands,
            renderer: other_renderer,
            hinted: other_hinted,
            textures,
        }: &Scene,
    ) -> Result<(), ()> {
        if !Arc::ptr_eq(&self.renderer, other_renderer) {
            // Mismatched Renderers
            return Err(());
        }

        if *other_hinted {
            if !self.hinted {
                // Trying to bring a "hinted" scene into an unhinted context.
                return Err(());
            }
            if let Some((dx, dy)) = extract_integer_translation(scene_transform) {
                // Update the transform to be a pure integer translation.
                // This is valid as the scene is hinted, so we know it won't be scaled.
                scene_transform = Affine::translate((dx, dy));
            } else {
                // Translation not hinting compatible.
                return Err(());
            }
        }
        let path_correction_factor = self.paths.append(other_paths);
        let correct_path = |path: PathId| PathId(path.0 + path_correction_factor);
        let correct_transform = |transform: Affine| scene_transform * transform;

        self.commands
            .extend(other_commands.iter().map(|command| match command {
                RenderCommand::DrawPath(transform, path) => {
                    RenderCommand::DrawPath(correct_transform(*transform), correct_path(*path))
                }
                RenderCommand::PushLayer(command) => RenderCommand::PushLayer(PushLayerCommand {
                    clip_transform: correct_transform(command.clip_transform),
                    clip_path: command.clip_path.map(correct_path),
                    blend_mode: command.blend_mode,
                    opacity: command.opacity,
                }),
                RenderCommand::PopLayer => RenderCommand::PopLayer,
                RenderCommand::SetPaint(affine, brush) => {
                    // Don't update the paint_transform, as it's already path local.
                    RenderCommand::SetPaint(*affine, brush.clone())
                }
                RenderCommand::BlurredRoundedRectPaint(brush) => {
                    // Don't update the paint_transform, as it's (currently) already path local.
                    RenderCommand::BlurredRoundedRectPaint(brush.clone())
                }
            }));

        // We avoid duplicating the handles where possible.
        // It is likely that there's a better data structure for this (a `HashSet`?)
        // but there isn't one provided in core/alloc.x
        // We expect the number of textures to be relatively small, so this O(N^2) isn't
        // an immediate optimisation target.
        for texture in textures {
            if !self.textures.contains(texture) {
                self.textures.push(texture.clone());
            }
        }
        Ok(())
    }

    fn fill_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        fill_rule: peniko::Fill,
        path: impl peniko::kurbo::Shape,
    ) {
        let idx = self.paths.prepare_fill(fill_rule, &path);
        self.commands.push(RenderCommand::DrawPath(transform, idx));
    }
    fn stroke_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        stroke_params: &peniko::kurbo::Stroke,
        path: impl peniko::kurbo::Shape,
    ) {
        let idx = self.paths.prepare_stroke(stroke_params.clone(), &path);
        self.commands.push(RenderCommand::DrawPath(transform, idx));
    }

    fn set_brush(
        &mut self,
        brush: impl Into<OurBrush>,
        // transform: peniko::kurbo::Affine,
        paint_transform: peniko::kurbo::Affine,
    ) {
        let brush = match brush.into() {
            peniko::Brush::Image(image) => {
                let id = image.image.id();
                // We expect there to be relatively few textures per scene, so an O(n) linear scan here is *fine*
                // (i.e. even though it's O(N^2) for total textures in a scene, we expect N to be small)
                if !self.textures.contains(&image.image) {
                    self.textures.push(image.image);
                }
                peniko::Brush::Image(ImageBrush {
                    sampler: image.sampler,
                    image: id,
                })
            }
            peniko::Brush::Solid(alpha_color) => peniko::Brush::Solid(alpha_color),
            peniko::Brush::Gradient(gradient) => peniko::Brush::Gradient(gradient),
        };
        self.commands
            .push(RenderCommand::SetPaint(paint_transform, brush));
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        paint_transform: peniko::kurbo::Affine,
        color: peniko::Color,
        rect: &peniko::kurbo::Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.commands.push(RenderCommand::BlurredRoundedRectPaint(
            BlurredRoundedRectBrush {
                paint_transform,
                color,
                rect: *rect,
                radius,
                std_dev,
            },
        ));
    }

    fn push_layer(
        &mut self,
        clip_transform: peniko::kurbo::Affine,
        clip_path: Option<impl peniko::kurbo::Shape>,
        blend_mode: Option<peniko::BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        let clip_idx = if let Some(clip_path) = clip_path {
            Some(self.paths.prepare_fill(
                // TODO?
                peniko::Fill::NonZero,
                &clip_path,
            ))
        } else {
            None
        };
        self.commands
            .push(RenderCommand::PushLayer(PushLayerCommand {
                clip_transform,
                clip_path: clip_idx,
                blend_mode,
                opacity,
            }));
    }
    fn pop_layer(&mut self) {
        self.commands.push(RenderCommand::PopLayer);
    }
}
