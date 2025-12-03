// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Recordings are convenience structures for drawing the same path multiple times.
//!
//! This can be used for (some) COLR glyphs.

// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{sync::Arc, vec::Vec};
use peniko::{BlendMode, kurbo::Affine};

use crate::{
    PaintScene, Renderer,
    paths::{PathSet, StoredPathId},
    texture::TextureId,
};

#[derive(Debug)]
pub enum RenderCommand {
    /// Draw a path with the current brush.
    DrawPath(StoredPathId),
    /// Push a new layer with optional clipping and effects.
    PushLayer(PushLayerCommand),
    /// Pop the current layer.
    PopLayer,
    /// Set the current paint.
    SetPaint(Affine, OurBrush),
    /// Set the current paint.
    BlurredRoundedRectPaint(BlurredRoundedRectBrush),
}

/// Command for pushing a new layer.
#[derive(Debug, Clone)]
pub struct PushLayerCommand {
    /// Clip path.
    pub clip_path: Option<StoredPathId>,
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
    width: u16,
    height: u16,
    origin_x_offset: i32,
    origin_y_offset: i32,
    renderer: Arc<dyn Renderer>,
}

impl Scene {
    pub fn new(renderer: Arc<dyn Renderer>) -> Self {
        Self {
            renderer,
            paths: PathSet::new(),
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
}

impl Scene {
    pub fn clear(&mut self) {
        self.commands.clear();
        self.paths.clear();
    }
}

pub type OurBrush = peniko::Brush<peniko::ImageBrush<TextureId>>;

impl PaintScene for Scene {
    fn fill_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        fill_rule: peniko::Fill,
        path: impl peniko::kurbo::Shape,
    ) {
        let idx = self.paths.prepare_fill(
            fill_rule,
            crate::paths::PreparedPathMeta {
                transform,
                x_offset: 0,
                y_offset: 0,
            },
            &path,
        );
        self.commands.push(RenderCommand::DrawPath(idx));
    }
    fn stroke_path(
        &mut self,
        transform: peniko::kurbo::Affine,
        stroke_params: &peniko::kurbo::Stroke,
        path: impl peniko::kurbo::Shape,
    ) {
        let idx = self.paths.prepare_stroke(
            stroke_params.clone(),
            crate::paths::PreparedPathMeta {
                transform,
                x_offset: 0,
                y_offset: 0,
            },
            &path,
        );
        self.commands.push(RenderCommand::DrawPath(idx));
    }
    fn set_brush(
        &mut self,
        brush: impl Into<OurBrush>,
        // transform: peniko::kurbo::Affine,
        paint_transform: peniko::kurbo::Affine,
    ) {
        self.commands
            .push(RenderCommand::SetPaint(paint_transform, brush.into()));
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
                crate::paths::PreparedPathMeta {
                    transform: clip_transform,
                    x_offset: 0,
                    y_offset: 0,
                },
                &clip_path,
            ))
        } else {
            None
        };
        self.commands
            .push(RenderCommand::PushLayer(PushLayerCommand {
                clip_path: clip_idx,
                blend_mode,
                opacity,
            }));
    }
    fn pop_layer(&mut self) {
        self.commands.push(RenderCommand::PopLayer);
    }
}

#[expect(
    unused_variables,
    clippy::todo,
    // There aren't any huge unknowns with writing this implementation, so it's fine to leave incomplete.
    reason = "Incomplete implementation, needs to exist for Vellos API and Hybrid to type check."
)]
impl Scene {
    pub fn draw_into<Target>(
        &self,
        scene: &mut Target,
        x_offset: i32,
        y_offset: i32,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>
    where
        Target: PaintScene,
    {
        // TODO: Reason about whether this needs to be - instead of +
        let x_offset = x_offset + self.origin_x_offset;
        let y_offset = y_offset + self.origin_y_offset;
        todo!();
    }

    pub fn draw_into_transformed<Target>(
        &self,
        scene: &mut Target,
        transform: Affine,
        // Error case for if:
        // - The `Scene` type doesn't line up
        //
        // Anything else?
    ) -> Result<(), ()>
    where
        Target: PaintScene,
    {
        // TODO: Reason about whether this needs to be - the offset.
        let transform = transform
            .pre_translate((self.origin_x_offset as f64, self.origin_y_offset as f64).into());
        todo!()
    }
}
