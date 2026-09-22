// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Replays a decoded scene through the shared `Renderer` interface.
//!
//! The generated snapshot test in `snapshot_test` must issue exactly the same calls, so keep
//! the two in sync when the scene model changes.

use crate::images::ImageTable;
use crate::scene::{
    ClipCommand, Command, DrawCommand, DrawStyle, FuzzScene, FuzzShape, HEIGHT, LayerKind,
    MAX_CLIP_DEPTH, MAX_LAYER_DEPTH, WIDTH,
};
use vello_common::kurbo::{Affine, Rect};
use vello_tests::renderer::Renderer;

/// Layers are all popped at the end because rendering requires it; pending clip paths are left in
/// place because both backends permit that.
///
/// `images` must have been registered on `renderer`, which starts from a reset state.
pub(crate) fn replay_scene(scene: &FuzzScene, renderer: &mut impl Renderer, images: &ImageTable) {
    renderer.set_transform(Affine::IDENTITY);
    renderer.set_paint(scene.background.to_color());
    renderer.fill_rect(&Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT)));

    let mut layer_depth = 0;
    let mut clip_depth = 0;
    let mut paint_transform = Affine::IDENTITY;
    for command in &scene.commands {
        match command {
            Command::Draw(draw) => replay_draw(draw, renderer, images, &mut paint_transform),
            Command::PushLayer(layer) if layer_depth < MAX_LAYER_DEPTH => {
                match layer {
                    LayerKind::Clip(shape) => renderer.push_clip_layer(&shape.to_path()),
                    LayerKind::Opacity(opacity) => {
                        renderer.push_opacity_layer(LayerKind::opacity(*opacity));
                    }
                    LayerKind::Blend(mix) => renderer.push_blend_layer(mix.to_blend_mode()),
                }
                layer_depth += 1;
            }
            Command::PopLayer if layer_depth > 0 => {
                renderer.pop_layer();
                layer_depth -= 1;
            }
            Command::PushClip(clip) if clip_depth < MAX_CLIP_DEPTH => {
                replay_clip(clip, renderer);
                clip_depth += 1;
            }
            Command::PopClip if clip_depth > 0 => {
                renderer.pop_clip_path();
                clip_depth -= 1;
            }
            Command::PushLayer(_) | Command::PopLayer | Command::PushClip(_) | Command::PopClip => {
            }
        }
    }
    for _ in 0..layer_depth {
        renderer.pop_layer();
    }
}

fn replay_clip(clip: &ClipCommand, renderer: &mut impl Renderer) {
    renderer.set_transform(clip.transform.to_affine());
    renderer.set_fill_rule(clip.fill_rule.to_fill());
    match &clip.shape {
        FuzzShape::Rect(rect) => renderer.push_clip_rect(&rect.to_rect()),
        FuzzShape::Path(path) => renderer.push_clip_path(&path.to_path()),
    }
}

/// `paint_transform` mirrors the renderer's current paint transform so it is only set on change.
fn replay_draw(
    draw: &DrawCommand,
    renderer: &mut impl Renderer,
    images: &ImageTable,
    paint_transform: &mut Affine,
) {
    renderer.set_transform(draw.transform.to_affine());
    if draw.paint.uses_paint_transform() {
        let new_paint_transform = draw.paint_transform.to_affine();
        if new_paint_transform != *paint_transform {
            renderer.set_paint_transform(new_paint_transform);
            *paint_transform = new_paint_transform;
        }
    }
    renderer.set_paint(draw.paint.to_paint(images));
    match &draw.style {
        DrawStyle::Fill(fill_rule) => {
            renderer.set_fill_rule(fill_rule.to_fill());
            match &draw.shape {
                FuzzShape::Rect(rect) => renderer.fill_rect(&rect.to_rect()),
                FuzzShape::Path(path) => renderer.fill_path(&path.to_path()),
            }
        }
        DrawStyle::Stroke(stroke) => {
            renderer.set_stroke(stroke.to_stroke());
            match &draw.shape {
                FuzzShape::Rect(rect) => renderer.stroke_rect(&rect.to_rect()),
                FuzzShape::Path(path) => renderer.stroke_path(&path.to_path()),
            }
        }
    }
}
