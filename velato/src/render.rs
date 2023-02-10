// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{model::*, Composition};

use vello::{
    kurbo::{self, Affine, PathEl, Rect},
    peniko::{self, Fill, Mix},
};

use std::ops::Range;

pub trait RenderSink {
    fn push_layer(
        &mut self,
        blend: impl Into<peniko::BlendMode>,
        alpha: f32,
        transform: Affine,
        shape: &impl kurbo::Shape,
    );

    fn pop_layer(&mut self);

    fn draw(
        &mut self,
        stroke: Option<&fixed::Stroke>,
        transform: Affine,
        brush: &fixed::Brush,
        shape: &impl kurbo::Shape,
    );
}

impl RenderSink for vello::SceneBuilder<'_> {
    fn push_layer(
        &mut self,
        blend: impl Into<peniko::BlendMode>,
        alpha: f32,
        transform: Affine,
        shape: &impl kurbo::Shape,
    ) {
        self.push_layer(blend, alpha, transform, shape);
    }

    fn pop_layer(&mut self) {
        self.pop_layer()
    }

    fn draw<'b>(
        &mut self,
        stroke: Option<&fixed::Stroke>,
        transform: Affine,
        brush: &fixed::Brush,
        shape: &impl kurbo::Shape,
    ) {
        if let Some(stroke) = stroke {
            self.stroke(stroke, transform, brush, None, shape);
        } else {
            self.fill(Fill::NonZero, transform, brush, None, shape);
        }
    }
}

/// Renders a composition into a scene builder.
#[derive(Default)]
pub struct Renderer {
    batch: Batch,
    mask_elements: Vec<PathEl>,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Renders the animation at a given time into the specified scene builder.
    pub fn render(
        &mut self,
        animation: &Composition,
        time: f32,
        transform: Affine,
        alpha: f32,
        sink: &mut impl RenderSink,
    ) {
        let frame = animation.frame_for_time(time);
        self.render_frame(animation, frame, transform, alpha, sink);
    }

    /// Renders the animation at a given frame into the specified scene builder.
    pub fn render_frame(
        &mut self,
        animation: &Composition,
        frame: f32,
        transform: Affine,
        alpha: f32,
        sink: &mut impl RenderSink,
    ) {
        let frame = frame.min(animation.frames.end);
        self.batch.clear();
        sink.push_layer(
            Mix::Clip,
            1.0,
            transform,
            &Rect::new(0.0, 0.0, animation.width as _, animation.height as _),
        );
        for layer in animation.layers.iter().rev() {
            if layer.is_mask {
                continue;
            }
            self.render_layer(
                animation,
                &animation.layers,
                layer,
                transform,
                alpha,
                frame,
                sink,
            );
        }
        sink.pop_layer();
    }

    #[allow(clippy::too_many_arguments)]
    fn render_layer(
        &mut self,
        animation: &Composition,
        layer_set: &[Layer],
        layer: &Layer,
        transform: Affine,
        alpha: f32,
        frame: f32,
        sink: &mut impl RenderSink,
    ) {
        if !layer.frames.contains(&frame) {
            return;
        }
        let clip = false; //layer.width != 0 && layer.height != 0;
        if clip {
            sink.push_layer(
                peniko::Mix::Clip,
                1.0,
                Affine::IDENTITY,
                &kurbo::Rect::from_origin_size(
                    (0.0, 0.0),
                    (layer.width as f64, layer.height as f64),
                ),
            );
        }
        let parent_transform = transform;
        let transform = self.compute_transform(layer_set, layer, parent_transform, frame);
        let full_rect = Rect::new(0.0, 0.0, animation.width as _, animation.height as _);
        if let Some((mode, mask_index)) = layer.mask_layer {
            // Extra layer to isolate blending for the mask
            sink.push_layer(Mix::Normal, 1.0, parent_transform, &full_rect);
            if let Some(mask) = layer_set.get(mask_index) {
                self.render_layer(
                    animation,
                    layer_set,
                    mask,
                    parent_transform,
                    alpha,
                    frame,
                    sink,
                );
            }
            sink.push_layer(mode, 1.0, parent_transform, &full_rect);
        }
        let alpha = alpha * layer.opacity.evaluate(frame) / 100.0;
        for mask in &layer.masks {
            let alpha = mask.opacity.evaluate(frame) / 100.0;
            mask.geometry.evaluate(frame, &mut self.mask_elements);
            sink.push_layer(Mix::Clip, alpha, transform, &self.mask_elements.as_slice());
            self.mask_elements.clear();
        }
        match &layer.content {
            Content::None => {}
            Content::Instance {
                name,
                time_remap: _,
            } => {
                if let Some(asset_layers) = animation.assets.get(name) {
                    let frame_delta = -layer.start_frame;
                    for layer in asset_layers.iter().rev() {
                        if layer.is_mask {
                            continue;
                        }
                        self.render_layer(
                            animation,
                            asset_layers,
                            layer,
                            transform,
                            alpha,
                            frame + frame_delta,
                            sink,
                        );
                    }
                }
            }
            Content::Shape(shapes) => {
                self.render_shapes(shapes, transform, alpha, frame);
                self.batch.render(sink);
                self.batch.clear();
            }
        }
        for _ in 0..layer.masks.len() + clip as usize + (layer.mask_layer.is_some() as usize * 2) {
            sink.pop_layer();
        }
    }

    fn render_shapes(&mut self, shapes: &[Shape], transform: Affine, alpha: f32, frame: f32) {
        // Keep track of our local top of the geometry stack. Any subsequent draws
        // are bounded by this.
        let geometry_start = self.batch.geometries.len();
        // Also keep track of top of draw stack for repeater evaluation.
        let draw_start = self.batch.draws.len();
        // Top to bottom, collect geometries and draws.
        for shape in shapes {
            match shape {
                Shape::Group(shapes, group_transform) => {
                    let (group_transform, group_alpha) =
                        if let Some(GroupTransform { transform, opacity }) = group_transform {
                            (
                                transform.evaluate(frame).to_owned(),
                                opacity.evaluate(frame) / 100.0,
                            )
                        } else {
                            (Affine::IDENTITY, 1.0)
                        };
                    self.render_shapes(
                        shapes,
                        transform * group_transform,
                        alpha * group_alpha,
                        frame,
                    );
                }
                Shape::Geometry(geometry) => {
                    self.batch.push_geometry(geometry, transform, frame);
                }
                Shape::Draw(draw) => {
                    self.batch.push_draw(draw, alpha, geometry_start, frame);
                }
                Shape::Repeater(repeater) => {
                    let repeater = repeater.evaluate(frame);
                    self.batch
                        .repeat(repeater.as_ref(), geometry_start, draw_start);
                }
            }
        }
    }

    /// Computes the transform for a single layer. This currently chases the
    /// full transform chain each time. If it becomes a bottleneck, we can
    /// implement caching.
    fn compute_transform(
        &self,
        layer_set: &[Layer],
        layer: &Layer,
        global_transform: Affine,
        frame: f32,
    ) -> Affine {
        let mut transform = layer.transform.evaluate(frame).to_owned();
        let mut parent_index = layer.parent;
        let mut count = 0usize;
        while let Some(index) = parent_index {
            // We don't check for cycles at import time, so this heuristic prevents
            // infinite loops.
            if count >= layer_set.len() {
                break;
            }
            if let Some(parent) = layer_set.get(index) {
                parent_index = parent.parent;
                transform = parent.transform.evaluate(frame).to_owned() * transform;
                count += 1;
            } else {
                break;
            }
        }
        global_transform * transform
    }
}

#[derive(Clone, Debug)]
struct DrawData {
    stroke: Option<fixed::Stroke>,
    brush: fixed::Brush,
    alpha: f32,
    /// Range into ShapeBatch::geometries
    geometry: Range<usize>,
}

impl DrawData {
    fn new(draw: &Draw, alpha: f32, geometry: Range<usize>, frame: f32) -> Self {
        Self {
            stroke: draw
                .stroke
                .as_ref()
                .map(|stroke| stroke.evaluate(frame).to_owned()),
            brush: draw.brush.evaluate(1.0, frame).to_owned(),
            alpha: alpha * draw.opacity.evaluate(frame) / 100.0,
            geometry,
        }
    }
}

#[derive(Clone, Debug)]
struct GeometryData {
    /// Range into ShapeBatch::elements
    elements: Range<usize>,
    transform: Affine,
}

#[derive(Default)]
struct Batch {
    elements: Vec<PathEl>,
    geometries: Vec<GeometryData>,
    draws: Vec<DrawData>,
    repeat_geometries: Vec<GeometryData>,
    repeat_draws: Vec<DrawData>,
    /// Length of geometries at time of most recent draw. This is
    /// used to prevent merging into already used geometries.
    drawn_geometry: usize,
}

impl Batch {
    fn push_geometry(&mut self, geometry: &Geometry, transform: Affine, frame: f32) {
        // Merge with the previous geometry if possible. There are two conditions:
        // 1. The previous geometry has not yet been referenced by a draw
        // 2. The geometries have the same transform
        if self.drawn_geometry < self.geometries.len()
            && self.geometries.last().map(|last| last.transform) == Some(transform)
        {
            geometry.evaluate(frame, &mut self.elements);
            self.geometries.last_mut().unwrap().elements.end = self.elements.len();
        } else {
            let start = self.elements.len();
            geometry.evaluate(frame, &mut self.elements);
            let end = self.elements.len();
            self.geometries.push(GeometryData {
                elements: start..end,
                transform,
            });
        }
    }

    fn push_draw(&mut self, draw: &Draw, alpha: f32, geometry_start: usize, frame: f32) {
        self.draws.push(DrawData::new(
            draw,
            alpha,
            geometry_start..self.geometries.len(),
            frame,
        ));
        self.drawn_geometry = self.geometries.len();
    }

    fn repeat(&mut self, repeater: &fixed::Repeater, geometry_start: usize, draw_start: usize) {
        // First move the relevant ranges of geometries and draws into side buffers
        self.repeat_geometries
            .extend(self.geometries.drain(geometry_start..));
        self.repeat_draws.extend(self.draws.drain(draw_start..));
        // Next, repeat the geometries and apply the offset transform
        for geometry in self.repeat_geometries.iter() {
            for i in 0..repeater.copies {
                let transform = repeater.transform(i);
                let mut geometry = geometry.clone();
                geometry.transform *= transform;
                self.geometries.push(geometry);
            }
        }
        // Finally, repeat the draws, taking into account opacity and the modified
        // newly repeated geometry ranges
        let start_alpha = repeater.start_opacity / 100.0;
        let end_alpha = repeater.end_opacity / 100.0;
        let delta_alpha = if repeater.copies > 1 {
            // See note in Skottie: AE does not cover the full opacity range
            (end_alpha - start_alpha) / repeater.copies as f32
        } else {
            0.0
        };
        for i in 0..repeater.copies {
            let alpha = start_alpha + delta_alpha * i as f32;
            if alpha <= 0.0 {
                continue;
            }
            for mut draw in self.repeat_draws.iter().cloned() {
                draw.alpha *= alpha;
                let count = draw.geometry.end - draw.geometry.start;
                draw.geometry.start = geometry_start
                    + (draw.geometry.start - geometry_start) * repeater.copies as usize;
                draw.geometry.end = draw.geometry.start + count * repeater.copies as usize;
                self.draws.push(draw);
            }
        }
        // Clear the side buffers
        self.repeat_geometries.clear();
        self.repeat_draws.clear();
        // Prevent merging until new geometries are pushed
        self.drawn_geometry = self.geometries.len();
    }

    fn render(&self, sink: &mut impl RenderSink) {
        // Process all draws in reverse
        for draw in self.draws.iter().rev() {
            // Some nastiness to avoid cloning the brush if unnecessary
            let modified_brush = if draw.alpha != 1.0 {
                Some(fixed::brush_with_alpha(&draw.brush, draw.alpha))
            } else {
                None
            };
            let brush = modified_brush.as_ref().unwrap_or(&draw.brush);
            for geometry in self.geometries[draw.geometry.clone()].iter() {
                let path = &self.elements[geometry.elements.clone()];
                let transform = geometry.transform;
                sink.draw(draw.stroke.as_ref(), transform, brush, &path);
            }
        }
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.geometries.clear();
        self.draws.clear();
        self.repeat_geometries.clear();
        self.repeat_draws.clear();
        self.drawn_geometry = 0;
    }
}
