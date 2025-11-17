// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use peniko::{
    BlendMode, Color, Fill,
    kurbo::{Affine, BezPath, Rect, Stroke},
};

use crate::prepared::{PreparedPathIndex, PreparedPaths};

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
    fn fill_path(&mut self, path: &BezPath);
    fn stroke_path(&mut self, path: &BezPath);
    // fn fill_rect(&mut self, rect: &Rect);
    // TODO: Rework:
    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32);
    // fn stroke_rect(&mut self, rect: &Rect);
    fn draw_prepared_path(
        &mut self,
        path_set: PreparedPaths,
        index: PreparedPathIndex,
        x_offset: i32,
        y_offset: i32,
    );
    // TODO: Something using prepared glyphs
    // fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self::GlyphRenderer>;
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    );
    // TODO: Semantics?
    fn flush(&mut self);
    // TODO: Why are there so many kinds of layers?
    // fn push_clip_layer(&mut self, path: &BezPath);
    // fn push_blend_layer(&mut self, blend_mode: BlendMode);
    // fn push_opacity_layer(&mut self, opacity: f32);
    // fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);

    fn set_stroke(&mut self, stroke: Stroke);
    // fn set_paint(&mut self, paint: impl Into<PaintType>);
    // fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    // This can be accessed through downcasting:
    // fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    // TODO: What does this mean?
    fn set_blend_mode(&mut self, blend_mode: BlendMode);
    fn width(&self) -> u16;
    fn height(&self) -> u16;
}
