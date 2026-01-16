// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use peniko::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, Brush, Color, Fill};

use crate::scene::Scene;
use crate::texture::TextureId;

/// The brush type used for most painting operations in Vello API.
pub type StandardBrush = peniko::Brush<peniko::ImageBrush<TextureId>>;

/// A 2d scene or canvas.
///
/// These types are used to prepare a sequence of vector shapes to later be drawn by a [`Renderer`].
/// From a [`Renderer`], you can create a canvas which renders to a given texture using [`Renderer::create_scene`].
/// Alternatively, you can use [`Scene`], a reusable `PaintScene`, for cases such as a single SVG, or a GUI widget.
///
/// The canvas this type represents combines three separate concerns.
/// These are:
///
/// - The brush, which describes what will be drawn by the following commands.
///   This can be a solid colour ([`set_solid_brush`]), a gradient, an image (both [`set_brush`]), or a blurred rounded rectangle ([`set_blurred_rounded_rect_brush`]).
///   The brush's coordinate system is currently relative to the drawn path's coordinate system.
///   <!-- We might also want to allow this to be explicitly unset, in which case drawing will fail. -->
/// - The area over which this brush is to be drawn.
///   This can either be a filled path ([`fill_path`])
/// - The layer stack, which allows for multiple 2d contexts to be blended together, with clipping.
///
/// This separation of brush and drawing area is an experimental aspect of this API.
/// More traditional drawing APIs either combines these concerns into single methods, or splits these
/// out even further (those would be a "stateless" or "stateful" API, respectively).
/// This middle ground is intended to allow drawing multiple shapes with a single brush,
/// so as to allow brush-specific work to be re-used, without having a mechanism like a brush id.
///
/// This design is explicitly an experiment, pending empirical verification.
// In particular, this is likely to fall down for practical drawing of text.
// For example, if you're drawing text with a gradient, something like emoji would end up changing the brush.
// However, for the sake of getting something landed, I'm not planning to change this now.
///
/// # Scene area
///
/// The canvas represented by this trait logically represents an unbounded area.
/// However, from a practical perspective, the area which might be visible depends on the specific scene type.
/// If it will be rendered to a texture (i.e. created using [`Renderer::create_scene`]) then only the texture's viewport is relevant.
/// If this is a [`Scene`], then when it is actually rendered (by being [`append`]ed to a scene created
/// using [`Renderer::create_scene`]) it could be transformed, so any part of the scene could be visible.
/// As such, the inferred clipping from the scene viewports is limited.
///
/// [`Renderer::create_scene`]: crate::Renderer::create_scene
/// [`set_blurred_rounded_rect_brush`]: PaintScene::set_blurred_rounded_rect_brush
/// [`set_brush`]: PaintScene::set_brush
/// [`set_solid_brush`]: PaintScene::set_solid_brush
/// [`fill_path`]: PaintScene::fill_path
/// [`append`]: PaintScene::append
/// [`Renderer`]: crate::Renderer
pub trait PaintScene: Any {
    // Error if associated with different renderer.
    // TODO: This also "clobbers" the brush; we need to document that.
    fn append(&mut self, transform: Affine, scene: &Scene) -> Result<(), ()>;

    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: impl Shape);
    fn stroke_path(&mut self, transform: Affine, stroke_params: &Stroke, path: impl Shape);

    fn set_brush(
        &mut self,
        brush: impl Into<StandardBrush>,
        // We'd like to support both brushes which are "object-local" and brushes which are "scene-local".
        // Image for example you want a gradient to be applied over a whole paragraph of text.
        // The naive solution would need to apply the gradient with a paint transform which undoes the
        // object's transform. That's clearly pretty poor, and we can do better.
        // However, this isn't exposed in the current Vellos, so we're choosing to defer this.
        // transform: Affine,
        paint_transform: Affine,
    );
    fn set_blurred_rounded_rect_brush(
        &mut self,
        // transform: Affine,
        paint_transform: Affine,
        color: Color,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    );
    fn set_solid_brush(&mut self, color: Color) {
        // The transform doesn't matter for a solid color brush.
        self.set_brush(Brush::Solid(color), Affine::IDENTITY);
    }

    fn fill_blurred_rounded_rect(
        &mut self,
        transform: Affine,
        color: Color,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    ) {
        self.set_blurred_rounded_rect_brush(transform, color, rect, radius, std_dev);
        // The impulse response of a gaussian filter is infinite.
        // For performance reason we cut off the filter at some extent where the response is close to zero.
        let kernel_size = (2.5 * std_dev) as f64;

        let shape: Rect = rect.inflate(kernel_size, kernel_size);
        self.fill_path(transform, Fill::EvenOdd, shape);
    }

    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<impl Shape>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
    );

    fn push_clip_layer(&mut self, clip_transform: Affine, path: impl Shape) {
        self.push_layer(clip_transform, Some(path), None, None);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(
            Affine::IDENTITY,
            // This needing to be turbofished shows a real footgun with the proposed "push_layer" API here.
            None::<BezPath>,
            Some(blend_mode),
            None,
        );
    }
    fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(Affine::IDENTITY, None::<BezPath>, None, Some(opacity));
    }

    fn pop_layer(&mut self);
}
