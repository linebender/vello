// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::Any;

use peniko::kurbo::{Affine, BezPath, Rect, Stroke};
use peniko::{BlendMode, Brush, Color, Fill, ImageBrush};

use crate::exact::ExactPathElements;
use crate::scene::Scene;
use crate::texture::TextureId;

/// The brush type used for most painting operations in Vello API.
pub type StandardBrush = Brush<ImageBrush<TextureId>>;

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
/// [`Renderer::create_scene`]: todo
/// [`set_blurred_rounded_rect_brush`]: PaintScene::set_blurred_rounded_rect_brush
/// [`set_brush`]: PaintScene::set_brush
/// [`set_solid_brush`]: PaintScene::set_solid_brush
/// [`fill_path`]: PaintScene::fill_path
/// [`append`]: PaintScene::append
/// [`Renderer`]: todo
pub trait PaintScene: Any {
    /// Insert the contents of `Scene` into the drawing sequence at this point, with the 2d affine
    /// `transform` applied to its contents.
    ///
    /// This will change the currently active brush, so you must reset the brush after calling this method.
    /// (TODO: Change the brush design, because that's stupid!)
    ///
    /// # Errors
    ///
    /// This returns an error if:
    /// - `scene` is hinted, but `self` cannot guarantee that the hinting properties would be maintained.
    ///   Hinted scenes can be appended to the final, renderer-specific, implementations of `PaintScene`
    ///   or to other hinted `Scene`s.
    /// - `scene` is hinted, but the transform would remove the hinting property.
    ///   For hinted scenes, the only valid transforms are integer translations.
    /// - `scene` does not apply to the same renderer as `self`.
    /// - `scene` has unbalanced layers (TODO: This isn't implemented yet).
    // TODO: The reason this method doesn't have a default impl is because of future support for custom paint commands.
    // However, it might be possible to also implement those directly in a generic impl.
    fn append(&mut self, transform: Affine, scene: &Scene) -> Result<(), ()>;

    /// Fill the interior of `shape` with the current brush.
    ///
    /// The shape may contain multiple subpaths (e.g., for shapes with holes like a donut).
    /// Each subpath is implicitly closed if needed with a straight line.
    /// For self-intersecting or nested subpaths, the `fill_rule` determines which areas are considered interior.
    ///
    /// Both `shape` and the current brush will be transformed using the 2d affine `transform`.
    /// See the documentation on [`Fill`]'s variants for details on when you might choose a given fill rule.
    // It would be really nice to have images of nested paths here, to explain winding numbers.
    fn fill_path(&mut self, transform: Affine, fill_rule: Fill, path: &impl ExactPathElements);

    /// Stroke along `path` with the current brush, following the given stroke parameters.
    ///
    /// Both the stroked area and the current brush will be transformed using the 2d affine `transform`.
    /// Dashes configured in the stroke parameter will be expanded.
    fn stroke_path(
        &mut self,
        transform: Affine,
        stroke_params: &Stroke,
        path: &impl ExactPathElements,
    );

    /// Set the current brush to `brush`.
    ///
    /// This method is used to set the brush for images and gradients.
    /// The `paint_transform` will be applied to only the brush contents, in
    /// addition to the transform applied to the object.
    /// For solid colors, this has no effect, and so you may prefer [`set_solid_brush`](PaintScene::set_solid_brush).
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

    /// Set the current brush to represent a rounded rectangle blurred with an approximate gaussian filter.
    ///
    /// **This method is currently unimplemented on all backends, so should not be used.**
    ///
    /// For performance reasons, shapes drawn with this brush should not extend more than approximately
    /// 2.5 times `std_dev` away from the edges of `rect` (as any such points will not be perceptably
    /// painted to, but calculations will still be performed for them).
    ///
    /// This method effectively draws the blurred rounded rectangle clipped to the
    /// shapes drawn with this brush.
    /// This clipping is useful for drawing box shadows, where the shadow should only
    /// be drawn outside the box.
    /// If just the blurred rounded rectangle is desired, without any clipping,
    /// use the simpler [`fill_blurred_rounded_rect`][PaintScene::fill_blurred_rounded_rect].
    /// For many users, that method will be easier to use.
    ///
    /// For details on the algorithm used, see the 2020 Blog Post describing the technique,
    /// [*Blurred rounded rectangles*](https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html).
    fn set_blurred_rounded_rect_brush(
        &mut self,
        // transform: Affine,
        paint_transform: Affine,
        color: Color,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
    );

    /// Set the current brush to a solid `color`.
    fn set_solid_brush(&mut self, color: Color) {
        // The transform doesn't matter for a solid color brush.
        self.set_brush(Brush::Solid(color), Affine::IDENTITY);
    }

    /// Draw a rounded rectangle blurred with an approximate gaussian filter.
    /// This method resets the current brush.
    ///
    /// **This method is currently unimplemented on some backends, so should not be used.**
    ///
    /// If the rounded rectangle needs to be clipped, you can instead use
    /// [`set_blurred_rounded_rect_brush`](PaintScene::set_blurred_rounded_rect_brush).
    ///
    /// The drawing is cut off 2.5 times `std_dev` away from the edges of `rect` for performance
    /// reasons, as any points outside of that area would not be perceptably painted to.
    /// <!-- Fun maths thing: I believe we can calculate exactly the amount of loss due to this approximation. -->
    ///
    /// For details on the algorithm used, see the 2020 Blog Post describing the technique,
    /// [*Blurred rounded rectangles*](https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html).
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
        self.fill_path(transform, Fill::EvenOdd, &shape);
    }

    /// Pushes a new layer clipped by the specified shape (if provided) and composed with
    /// previous layers using the specified blend mode.
    ///
    /// <!-- TODO: The `clip_style` controls how the `clip` shape is interpreted.
    ///
    /// - Use [`Fill`] to clip to the interior of the shape, with the chosen fill rule.
    /// - Use [`Stroke`] (via `&Stroke`) to clip to the stroked outline of the shape.
    /// -->
    /// Every drawing command after this call will be clipped by the shape
    /// until the layer is [popped](PaintScene::pop_layer).
    /// For layers which are only added for clipping, you should
    /// use [`push_clip_layer`](PaintScene::push_clip_layer) instead.
    ///
    /// Opacity must be between 0.0 and 1.0, if provided.
    /// Layers with an opacity of zero may be optimised out, but this is not currently guaranteed.
    ///
    /// **However, the transforms are *not* saved or modified by the layer stack.**
    /// That is, the `transform` argument to this function only applies a transform to the `clip` shape.
    fn push_layer(
        &mut self,
        clip_transform: Affine,
        clip_path: Option<&impl ExactPathElements>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
    );

    /// Pushes a new layer clipped by the specified `clip` shape.
    ///
    /// <!-- TODO: Determine and document isolation properties. -->
    /// <!-- TODO: clip_style as above -->
    /// Every drawing command after this call will be clipped by the shape
    /// until the layer is [popped](PaintScene::pop_layer).
    ///
    /// **However, the transforms are *not* saved or modified by the layer stack.**
    /// That is, the `transform` argument to this function only applies a transform to the `clip` shape.
    fn push_clip_layer(&mut self, clip_transform: Affine, path: &impl ExactPathElements) {
        self.push_layer(clip_transform, Some(path), None, None);
    }

    /// Pushes a new layer which is not clipped and composed with previous layers using the given blend mode.
    ///
    /// Every drawing command after this call will be composed as described
    /// until the layer is [popped](PaintScene::pop_layer).
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(
            Affine::IDENTITY,
            // This needing to be turbofished shows a real footgun with the proposed "push_layer" API here.
            None::<&BezPath>,
            Some(blend_mode),
            None,
        );
    }

    /// Pushes a new layer which is not clipped and composed with previous layers with its opacity
    /// multiplied by the given value.
    ///
    /// Opacity must be between 0.0 and 1.0.
    /// Layers with an opacity of zero may be optimised out, but this is not currently guaranteed.
    ///
    /// Every drawing command after this call will be composed as described
    /// until the layer is [popped](PaintScene::pop_layer).
    fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(Affine::IDENTITY, None::<&BezPath>, None, Some(opacity));
    }

    /// Pop the most recently pushed layer.
    ///
    /// All open layers in a [`Scene`] must be popped before [appending](PaintScene::append)
    /// it to another scene.
    fn pop_layer(&mut self);
}
