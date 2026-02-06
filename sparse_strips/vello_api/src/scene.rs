// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Implementation types for [`Scene`], a reusable sequence of drawing commands.
//!
//! The types in this module (aside from `Scene`) are intended to be used by implementations of Vello API.
//!
//! Consumers of Vello API should interact with `Scene` by using the methods from its implementation of [`PaintScene`].
//! These completed scenes can then be applied to renderer-specific drawing types using [`PaintScene::append`].

use alloc::vec::Vec;

use peniko::{
    BlendMode,
    kurbo::{self, Affine},
};

use crate::{
    PaintScene, StandardBrush,
    exact::ExactPathElements,
    paths::{PathId, PathSet},
};

#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// A single render command in a `Scene`. Each [`PaintScene`] method on the scene adds one of these.
///
/// The [`PathId`]s contained within are the index into the pathset associated with this `Scene`.
/// As such, when moving these commands between scenes, the path id must be updated.
/// (N.B. this will be less true when we get path caching, if it follows the expected design).
///
/// The abstract renderer these commands operate on has the state described in the [`PaintScene`] trait; that is, the current brush and a layer stack.
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
    SetPaint(Affine, StandardBrush),
    /// Set the paint to be a blurred rounded rectangle.
    ///
    /// This is useful for box shadows.
    BlurredRoundedRectPaint(BlurredRoundedRectBrush),
}

/// Command for pushing a new layer.
#[derive(Debug, Clone)]
pub struct PushLayerCommand {
    /// The transform which will be applied to `clip_path`.
    pub clip_transform: Affine,
    /// Clip path.
    pub clip_path: Option<PathId>,
    /// Blend mode.
    pub blend_mode: Option<BlendMode>,
    /// Opacity.
    pub opacity: Option<f32>,
}

/// Command for setting the brush to be a blurred rounded rectangle.
#[derive(Debug, Clone)]
pub struct BlurredRoundedRectBrush {
    /// The transform which will be applied to the rectangle to be drawn.
    /// This applies after the path transform.
    pub paint_transform: Affine,
    /// The color of the rectangle.
    pub color: peniko::Color,
    /// The rectangle before transformation.
    pub rect: kurbo::Rect,
    /// The corner radius.
    pub radius: f32,
    /// The standard deviation of the blur.
    /// Higher means a "more blurred" rectangle
    pub std_dev: f32,
}

/// A reusable sequence of drawing commands for renders to a specific renderer.
///
/// # Hinting
///
/// A `Scene` can optionally be marked as "hinted".
/// This means that the units of any input drawing operations will always fall
/// on the physical pixel grid when the content is rendered.
/// This is especially useful for improving the clarity of text, but it also
/// allows drawing single-pixel lines for user interfaces.
/// This hinting property allows improved performance through caching
/// of intermediate rendering stages.
///
/// However, this does mean that more advanced transformations are not possible on hinted scenes.
///
/// # Usage
///
/// This type has public fields as an interim basis.
/// To use this type, you should use the methods on its implementation of [`PaintScene`] instead.
#[derive(Debug)]
// TODO: Reason about visibility; do we want a more limited "expose" operation
// which lets you read all the fields?
// This also applies to `PathSet`
pub struct Scene {
    /// The paths in this Scene.
    pub paths: PathSet,
    /// The ordered sequence of render commands.
    pub commands: Vec<RenderCommand>,
    /// Whether this `Scene` is hinted.
    pub hinted: bool,
}

impl Scene {
    /// Create a new reusable `Scene`.
    ///
    /// If `hinted` is true, this Scene is hinted.
    /// See the documentation on this type for details of what that means.
    pub fn new(hinted: bool) -> Self {
        Self {
            paths: PathSet::new(),
            commands: Vec::new(),
            hinted,
        }
    }

    /// Removes all content from this `Scene`.
    ///
    /// Does not reset the hinted value.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.paths.clear();
    }

    /// Returns true if this `Scene` is hinted.
    ///
    /// See the type level documentation for more information.
    pub fn hinted(&self) -> bool {
        self.hinted
    }
}

/// Extract the translation component from a 2d Affine transformation, if the
/// transform is equivalent to an exact integer translation.
///
/// Returns the x and y translations on success.
///
/// This method assumes that the translations will be at a reasonable scale for 2d rendering.
/// Whilst we don't validate this precisely, values less than ~10 billion pixels should be fine.
pub fn extract_integer_translation(transform: Affine) -> Option<(f64, f64)> {
    fn is_nearly(a: f64, b: f64) -> bool {
        // TODO: This is a very arbitrary threshold.
        // It's valid for it to be as high as it is, because this is in units of a pixel,
        // so 1/100th of a pixel is negligible.
        (a - b).abs() < 0.01
    }
    let [a, b, c, d, dx, dy] = transform.as_coeffs();
    // If there's a skew, rotation or scale, then the transform is not compatible with hinting.
    if !(is_nearly(a, 1.0) && is_nearly(b, 0.0) && is_nearly(c, 0.0) && is_nearly(d, 1.0)) {
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
            paths: other_paths,
            commands: other_commands,
            hinted: other_hinted,
        }: &Scene,
    ) -> Result<(), ()> {
        // if !Arc::ptr_eq(&self.renderer, other_renderer) {
        //     // Mismatched Renderers
        //     return Err(());
        // }

        if *other_hinted {
            if !self.hinted {
                // Trying to bring a "hinted" scene into an unhinted context.
                return Err(());
            }
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

        Ok(())
    }

    fn fill_path(
        &mut self,
        transform: Affine,
        fill_rule: peniko::Fill,
        path: &impl ExactPathElements,
    ) {
        let idx = self.paths.prepare_shape(&path, fill_rule);
        self.commands.push(RenderCommand::DrawPath(transform, idx));
    }

    fn stroke_path(
        &mut self,
        transform: Affine,
        stroke_params: &kurbo::Stroke,
        path: &impl ExactPathElements,
    ) {
        let idx = self.paths.prepare_shape(&path, stroke_params.clone());
        self.commands.push(RenderCommand::DrawPath(transform, idx));
    }

    fn set_brush(&mut self, brush: impl Into<StandardBrush>, paint_transform: Affine) {
        let brush = brush.into();
        self.commands
            .push(RenderCommand::SetPaint(paint_transform, brush));
    }

    fn set_blurred_rounded_rect_brush(
        &mut self,
        paint_transform: Affine,
        color: peniko::Color,
        rect: &kurbo::Rect,
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
        clip_transform: Affine,
        clip_path: Option<&impl ExactPathElements>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        // mask: Option<Mask>,
    ) {
        let clip_idx = if let Some(clip_path) = clip_path {
            Some(self.paths.prepare_shape(
                &clip_path,
                // TODO: Make this configurable for clip paths.
                peniko::Fill::NonZero,
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

#[cfg(test)]
mod test {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

    use peniko::kurbo::Affine;

    use crate::scene::extract_integer_translation;

    #[test]
    fn integer_translations() {
        let coords = [
            (10., 10.),
            (1.00001, 1.),
            (0.99999998, 1.),
            (0.0000001, 0.),
            (10_000., 10_000.),
        ];
        for (real_x, rounded_x) in coords {
            for (real_y, rounded_y) in coords {
                let xform = Affine::translate((real_x, real_y));
                let (extracted_x, extracted_y) = extract_integer_translation(xform)
                    .expect("Passed coordinates are all near integers.");
                assert_eq!(extracted_x, rounded_x);
                assert_eq!(extracted_y, rounded_y);
            }
        }
    }

    #[test]
    fn unhintable_transforms() {
        let transforms = [
            Affine::translate((10.5, 0.)),
            Affine::skew(1.0, 0.5),
            // Technically, PI/2 and PI *could* be hinted, but they can't reuse the cached strips.
            Affine::rotate(FRAC_PI_2),
            Affine::rotate(PI),
            Affine::rotate(FRAC_PI_3),
        ];
        for xform in transforms {
            assert!(
                extract_integer_translation(xform).is_none(),
                "{xform:?} unexpectedly was treated as hintable."
            );
        }
    }
}
