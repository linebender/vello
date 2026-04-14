// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Traits for rendering glyphs and replaying them.

use crate::atlas::{AtlasPaint, AtlasSlot};
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Affine, BezPath, Rect};
use crate::peniko::BlendMode;
use vello_common::paint::{Image, ImageSource, Tint};

// TODO: This trait is only temporary and will hopefully be replaced once we have a better
// unifying imaging API.
/// A sink for low-level glyph drawing commands.
pub trait DrawSink {
    /// Set the current transform.
    fn set_transform(&mut self, t: Affine);
    /// Set the current paint.
    fn set_paint(&mut self, paint: AtlasPaint);
    /// Set the paint transform.
    fn set_paint_transform(&mut self, t: Affine);
    /// Fill a path with the current paint and transform.
    fn fill_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint and transform.
    fn fill_rect(&mut self, rect: &Rect);
    /// Push a clip layer defined by a path.
    fn push_clip_layer(&mut self, clip: &BezPath);
    /// Push a blend/compositing layer.
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    /// Pop the most recent clip or blend layer.
    fn pop_layer(&mut self);
    /// Width of the surface.
    fn width(&self) -> u16;
    /// Height of the surface.
    fn height(&self) -> u16;
}

/// A stateful renderer that can draw sequences of cached and uncached glyphs.
pub trait GlyphRenderer: DrawSink {
    /// The type of state used by the renderer.
    type SavedState;

    /// Save the current state.
    fn save_state(&mut self) -> Self::SavedState;

    /// Restore the current state.
    fn restore_state(&mut self, state: Self::SavedState);

    /// Stroke a path with the current paint and stroke settings.
    fn stroke_path(&mut self, path: &BezPath);

    /// Set the current paint to an image.
    fn set_paint_image(&mut self, image: Image);

    /// Set the tint for subsequent image draws.
    fn set_tint(&mut self, tint: Option<Tint>);

    /// Get the context color from the renderer's current paint, used for resolving the
    /// context-dependent colors of COLR glyphs.
    fn get_context_color(&self) -> AlphaColor<Srgb>;

    // Hopefully we can get rid of those below in the future.

    /// Construct the [`ImageSource`] for sampling a cached glyph from the atlas.
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource;

    /// Compute the paint transform for sampling a cached glyph from the atlas.
    fn atlas_paint_transform(&self, atlas_slot: &AtlasSlot) -> Affine;
}
