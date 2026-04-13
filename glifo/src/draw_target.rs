// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Unified 2D drawing surface traits for glyph rendering.
//!
//! [`DrawTarget`] is the minimal object-safe trait used by COLR painting and
//! atlas command replay.
//!
//! [`Renderer`] extends it with state management, image painting, tints, and
//! atlas addressing — everything needed so that glifo can own all glyph
//! rendering orchestration without backend-specific marker types.

use crate::atlas::AtlasSlot;
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Affine, BezPath, Rect};
use crate::peniko::{BlendMode, Gradient};
use vello_common::paint::{Image, ImageSource, Tint};

/// Minimal 2D drawing surface used by COLR painting and atlas command replay.
///
/// This trait is **object-safe** (no associated types) so that [`ColrPainter`]
/// can store `&mut dyn DrawTarget`.
///
/// [`ColrPainter`]: crate::colr::ColrPainter
pub trait DrawTarget {
    /// Set the current transform.
    fn set_transform(&mut self, t: Affine);
    /// Set the current paint to a solid colour.
    fn set_paint_solid(&mut self, color: AlphaColor<Srgb>);
    /// Set the current paint to a gradient.
    fn set_paint_gradient(&mut self, gradient: Gradient);
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
    /// The full surface rectangle (used by COLR fills that cover the entire area).
    fn surface_rect(&self) -> Rect;

    /// Fill the entire surface with a solid colour.
    ///
    /// COLR compositing relies on clip layers to restrict the painted region,
    /// so fills always cover the full surface.
    fn fill_solid(&mut self, color: AlphaColor<Srgb>) {
        let rect = self.surface_rect();
        self.set_paint_solid(color);
        self.fill_rect(&rect);
    }

    /// Fill the entire surface with a gradient.
    ///
    /// COLR compositing relies on clip layers to restrict the painted region,
    /// so fills always cover the full surface.
    fn fill_gradient(&mut self, gradient: Gradient) {
        let rect = self.surface_rect();
        self.set_paint_gradient(gradient);
        self.fill_rect(&rect);
    }
}

/// Full rendering surface for glyph rendering.
///
/// Extends [`DrawTarget`] with state save/restore, stroke support, image
/// painting, tints, and atlas-specific addressing. Backends implement this
/// trait and glifo handles all glyph rendering orchestration generically.
pub trait Renderer: DrawTarget {
    /// Opaque saved state returned by [`save_state`](Self::save_state).
    type SavedState;

    /// Save a snapshot of the current rendering state (transform, paint, etc.).
    fn save_state(&mut self) -> Self::SavedState;

    /// Restore a previously saved rendering state.
    fn restore_state(&mut self, state: Self::SavedState);

    /// Stroke a path with the current paint and stroke settings.
    fn stroke_path(&mut self, path: &BezPath);

    /// Set the current paint to an image.
    fn set_paint_image(&mut self, image: Image);

    /// Set the tint for subsequent image draws.
    fn set_tint(&mut self, tint: Option<Tint>);

    /// Get the context color from the renderer's current paint.
    ///
    /// Used by COLR glyphs to determine the foreground colour for palette
    /// index `0xFFFF`. Returns black if the paint is not a solid colour.
    fn get_context_color(&self) -> AlphaColor<Srgb>;

    /// Construct the [`ImageSource`] for sampling a cached glyph from the atlas.
    ///
    /// CPU backends return a page-level image ID; GPU backends return the
    /// per-allocation image ID assigned by `ImageCache`.
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource;

    /// Compute the paint transform for sampling a cached glyph from the atlas.
    ///
    /// CPU backends translate by the slot's page-level offset; GPU backends
    /// translate by the glyph padding only (the per-allocation offset is
    /// already resolved by the image cache).
    fn atlas_paint_transform(&self, atlas_slot: &AtlasSlot) -> Affine;
}
