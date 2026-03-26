// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deferred atlas rendering commands.
//!
//! During glyph encoding, outline and COLR glyph draw commands are recorded
//! into an [`AtlasCommandRecorder`] rather than being executed immediately.
//! At render time the application drains the pending recorders (grouped by
//! atlas page) and replays them into a single glyph renderer that is reset
//! between pages.
//!
//! This approach:
//! - Supports multiple atlas pages (not just page 0)
//! - Keeps a single glyph renderer (same atlas page size)
//! - Mirrors the `drain_pending_uploads` pattern used for bitmap glyphs

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Affine, BezPath, Rect};
use crate::peniko::{BlendMode, Gradient};

/// Paint type for atlas commands.
#[derive(Clone, Debug)]
pub enum AtlasPaint {
    /// A solid colour (used for outlines and COLR solid fills).
    Solid(AlphaColor<Srgb>),
    /// A gradient (used for COLR gradient fills).
    Gradient(Gradient),
}

impl From<AlphaColor<Srgb>> for AtlasPaint {
    fn from(c: AlphaColor<Srgb>) -> Self {
        Self::Solid(c)
    }
}

impl From<Gradient> for AtlasPaint {
    fn from(g: Gradient) -> Self {
        Self::Gradient(g)
    }
}

/// A single draw command recorded for deferred atlas rendering.
///
/// The variants correspond 1:1 to the methods on [`AtlasReplayTarget`].
///
/// [`AtlasReplayTarget`]: crate::renderers::vello_renderer::AtlasReplayTarget
#[derive(Clone, Debug)]
pub enum AtlasCommand {
    /// Set the current transform.
    SetTransform(Affine),
    /// Set the current paint (solid colour or gradient).
    SetPaint(AtlasPaint),
    /// Set the paint transform.
    SetPaintTransform(Affine),
    /// Fill a path with the current paint and transform.
    FillPath(Arc<BezPath>),
    /// Fill a rectangle with the current paint and transform.
    FillRect(Rect),
    /// Push a clip layer defined by a path.
    PushClipLayer(Arc<BezPath>),
    /// Push a blend/compositing layer.
    PushBlendLayer(BlendMode),
    /// Pop the most recent clip or blend layer.
    PopLayer,
}

/// Records atlas draw commands for a single atlas page.
///
/// The recorder exposes the same method API as the actual renderers
/// (`RenderContext`, `Scene`). It also implements [`ColrRenderer`] so
/// that [`ColrPainter`] can write into it directly.
///
/// [`ColrRenderer`]: crate::colr::ColrRenderer
/// [`ColrPainter`]: crate::colr::ColrPainter
pub struct AtlasCommandRecorder {
    /// Which atlas page these commands target.
    pub page_index: u32,
    /// The recorded commands.
    pub commands: Vec<AtlasCommand>,
    /// Width of the glyph renderer / atlas page (pixels).
    width: u16,
    /// Height of the glyph renderer / atlas page (pixels).
    height: u16,
}

impl AtlasCommandRecorder {
    /// Create a new recorder for the given atlas page.
    ///
    /// `width` and `height` must match the glyph renderer dimensions
    /// (i.e. the atlas page size) so that COLR `fill_solid` / `fill_gradient`
    /// produce correctly-sized fill rects.
    pub fn new(page_index: u32, width: u16, height: u16) -> Self {
        Self {
            page_index,
            commands: Vec::new(),
            width,
            height,
        }
    }

    /// Width of the atlas page in pixels.
    #[inline]
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Height of the atlas page in pixels.
    #[inline]
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Set the current transform.
    #[inline]
    pub fn set_transform(&mut self, t: Affine) {
        self.commands.push(AtlasCommand::SetTransform(t));
    }

    /// Set the current paint (accepts `AlphaColor<Srgb>` or `Gradient`).
    #[inline]
    pub fn set_paint(&mut self, paint: impl Into<AtlasPaint>) {
        self.commands.push(AtlasCommand::SetPaint(paint.into()));
    }

    /// Set the paint transform.
    #[inline]
    pub fn set_paint_transform(&mut self, t: Affine) {
        self.commands.push(AtlasCommand::SetPaintTransform(t));
    }

    /// Fill a path with the current paint and transform.
    #[inline]
    pub fn fill_path(&mut self, path: &Arc<BezPath>) {
        self.commands.push(AtlasCommand::FillPath(Arc::clone(path)));
    }

    /// Fill a rectangle with the current paint and transform.
    #[inline]
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.commands.push(AtlasCommand::FillRect(*rect));
    }

    /// Push a clip layer defined by a path.
    ///
    /// Takes ownership of the `BezPath` and wraps it in `Arc`. COLR clip paths
    /// are always freshly constructed, so this avoids any cloning.
    #[inline]
    pub fn push_clip_layer(&mut self, clip: BezPath) {
        self.commands
            .push(AtlasCommand::PushClipLayer(Arc::new(clip)));
    }

    /// Push a blend/compositing layer.
    #[inline]
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.commands.push(AtlasCommand::PushBlendLayer(blend_mode));
    }

    /// Pop the most recent clip or blend layer.
    #[inline]
    pub fn pop_layer(&mut self) {
        self.commands.push(AtlasCommand::PopLayer);
    }
}

impl core::fmt::Debug for AtlasCommandRecorder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AtlasCommandRecorder")
            .field("page_index", &self.page_index)
            .field("commands", &self.commands.len())
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}
