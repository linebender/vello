// Copyright 2026 the Vello Authors
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
use vello_common::paint::PaintType;

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

impl From<AtlasPaint> for PaintType {
    fn from(paint: AtlasPaint) -> Self {
        match paint {
            AtlasPaint::Solid(color) => Self::Solid(color),
            AtlasPaint::Gradient(gradient) => Self::Gradient(gradient),
        }
    }
}

/// A single draw command recorded for deferred atlas rendering.
///
/// The variants correspond 1:1 to the methods on [`DrawSink`].
///
/// [`DrawSink`]: crate::interface::DrawSink
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
/// (`RenderContext`, `Scene`). It also implements [`DrawSink`] so
/// that [`ColrPainter`] can write into it directly.
///
/// [`DrawSink`]: crate::interface::DrawSink
/// [`ColrPainter`]: crate::colr::ColrPainter
pub struct AtlasCommandRecorder {
    /// Which atlas page these commands target.
    pub page_index: u32,
    /// The recorded commands.
    pub commands: Vec<AtlasCommand>,
    /// Width of the glyph renderer / atlas page (pixels).
    pub(crate) width: u16,
    /// Height of the glyph renderer / atlas page (pixels).
    pub(crate) height: u16,
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
