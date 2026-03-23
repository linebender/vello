// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas slot and rasterization data structures.

use vello_common::paint::ImageId;

/// Location and metrics of a cached glyph within an atlas page.
///
/// One slot is stored per distinct (font, glyph ID, size, subpixel offset)
/// combination in [`GlyphAtlas`](super::GlyphAtlas).
#[derive(Clone, Copy, Debug)]
pub struct AtlasSlot {
    /// The image ID for this glyph in the [`ImageCache`].
    ///
    /// Used for deallocation and for looking up the atlas page/offset.
    ///
    /// [`ImageCache`]: vello_common::image_cache::ImageCache
    pub image_id: ImageId,
    /// Which atlas page contains this glyph.
    pub page_index: u32,
    /// X position in atlas (pixels).
    pub x: u16,
    /// Y position in atlas (pixels).
    pub y: u16,
    /// Width of glyph bitmap (pixels).
    pub width: u16,
    /// Height of glyph bitmap (pixels).
    pub height: u16,
    /// Horizontal bearing (offset from glyph origin to left edge of bitmap).
    pub bearing_x: i16,
    /// Vertical bearing (offset from glyph origin to top edge of bitmap).
    pub bearing_y: i16,
}

/// Metadata for a rasterized glyph (no pixel data).
///
/// Returned by rasterization to communicate bitmap dimensions and
/// bearing offsets.
#[derive(Clone, Copy, Debug)]
pub struct RasterMetrics {
    /// Width of the rasterized glyph (pixels).
    pub width: u16,
    /// Height of the rasterized glyph (pixels).
    pub height: u16,
    /// Horizontal bearing (offset from glyph origin to left edge of bitmap).
    pub bearing_x: i16,
    /// Vertical bearing (offset from glyph origin to top edge of bitmap).
    pub bearing_y: i16,
}
