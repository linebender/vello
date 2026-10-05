// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `Tiler.h` (`Tiles::makeTilesMSAA` and helpers), at 16×16 tiles.
//!
//! OWNER: tiler worker. Frozen interface (do not change signatures without coordinating):
//!
//! ```ignore
//! pub struct Tiles { .. }
//! impl Tiles {
//!     pub fn new() -> Self;
//!     pub fn reset(&mut self);
//!     /// Returns true if a culling event occurred (lines left of `clip` wrote to `histogram`).
//!     pub fn make_tiles_msaa(&mut self, lines: &[Line], clip: IRect, histogram: &mut WindingHistogram) -> bool;
//!     pub fn sort_tiles(&mut self);
//!     pub fn tiles(&self) -> &[Tile];
//! }
//! ```

#![allow(dead_code, reason = "Being filled in by the port")]

use super::tile::{IRect, Line, Tile, WindingHistogram};

/// Tiles touched by a path's lines.
#[derive(Debug, Default)]
pub(super) struct Tiles {
    tiles: Vec<Tile>,
}

impl Tiles {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn reset(&mut self) {
        self.tiles.clear();
    }

    /// Port of `Tiles::makeTilesMSAA(polyline, clip, histogram)`.
    pub(super) fn make_tiles_msaa(
        &mut self,
        _lines: &[Line],
        _clip: IRect,
        _histogram: &mut WindingHistogram,
    ) -> bool {
        unimplemented!("tiler port pending")
    }

    pub(super) fn sort_tiles(&mut self) {
        self.tiles.sort_unstable_by_key(|t| t.to_bits());
    }

    pub(super) fn tiles(&self) -> &[Tile] {
        &self.tiles
    }
}
