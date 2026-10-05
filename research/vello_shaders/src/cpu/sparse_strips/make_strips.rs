// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `MakeStrips::TraverseCPU` plus the per-path driver from `StripGenerator.cpp`.
//!
//! OWNER: raster worker. Frozen interface: `Rasterizer::new` and `Rasterizer::rasterize_path`.

use super::sink::StripSink;
use super::tile::Line;

/// Per-scene rasterizer state, reused across paths (histogram, tile buffer, processors).
#[derive(Debug)]
pub(super) struct Rasterizer {
    width: u32,
    height: u32,
}

impl Rasterizer {
    pub(super) fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Rasterize one path's lines into `sink`, clipped to the viewport.
    ///
    /// TEMPORARY: delegates to the reference rasterizer until the Skia port lands.
    pub(super) fn rasterize_path(&mut self, lines: &[Line], even_odd: bool, sink: &mut StripSink) {
        super::reference::rasterize_path_reference(lines, even_odd, self.width, self.height, sink);
    }
}
