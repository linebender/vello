// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Output sink: turns resolved mask tiles and solid spans into GPU records.

use bytemuck::{Pod, Zeroable};

use super::{KIND_DELTA, MASK_WORDS_PER_TILE, TILE_SIZE};

/// One record of the CPU → GPU contract, consumed by `strip_scatter.wgsl`.
///
/// - Mask tile: `payload & KIND_DELTA == 0`, and `payload` is the mask block index.
///   The block is `masks[payload * MASK_WORDS_PER_TILE..][..MASK_WORDS_PER_TILE]`.
/// - Backdrop delta: `payload & KIND_DELTA != 0`, and the low 16 bits hold an `i16`
///   (`+1` at the first tile of a solid span, `-1` at the tile just past its end).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct StripRecord {
    /// Path index, equal to the draw object index.
    pub path_ix: u32,
    /// `tile_x | (tile_y << 16)`, in 16×16 pixel tiles.
    pub xy: u32,
    /// See the type-level docs.
    pub payload: u32,
}

impl StripRecord {
    #[inline]
    pub fn mask(path_ix: u32, tile_x: u32, tile_y: u32, block: u32) -> Self {
        debug_assert!(block & KIND_DELTA == 0);
        Self {
            path_ix,
            xy: tile_x | (tile_y << 16),
            payload: block,
        }
    }

    #[inline]
    pub fn delta(path_ix: u32, tile_x: u32, tile_y: u32, delta: i16) -> Self {
        Self {
            path_ix,
            xy: tile_x | (tile_y << 16),
            payload: KIND_DELTA | (delta as u16 as u32),
        }
    }

    #[inline]
    pub fn is_delta(&self) -> bool {
        self.payload & KIND_DELTA != 0
    }

    #[inline]
    pub fn tile_x(&self) -> u32 {
        self.xy & 0xffff
    }

    #[inline]
    pub fn tile_y(&self) -> u32 {
        self.xy >> 16
    }

    #[inline]
    pub fn delta_value(&self) -> i16 {
        self.payload as u16 as i16
    }
}

/// Collects the output of strip generation for all paths.
///
/// The rasterizer (the Skia port, or the reference rasterizer) drives it with two calls,
/// mirroring Skia's `TraverseCPU`:
/// - [`StripSink::add_wide`] for a solid interior span (Skia's `WideTiles::addTile`).
/// - [`StripSink::scratch_mut`] then [`StripSink::push_mask_tile`] for an antialiased tile
///   (Skia's `requestAlphaSpace` + `resolveWindingToAlpha`).
#[derive(Debug)]
pub struct StripSink {
    path_ix: u32,
    width_in_tiles: u32,
    /// Output records, in no particular order.
    pub records: Vec<StripRecord>,
    /// Mask blocks, `MASK_WORDS_PER_TILE` words each.
    pub masks: Vec<u32>,
    scratch: [u32; MASK_WORDS_PER_TILE],
}

impl StripSink {
    pub fn new(width_in_tiles: u32) -> Self {
        Self {
            path_ix: 0,
            width_in_tiles,
            records: Vec::new(),
            masks: Vec::new(),
            scratch: [0; MASK_WORDS_PER_TILE],
        }
    }

    /// Set the path that subsequent output belongs to.
    pub fn begin_path(&mut self, path_ix: u32) {
        self.path_ix = path_ix;
    }

    /// Solid interior span covering pixels `[x, x + w)` of the tile row whose top is `y`.
    ///
    /// `x` and `y` are in pixels and must be multiples of [`TILE_SIZE`]. The span end is rounded
    /// up to whole tiles, so a span ending at a ragged viewport edge covers the last tile.
    pub fn add_wide(&mut self, x: u32, y: u32, w: u32) {
        if w == 0 {
            return;
        }
        let ts = TILE_SIZE as u32;
        debug_assert!(
            x.is_multiple_of(ts) && y.is_multiple_of(ts),
            "span not tile aligned: {x},{y}"
        );
        let tx0 = x / ts;
        let tx1 = (x + w).div_ceil(ts);
        self.add_tile_span(tx0, tx1, y / ts);
    }

    /// Solid span covering tiles `[tx0, tx1)` of tile row `ty`.
    pub fn add_tile_span(&mut self, tx0: u32, tx1: u32, ty: u32) {
        if tx0 >= tx1 || tx0 >= self.width_in_tiles {
            return;
        }
        // The closing delta past the last column is never needed.
        let tx1 = tx1.min(self.width_in_tiles);
        // Merge with the previous span if it ends exactly where this one starts.
        if let Some(last) = self.records.last_mut()
            && last.path_ix == self.path_ix
            && last.is_delta()
            && last.delta_value() == -1
            && last.tile_y() == ty
            && last.tile_x() == tx0
        {
            if tx1 < self.width_in_tiles {
                *last = StripRecord::delta(self.path_ix, tx1, ty, -1);
            } else {
                self.records.pop();
            }
            return;
        }
        self.records
            .push(StripRecord::delta(self.path_ix, tx0, ty, 1));
        if tx1 < self.width_in_tiles {
            self.records
                .push(StripRecord::delta(self.path_ix, tx1, ty, -1));
        }
    }

    /// Scratch block for the next mask tile. The caller must overwrite all words.
    ///
    /// Layout: pixel `(x, y)` of the tile is the `u16` at index `y * 16 + x`, i.e. word
    /// `y * 8 + x / 2`, low half for even `x`. Bit `k` is sample `k` (see
    /// [`super::MSAA16_PATTERN`]).
    #[inline]
    pub fn scratch_mut(&mut self) -> &mut [u32; MASK_WORDS_PER_TILE] {
        &mut self.scratch
    }

    /// Commit the scratch block as the mask of tile `(tile_x, tile_y)`.
    ///
    /// Empty tiles are dropped and fully covered tiles become one-tile solid spans.
    pub fn push_mask_tile(&mut self, tile_x: u32, tile_y: u32) {
        if tile_x >= self.width_in_tiles {
            return;
        }
        if self.scratch.iter().all(|&w| w == 0) {
            return;
        }
        if self.scratch.iter().all(|&w| w == u32::MAX) {
            self.add_tile_span(tile_x, tile_x + 1, tile_y);
            return;
        }
        let block = (self.masks.len() / MASK_WORDS_PER_TILE) as u32;
        self.masks.extend_from_slice(&self.scratch);
        self.records
            .push(StripRecord::mask(self.path_ix, tile_x, tile_y, block));
    }
}
