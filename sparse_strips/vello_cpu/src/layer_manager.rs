// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer management for filter effects rendering.
//!
//! This module provides infrastructure for managing persistent layer buffers
//! that can be rendered to independently and then composited together. This is
//! necessary for spatial filter effects that require access to a fully-rendered
//! layer (e.g., Gaussian blur).
//!
//! Layers are structured as collections of wide tiles, matching the organization
//! used by the Fine rasterizer's blend buffer. Each layer maintains its own
//! bounding box in wide tile coordinates, allowing efficient memory usage for
//! layers that don't span the entire render target.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::fine::{Numeric, SCRATCH_BUF_SIZE, ScratchBuf};
use vello_common::coarse::Bbox;

/// Manages persistent layer storage for filter effects.
///
/// Each layer is allocated as a `Vec` of `ScratchBuf`, one per wide tile.
/// This matches the structure of `blend_buf` in the Fine rasterizer.
#[derive(Debug)]
pub(crate) struct LayerManager<T: Numeric> {
    /// Map of layer ID to (layer buffers, wtile bounding box)
    layers: BTreeMap<u32, (Vec<ScratchBuf<T>>, Bbox)>,
    /// Next available layer ID
    next_id: u32,
}

impl<T: Numeric> LayerManager<T> {
    /// Create a new, empty layer manager.
    ///
    /// Layers can be allocated on-demand using [`allocate_layer_with_id`](Self::allocate_layer_with_id).
    pub(crate) fn new() -> Self {
        Self {
            layers: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Allocate a layer buffer with a specific ID and wide tile bounding box.
    ///
    /// The layer is initialized with zero values (transparent for u8, 0.0 for f32).
    /// If a layer with this ID already exists, this does nothing.
    ///
    /// # Parameters
    /// - `layer_id`: A unique identifier for this layer
    /// - `wtile_bbox`: The bounding box of the layer in wide tile coordinates. Only tiles
    ///   within this bbox will be allocated, reducing memory usage for partial layers.
    pub(crate) fn allocate_layer_with_id(&mut self, layer_id: u32, wtile_bbox: Bbox) {
        if self.layers.contains_key(&layer_id) {
            return;
        }

        // Calculate number of tiles needed for this bbox
        let tiles_needed = usize::from(wtile_bbox.width()) * usize::from(wtile_bbox.height());

        // Allocate one ScratchBuf per wide tile within the bbox
        let layer = vec![[T::ZERO; SCRATCH_BUF_SIZE]; tiles_needed];

        self.layers.insert(layer_id, (layer, wtile_bbox));

        // Update next_id if necessary
        if layer_id >= self.next_id {
            self.next_id = layer_id + 1;
        }
    }

    /// Get a mutable reference to a layer's `Vec` of `ScratchBuf`.
    ///
    /// This provides access to all tiles in the layer for operations that need
    /// to process multiple tiles at once (e.g., applying spatial filters).
    ///
    /// Returns [`None`] if the layer doesn't exist.
    pub(crate) fn get_layer_mut(&mut self, layer_id: u32) -> Option<&mut Vec<ScratchBuf<T>>> {
        self.layers.get_mut(&layer_id).map(|(layer, _)| layer)
    }

    /// Get a mutable reference to a specific wide tile's `ScratchBuf` within a layer.
    ///
    /// This is used when rendering individual tiles directly to a layer or when
    /// compositing from one layer to another on a per-tile basis.
    ///
    /// Returns [`None`] if the layer doesn't exist or if the tile position is out of bounds
    /// relative to the layer's bounding box.
    ///
    /// # Parameters
    /// - `layer_id`: The ID of the layer to access
    /// - `tile_x`: The x-coordinate of the wide tile in global wide tile coordinates
    /// - `tile_y`: The y-coordinate of the wide tile in global wide tile coordinates
    pub(crate) fn get_layer_tile_mut(
        &mut self,
        layer_id: u32,
        tile_x: u16,
        tile_y: u16,
    ) -> Option<&mut ScratchBuf<T>> {
        let (layer, bbox) = self.layers.get_mut(&layer_id)?;

        // Check if tile position is within the layer's bounding box
        if !bbox.contains(tile_x, tile_y) {
            return None;
        }

        // Convert global tile coordinates to local (bbox-relative) coordinates
        let local_x = tile_x - bbox.x0();
        let local_y = tile_y - bbox.y0();

        // Calculate the index using the layer's local dimensions
        let layer_width = usize::from(bbox.width());
        let tile_index = usize::from(local_y) * layer_width + usize::from(local_x);

        layer.get_mut(tile_index)
    }
}
