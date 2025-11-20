// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer management for filter effects rendering.
//!
//! This module provides infrastructure for managing persistent layer buffers
//! that can be rendered to independently and then composited together. This is
//! necessary for spatial filter effects that require access to a fully-rendered
//! layer (e.g., Gaussian blur).
//!
//! Layers are stored as Pixmap instances (row-major, u8 RGBA format), which
//! allows efficient filter operations without layout conversion overhead.
//! Each layer maintains its own bounding box in wide tile coordinates, allowing
//! efficient memory usage for layers that don't span the entire render target.

use crate::region::Region;
use hashbrown::HashMap;
use vello_common::coarse::WideTilesBbox;
use vello_common::pixmap::Pixmap;

/// Manages persistent layer storage for filter effects.
///
/// Each layer is allocated as a `Pixmap` with row-major RGBA8 format.
/// This eliminates conversion overhead for filter operations.
#[derive(Debug)]
pub struct LayerManager {
    /// Map of layer ID to (Pixmap, wtile bounding box).
    /// The Pixmap contains the layer's pixel data, and the Bbox defines which
    /// wide tiles this layer occupies (in wide tile coordinates).
    layers: HashMap<u32, (Pixmap, WideTilesBbox)>,
    /// Next available layer ID for automatic allocation.
    next_id: u32,
    /// Reusable scratch buffer for filter operations that need temporary storage.
    /// Examples include separable convolution passes (e.g., Gaussian blur) or
    /// intermediate compositing results. This buffer is lazily allocated and
    /// automatically resized as needed to avoid repeated allocations.
    scratch_buffer: Option<Pixmap>,
}

impl Default for LayerManager {
    /// Creates a new, empty layer manager with no allocated layers.
    ///
    /// Layer IDs start from 1 (ID 0 is reserved for internal use).
    fn default() -> Self {
        Self {
            layers: HashMap::new(),
            next_id: 1,
            scratch_buffer: None,
        }
    }
}

impl LayerManager {
    /// Create a new, empty layer manager.
    ///
    /// Layers can be registered on-demand using [`register_layer`](Self::register_layer).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a layer with a specific ID, wide tile bounding box, and pixel data.
    ///
    /// The provided `Pixmap` should already contain the layer's pixel data (e.g., the
    /// result of rendering a layer or applying a filter effect).
    ///
    /// If a layer with this ID already exists, this method does nothing (no replacement).
    ///
    /// # Parameters
    /// - `layer_id`: A unique identifier for this layer. Layer ID 0 is reserved for internal use.
    /// - `wtile_bbox`: The bounding box of the layer in wide tile coordinates. This defines
    ///   the region of the layer that contains valid pixel data, enabling efficient memory
    ///   usage for layers that don't span the entire render target.
    /// - `pixmap`: The pixel data for this layer in row-major RGBA8 format.
    pub fn register_layer(&mut self, layer_id: u32, wtile_bbox: WideTilesBbox, pixmap: Pixmap) {
        self.layers.insert(layer_id, (pixmap, wtile_bbox));

        if layer_id >= self.next_id {
            self.next_id = layer_id + 1;
        }
    }

    /// Get a mutable `Region` view into a specific wide tile within a layer.
    ///
    /// This extracts a tile-sized region from the layer's Pixmap, allowing
    /// rendering or compositing operations on individual tiles without copying
    /// pixel data. The returned Region provides a mutable view into the underlying
    /// layer buffer.
    ///
    /// Returns [`None`] if:
    /// - The layer doesn't exist
    /// - The tile position is outside the layer's bounding box
    /// - The tile extraction fails (e.g., invalid coordinates)
    ///
    /// # Parameters
    /// - `layer_id`: The ID of the layer to access
    /// - `tile_x`: The x-coordinate of the wide tile in global wide tile coordinates
    /// - `tile_y`: The y-coordinate of the wide tile in global wide tile coordinates
    ///
    /// # Coordinate Systems
    /// The input coordinates (`tile_x`, `tile_y`) are in *global* wide tile space,
    /// but they are automatically converted to *local* coordinates relative to the
    /// layer's bounding box before extracting the region.
    pub fn layer_tile_region_mut(
        &mut self,
        layer_id: u32,
        tile_x: u16,
        tile_y: u16,
    ) -> Option<Region<'_>> {
        let (pixmap, bbox) = self.layers.get_mut(&layer_id)?;

        // Ensure the requested tile is within the layer's allocated bounds
        if !bbox.contains(tile_x, tile_y) {
            return None;
        }

        // Convert global tile coordinates to layer-local (bbox-relative) coordinates
        let local_x = tile_x - bbox.x0();
        let local_y = tile_y - bbox.y0();

        // Extract a mutable Region view of the tile from the underlying pixmap
        Region::from_pixmap_tile(pixmap, local_x, local_y)
    }

    /// Get or create a scratch buffer of at least the requested dimensions.
    ///
    /// This buffer is reused across filter operations to minimize allocations.
    /// If the existing buffer is large enough, it's reused; otherwise, a new
    /// (larger) buffer is allocated.
    ///
    /// # Parameters
    /// - `width`: Minimum width in pixels
    /// - `height`: Minimum height in pixels
    ///
    /// # Returns
    /// A mutable reference to a `Pixmap` of at least `width × height` pixels.
    /// The actual buffer may be larger than requested if it was previously allocated
    /// with larger dimensions.
    pub fn get_scratch_buffer(&mut self, width: u16, height: u16) -> &mut Pixmap {
        match &mut self.scratch_buffer {
            None => {
                // No buffer exists yet, allocate a new one
                self.scratch_buffer = Some(Pixmap::new(width, height));
            }
            Some(buf) if buf.width() < width || buf.height() < height => {
                // Existing buffer is too small, resize it
                buf.resize(width, height);
            }
            // Buffer is already large enough, reuse it without reallocation
            Some(_) => {}
        }

        self.scratch_buffer.as_mut().unwrap()
    }
}
