// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use vello_common::coarse::WideTile;
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;

/// A rectangular row-major view into a pixmap.
///
/// Most callers create regions only for the row span they are about to pack or
/// unpack. The region can be narrower than a wide tile and can also be shorter
/// than [`Tile::HEIGHT`] at the bottom edge.
#[derive(Default, Debug)]
pub struct Region<'a> {
    pub width: u16,
    pub height: u16,
    areas: [&'a mut [u8]; Tile::HEIGHT as usize],
}

impl<'a> Region<'a> {
    pub(crate) fn new(
        areas: [&'a mut [u8]; Tile::HEIGHT as usize],
        width: u16,
        height: u16,
    ) -> Self {
        Self {
            areas,
            width,
            height,
        }
    }

    pub(crate) fn from_buffer_at(
        buffer: &'a mut [u8],
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        buffer_width: u16,
    ) -> Option<Self> {
        let row_stride = usize::from(buffer_width) * COLOR_COMPONENTS;
        if row_stride == 0 {
            return None;
        }

        let buffer_height = buffer.len() / row_stride;
        if usize::from(x) >= usize::from(buffer_width) || usize::from(y) >= buffer_height {
            return None;
        }

        let width = width.min(buffer_width - x);
        let height = height.min(u16::try_from(buffer_height - usize::from(y)).unwrap_or(u16::MAX));
        if width == 0 || height == 0 {
            return None;
        }

        let start_offset = usize::from(y) * row_stride + usize::from(x) * COLOR_COMPONENTS;
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut remaining = &mut buffer[start_offset..];
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for area in areas.iter_mut().take(usize::from(height)) {
            let (row, rest) = remaining.split_at_mut(row_width_bytes);
            *area = row;
            let skip = (row_stride - row_width_bytes).min(rest.len());
            remaining = &mut rest[skip..];
        }

        Some(Self::new(areas, width, height))
    }

    /// Extracts a `Region` from a pixmap at the specified tile coordinates.
    ///
    /// The region corresponds to a wide tile area (`WideTile::WIDTH` × `Tile::HEIGHT` pixels),
    /// starting at pixel coordinates `(tile_x * WideTile::WIDTH, tile_y * Tile::HEIGHT)`.
    /// Regions at the right or bottom edges may be smaller if they extend beyond the pixmap bounds.
    ///
    /// Returns `None` if the tile coordinates are completely outside the pixmap bounds.
    ///
    /// # Arguments
    /// * `pixmap` - The pixmap to extract from
    /// * `tile_x` - Tile column index (in tile units, not pixels)
    /// * `tile_y` - Tile row index (in tile units, not pixels)
    pub(crate) fn from_pixmap_tile(
        pixmap: &'a mut Pixmap,
        tile_x: u16,
        tile_y: u16,
    ) -> Option<Self> {
        let pixmap_width = pixmap.width();
        let pixmap_height = pixmap.height();

        // Calculate pixel coordinates for this tile
        let base_x = tile_x * WideTile::WIDTH;
        let base_y = tile_y * Tile::HEIGHT;

        // Check bounds
        if base_x >= pixmap_width || base_y >= pixmap_height {
            return None;
        }

        // Calculate actual region dimensions (might be smaller at edges)
        let region_width = WideTile::WIDTH.min(pixmap_width - base_x);
        let region_height = Tile::HEIGHT.min(pixmap_height - base_y);

        // Get mutable access to the pixmap's buffer
        let buffer = pixmap.data_as_u8_slice_mut();

        // Split buffer into row slices for this tile
        let row_stride = pixmap_width as usize * COLOR_COMPONENTS;
        let start_offset = (base_y as usize * row_stride) + (base_x as usize * COLOR_COMPONENTS);
        let region_width_bytes = region_width as usize * COLOR_COMPONENTS;

        // Skip to the start of our tile region
        let tile_buffer = &mut buffer[start_offset..];

        // Extract individual row slices using safe split operations
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        // Use split_at_mut to safely extract each row
        let mut remaining = tile_buffer;
        for (i, area) in areas.iter_mut().take(region_height as usize).enumerate() {
            if i > 0 {
                // Skip rows we've already processed (advance by stride - region_width_bytes)
                let skip = row_stride - region_width_bytes;
                remaining = &mut remaining[skip..];
            }

            let (row, rest) = remaining.split_at_mut(region_width_bytes.min(remaining.len()));
            *area = row;
            remaining = rest;
        }

        Some(Self::new(areas, region_width, region_height))
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT as usize] {
        &mut self.areas
    }
}
