// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use alloc::vec::Vec;
use vello_common::coarse::WideTile;
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;

#[derive(Debug)]
pub struct Regions<'a> {
    regions: Vec<Region<'a>>,
}

impl<'a> Regions<'a> {
    pub fn new(width: u16, height: u16, mut buffer: &'a mut [u8]) -> Self {
        let buf_width = usize::from(width);
        let buf_height = usize::from(height);

        let row_advance = buf_width * COLOR_COMPONENTS;

        let height_regions = buf_height.div_ceil(usize::from(Tile::HEIGHT));
        let width_regions = buf_width.div_ceil(usize::from(WideTile::WIDTH));

        let mut regions = Vec::with_capacity(height_regions * width_regions);

        let mut next_lines: [&'a mut [u8]; Tile::HEIGHT as usize] =
            [&mut [], &mut [], &mut [], &mut []];

        for y in 0..height_regions {
            let base_y = y * usize::from(Tile::HEIGHT);
            let region_height = usize::from(Tile::HEIGHT).min(buf_height - base_y);

            for line in next_lines.iter_mut().take(region_height) {
                let (head, tail) = buffer.split_at_mut(row_advance);
                *line = head;
                buffer = tail;
            }

            for x in 0..width_regions {
                let mut areas: [&mut [u8]; Tile::HEIGHT as usize] =
                    [&mut [], &mut [], &mut [], &mut []];

                // All rows have the same width, so we can just take the first row.
                let region_width =
                    (usize::from(WideTile::WIDTH) * COLOR_COMPONENTS).min(next_lines[0].len());

                for h in 0..region_height {
                    let next = core::mem::take(&mut next_lines[h]);
                    let (head, tail) = next.split_at_mut(region_width);
                    areas[h] = head;
                    next_lines[h] = tail;
                }

                regions.push(Region::new(
                    areas,
                    u16::try_from(x).unwrap(),
                    u16::try_from(y).unwrap(),
                    region_width as u16 / COLOR_COMPONENTS as u16,
                    region_height as u16,
                ));
            }
        }

        Self { regions }
    }

    /// Apply the given function to each region. The functions will be applied
    /// in parallel in the current threadpool.
    #[cfg(feature = "multithreading")]
    pub fn update_regions_par(&mut self, func: impl Fn(&mut Region<'_>) + Send + Sync) {
        use rayon::iter::ParallelIterator;
        use rayon::prelude::IntoParallelRefMutIterator;

        self.regions.par_iter_mut().for_each(func);
    }

    /// Apply the given function to each region.
    pub fn update_regions(&mut self, func: impl FnMut(&mut Region<'_>)) {
        self.regions.iter_mut().for_each(func);
    }
}

/// A rectangular region containing the pixels from one wide tile.
///
/// For wide tiles at the right/bottom edge, it might contain less pixels
/// than the actual wide tile, if the pixmap width/height isn't a multiple of the
/// tile width/height.
#[derive(Default, Debug)]
pub struct Region<'a> {
    /// The x coordinate of the wide tile this region covers.
    pub(crate) x: u16,
    /// The y coordinate of the wide tile this region covers.
    pub(crate) y: u16,
    pub width: u16,
    pub height: u16,
    areas: [&'a mut [u8]; Tile::HEIGHT as usize],
}

impl<'a> Region<'a> {
    pub(crate) fn new(
        areas: [&'a mut [u8]; Tile::HEIGHT as usize],
        x: u16,
        y: u16,
        width: u16,
        height: u16,
    ) -> Self {
        Self {
            areas,
            x,
            y,
            width,
            height,
        }
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

        Some(Self::new(
            areas,
            tile_x,
            tile_y,
            region_width,
            region_height,
        ))
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT as usize] {
        &mut self.areas
    }
}
