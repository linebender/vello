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
    /// Creates regions from a buffer where the buffer dimensions match the render dimensions.
    pub fn new(width: u16, height: u16, buffer: &'a mut [u8]) -> Self {
        Self::new_at_offset(width, height, 0, 0, width, height, buffer)
    }

    /// Creates regions from a buffer at a specific offset.
    ///
    /// This is used for rendering to a sub-region of a larger buffer. The regions
    /// cover the area of size (`width` × `height`) placed at pixel offset
    /// (`dst_x`, `dst_y`) in the destination buffer.
    ///
    /// # Arguments
    /// * `width` - Width of the content being rendered
    /// * `height` - Height of the content being rendered
    /// * `dst_x` - X offset in the destination buffer
    /// * `dst_y` - Y offset in the destination buffer
    /// * `dst_buffer_width` - Total width of the destination buffer
    /// * `dst_buffer_height` - Total height of the destination buffer
    /// * `buffer` - The destination buffer (RGBA, 4 bytes per pixel)
    pub fn new_at_offset(
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        mut buffer: &'a mut [u8],
    ) -> Self {
        // Calculate effective render area (clamped to destination bounds)
        let effective_width = width.min(dst_buffer_width.saturating_sub(dst_x)) as usize;
        let effective_height = height.min(dst_buffer_height.saturating_sub(dst_y)) as usize;

        if effective_width == 0 || effective_height == 0 {
            return Self {
                regions: Vec::new(),
            };
        }

        let width_regions = effective_width.div_ceil(WideTile::WIDTH as usize);
        let height_regions = effective_height.div_ceil(Tile::HEIGHT as usize);

        let mut regions = Vec::with_capacity(width_regions * height_regions);

        let row_stride = dst_buffer_width as usize * COLOR_COMPONENTS;
        let render_row_bytes = effective_width * COLOR_COMPONENTS;

        // Calculate starting offset in the buffer
        let start_offset = (dst_y as usize * row_stride) + (dst_x as usize * COLOR_COMPONENTS);
        buffer = &mut buffer[start_offset..];

        let mut next_lines: [&'a mut [u8]; Tile::HEIGHT as usize] =
            [&mut [], &mut [], &mut [], &mut []];

        for y in 0..height_regions {
            let base_y = y * Tile::HEIGHT as usize;
            let region_height = (Tile::HEIGHT as usize).min(effective_height - base_y);

            // Extract Tile::HEIGHT rows from the buffer
            // Each row is at row_stride intervals
            for line in next_lines.iter_mut().take(region_height) {
                // Take only the render area portion of this row
                let (render_portion, rest) = buffer.split_at_mut(render_row_bytes);
                *line = render_portion;
                // Skip the remainder of this buffer row to get to the next row.
                // On the last row (when clipped to pixmap edge), there may be less
                // data remaining, so cap the skip to what's available.
                let skip = (row_stride - render_row_bytes).min(rest.len());
                buffer = &mut rest[skip..];
            }

            // Split each row horizontally into tile-width chunks
            for x in 0..width_regions {
                let mut areas: [&mut [u8]; Tile::HEIGHT as usize] =
                    [&mut [], &mut [], &mut [], &mut []];

                let base_x = x * WideTile::WIDTH as usize;
                let region_width_bytes =
                    ((WideTile::WIDTH as usize).min(effective_width - base_x)) * COLOR_COMPONENTS;

                for h in 0..region_height {
                    let next = core::mem::take(&mut next_lines[h]);
                    let (head, tail) = next.split_at_mut(region_width_bytes);
                    areas[h] = head;
                    next_lines[h] = tail;
                }

                regions.push(Region::new(
                    areas,
                    u16::try_from(x).unwrap(),
                    u16::try_from(y).unwrap(),
                    (region_width_bytes / COLOR_COMPONENTS) as u16,
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
