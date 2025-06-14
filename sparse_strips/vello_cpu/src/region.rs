// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use alloc::vec::Vec;
use vello_common::coarse::WideTile;
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

                for h in 0..region_height {
                    let region_width =
                        (usize::from(WideTile::WIDTH) * COLOR_COMPONENTS).min(next_lines[h].len());
                    let next = core::mem::take(&mut next_lines[h]);
                    let (head, tail) = next.split_at_mut(region_width);
                    areas[h] = head;
                    next_lines[h] = tail;
                }

                regions.push(Region::new(
                    areas,
                    u16::try_from(x).unwrap(),
                    u16::try_from(y).unwrap(),
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
    areas: [&'a mut [u8]; Tile::HEIGHT as usize],
}

impl<'a> Region<'a> {
    pub(crate) fn new(areas: [&'a mut [u8]; Tile::HEIGHT as usize], x: u16, y: u16) -> Self {
        Self { areas, x, y }
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }
}
