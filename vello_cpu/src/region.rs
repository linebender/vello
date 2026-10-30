// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use crate::span::Span;
use alloc::vec::Vec;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;

/// A view into a part of a single strip row of a pixmap.
#[derive(Default, Debug)]
pub struct Region<'a> {
    pub(crate) row_idx: usize,
    scene_x0: u16,
    tile_y_start: u16,
    width: u16,
    pub(crate) height: u16,
    areas: [&'a mut [u8]; Tile::HEIGHT as usize],
}

impl<'a> Region<'a> {
    #[doc(hidden)]
    pub fn new(pixmap: &'a mut PixmapMut<'_>, rect: RectU16) -> Self {
        Self::new_from_row(pixmap, rect, 0)
    }

    pub(crate) fn new_from_row(
        pixmap: &'a mut PixmapMut<'_>,
        rect: RectU16,
        row_idx: usize,
    ) -> Self {
        let width = rect.width();
        let height = rect.height().min(Tile::HEIGHT);
        let row_stride = usize::from(pixmap.width()) * COLOR_COMPONENTS;
        let start_offset = usize::from(rect.y0) * row_stride;
        let x_offset_bytes = usize::from(rect.x0) * COLOR_COMPONENTS;
        let buffer = pixmap.data_mut();
        Self::from_rows(
            row_idx,
            0,
            0,
            width,
            height,
            row_stride,
            x_offset_bytes,
            &mut buffer[start_offset..],
        )
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub(crate) fn width(&self) -> u16 {
        self.width
    }

    pub(crate) fn span(&self) -> Span {
        Span::new(self.scene_x0, self.width)
    }

    pub(crate) fn tile_rows(&self) -> core::ops::Range<u16> {
        self.tile_y_start..self.tile_y_start + self.height
    }

    /// Return a horizontal sub-span of the region.
    pub(crate) fn sub_span(&mut self, x: u16, width: u16) -> Region<'_> {
        let x_offset_bytes = usize::from(x) * COLOR_COMPONENTS;
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for (source, area) in self
            .areas
            .iter_mut()
            .zip(areas.iter_mut())
            .skip(usize::from(self.tile_y_start))
            .take(usize::from(self.height))
        {
            let (_, source) = source.split_at_mut(x_offset_bytes);
            let (source, _) = source.split_at_mut(row_width_bytes);
            *area = source;
        }

        Region {
            row_idx: self.row_idx,
            scene_x0: self.scene_x0 + x,
            tile_y_start: self.tile_y_start,
            width,
            height: self.height,
            areas,
        }
    }

    pub(crate) fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT as usize] {
        &mut self.areas
    }

    fn from_rows(
        row_idx: usize,
        scene_x0: u16,
        tile_y_start: u16,
        width: u16,
        height: u16,
        row_stride: usize,
        x_offset_bytes: usize,
        mut rows: &'a mut [u8],
    ) -> Self {
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for area in areas
            .iter_mut()
            .skip(usize::from(tile_y_start))
            .take(usize::from(height))
        {
            let (row, rest) = rows.split_at_mut(row_stride);
            let (_, row) = row.split_at_mut(x_offset_bytes);
            let (row, _) = row.split_at_mut(row_width_bytes);
            *area = row;
            rows = rest;
        }

        Self {
            row_idx,
            scene_x0,
            tile_y_start,
            width,
            height,
            areas,
        }
    }
}

/// Split a pixmap into an array of regions.
pub(crate) struct Regions<'a> {
    regions: Vec<Region<'a>>,
}

impl<'a> Regions<'a> {
    pub(crate) fn new(
        target: &'a mut PixmapMut<'_>,
        scene_size: (u16, u16),
        offset: (i32, i32),
        row_count: usize,
    ) -> Self {
        let (dst_x, dst_y) = offset;
        let (scene_width, scene_height) = scene_size;
        let max_height = u16::try_from(row_count * usize::from(Tile::HEIGHT)).unwrap_or(u16::MAX);
        let Some(visible) = RectU16::visible_scene_bounds(
            SizeU16::from_wh(scene_width, scene_height.min(max_height)),
            SizeU16::from_wh(target.width(), target.height()),
            offset,
        ) else {
            return Self {
                regions: Vec::new(),
            };
        };

        let first_row_idx = visible.y0 / Tile::HEIGHT;
        let last_row_idx = (visible.y1 - 1) / Tile::HEIGHT;

        let stride = usize::from(target.width()) * COLOR_COMPONENTS;
        let x_offset_bytes =
            usize::try_from(i32::from(visible.x0) + dst_x).unwrap() * COLOR_COMPONENTS;
        let render_bytes = usize::from(visible.height()) * stride;
        let target_y0 = usize::try_from(i32::from(visible.y0) + dst_y).unwrap();
        let target = target.data_mut();
        let mut remaining = &mut target[target_y0 * stride..][..render_bytes];
        let mut regions = Vec::with_capacity(usize::from(last_row_idx - first_row_idx + 1));

        for row_idx in first_row_idx..=last_row_idx {
            let band_y0 = (row_idx * Tile::HEIGHT).max(visible.y0);
            let band_y1 = (row_idx * Tile::HEIGHT)
                .saturating_add(Tile::HEIGHT)
                .min(visible.y1);
            let band_height = band_y1 - band_y0;
            let (buffer, rest) = remaining.split_at_mut(usize::from(band_height) * stride);
            regions.push(Region::from_rows(
                usize::from(row_idx),
                visible.x0,
                band_y0 - row_idx * Tile::HEIGHT,
                visible.width(),
                band_height,
                stride,
                x_offset_bytes,
                buffer,
            ));
            remaining = rest;
        }

        Self { regions }
    }

    pub(crate) fn update(&mut self, func: impl FnMut(&mut Region<'_>)) {
        self.regions.iter_mut().for_each(func);
    }

    #[cfg(feature = "multithreading")]
    pub(crate) fn update_par(&mut self, func: impl Fn(&mut Region<'_>) + Send + Sync) {
        use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

        self.regions.par_iter_mut().for_each(func);
    }
}

#[cfg(test)]
mod tests {
    use super::Regions;
    use vello_common::pixmap::Pixmap;

    #[test]
    fn regions_with_off_target_offsets_do_not_panic() {
        for offset in [
            (20, 0),
            (0, 20),
            (-20, 0),
            (0, -20),
            (i32::MIN, i32::MIN),
            (i32::MAX, i32::MAX),
        ] {
            let mut pixmap = Pixmap::new(10, 10);
            let mut pixmap = pixmap.as_mut();
            let _regions = Regions::new(&mut pixmap, (4, 4), offset, 1);
        }
    }
}
