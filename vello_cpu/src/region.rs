// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use alloc::vec::Vec;
use vello_common::geometry::RectU16;
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;

/// A view into a part of a single strip row of a pixmap.
#[derive(Default, Debug)]
pub struct Region<'a> {
    pub(crate) row_idx: usize,
    x_start: u16,
    y_start: u16,
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
        let x_offset = usize::from(rect.x0) * COLOR_COMPONENTS;
        let buffer = pixmap.data_mut();
        Self::from_rows(
            row_idx,
            0,
            0,
            width,
            height,
            row_stride,
            x_offset,
            &mut buffer[start_offset..],
        )
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub(crate) fn width(&self) -> u16 {
        self.width
    }

    pub(crate) fn x_start(&self) -> u16 {
        self.x_start
    }

    pub(crate) fn rows(&self) -> core::ops::Range<u16> {
        self.y_start..self.y_start + self.height
    }

    /// Return a horizontal sub-span of the region.
    pub(crate) fn sub_span(&mut self, x: u16, width: u16) -> Region<'_> {
        let x_offset = usize::from(x) * COLOR_COMPONENTS;
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for (source, area) in self
            .areas
            .iter_mut()
            .zip(areas.iter_mut())
            .skip(usize::from(self.y_start))
            .take(usize::from(self.height))
        {
            let (_, source) = source.split_at_mut(x_offset);
            let (source, _) = source.split_at_mut(row_width_bytes);
            *area = source;
        }

        Region {
            row_idx: self.row_idx,
            x_start: self.x_start + x,
            y_start: self.y_start,
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
        x_start: u16,
        y_start: u16,
        width: u16,
        height: u16,
        row_stride: usize,
        x_offset: usize,
        mut rows: &'a mut [u8],
    ) -> Self {
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for area in areas
            .iter_mut()
            .skip(usize::from(y_start))
            .take(usize::from(height))
        {
            let (row, rest) = rows.split_at_mut(row_stride);
            let (_, row) = row.split_at_mut(x_offset);
            let (row, _) = row.split_at_mut(row_width_bytes);
            *area = row;
            rows = rest;
        }

        Self {
            row_idx,
            x_start,
            y_start,
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
        let empty = Self {
            regions: Vec::new(),
        };
        let (dst_x, dst_y) = offset;
        let (scene_width, scene_height) = scene_size;
        let max_height = u16::try_from(row_count * usize::from(Tile::HEIGHT)).unwrap_or(u16::MAX);
        let scene_height = scene_height.min(max_height);

        let x0 = dst_x.saturating_neg().clamp(0, i32::from(scene_width));
        let y0 = dst_y.saturating_neg().clamp(0, i32::from(scene_height));
        let x1 = i32::from(target.width())
            .saturating_sub(dst_x)
            .clamp(x0, i32::from(scene_width));
        let y1 = i32::from(target.height())
            .saturating_sub(dst_y)
            .clamp(y0, i32::from(scene_height));

        if x0 == x1 || y0 == y1 {
            return empty;
        }

        let [x0, y0, x1, y1] = [x0, y0, x1, y1].map(|v| u16::try_from(v).unwrap());
        let width = x1 - x0;
        let first_row = y0 / Tile::HEIGHT;
        let last_row = (y1 - 1) / Tile::HEIGHT;

        let stride = usize::from(target.width()) * COLOR_COMPONENTS;
        let x_offset = usize::try_from(i32::from(x0) + dst_x).unwrap() * COLOR_COMPONENTS;
        let render_bytes = usize::from(y1 - y0) * stride;
        let start_row = usize::try_from(i32::from(y0) + dst_y).unwrap();
        let target = target.data_mut();
        let mut remaining = &mut target[start_row * stride..][..render_bytes];
        let mut regions = Vec::with_capacity(usize::from(last_row - first_row + 1));

        for row in first_row..=last_row {
            let row_y0 = (row * Tile::HEIGHT).max(y0);
            let row_y1 = (row * Tile::HEIGHT).saturating_add(Tile::HEIGHT).min(y1);
            let row_height = row_y1 - row_y0;
            let (buffer, rest) = remaining.split_at_mut(usize::from(row_height) * stride);
            regions.push(Region::from_rows(
                usize::from(row),
                x0,
                row_y0 - row * Tile::HEIGHT,
                width,
                row_height,
                stride,
                x_offset,
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
