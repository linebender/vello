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

    /// Return a horizontal sub-span of the region.
    pub(crate) fn sub_span(&mut self, x: u16, width: u16) -> Region<'_> {
        let x_offset = usize::from(x) * COLOR_COMPONENTS;
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for (source, area) in self
            .areas
            .iter_mut()
            .take(usize::from(self.height))
            .zip(areas.iter_mut())
        {
            let (_, source) = source.split_at_mut(x_offset);
            let (source, _) = source.split_at_mut(row_width_bytes);
            *area = source;
        }

        Region {
            row_idx: self.row_idx,
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
        width: u16,
        height: u16,
        row_stride: usize,
        x_offset: usize,
        mut rows: &'a mut [u8],
    ) -> Self {
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for area in areas.iter_mut().take(usize::from(height)) {
            let (row, rest) = rows.split_at_mut(row_stride);
            let (_, row) = row.split_at_mut(x_offset);
            let (row, _) = row.split_at_mut(row_width_bytes);
            *area = row;
            rows = rest;
        }

        Self {
            row_idx,
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
        offset: (u16, u16),
        row_count: usize,
    ) -> Self {
        let (dst_x, dst_y) = offset;

        let (scene_width, scene_height) = scene_size;
        let width = scene_width.min(target.width().saturating_sub(dst_x));
        let height = scene_height.min(target.height().saturating_sub(dst_y));

        if width == 0 || height == 0 {
            return Self {
                regions: Vec::new(),
            };
        }

        let row_count = row_count.min(usize::from(height).div_ceil(Tile::HEIGHT as usize));
        let stride = usize::from(target.width()) * COLOR_COMPONENTS;
        let x_offset = usize::from(dst_x) * COLOR_COMPONENTS;
        let render_bytes = usize::from(height) * stride;
        let target = target.data_mut();
        let mut remaining = &mut target[usize::from(dst_y) * stride..][..render_bytes];
        let mut regions = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            let row_y = row_idx as u16 * Tile::HEIGHT;
            let row_height = (height - row_y).min(Tile::HEIGHT);
            let band_len = usize::from(row_height) * stride;
            let (buffer, rest) = remaining.split_at_mut(band_len);
            regions.push(Region::from_rows(
                row_idx, width, row_height, stride, x_offset, buffer,
            ));
            remaining = rest;
        }

        Self { regions }
    }

    pub(crate) fn update(&mut self, func: impl FnMut(&mut Region<'_>)) {
        self.regions.iter_mut().for_each(func);
    }

    /// Like `update`, but polls `cancel` before each strip-row region and stops
    /// early — leaving the remaining regions unrendered — if it returns `true`.
    /// Returns `true` if every region was rendered.
    ///
    /// Each region is one strip row (`Tile::HEIGHT` scanlines), so this bounds
    /// cancellation latency to roughly one strip row's rasterization.
    pub(crate) fn update_cancellable(
        &mut self,
        cancel: &(dyn Fn() -> bool + Sync),
        mut func: impl FnMut(&mut Region<'_>),
    ) -> bool {
        for region in self.regions.iter_mut() {
            if cancel() {
                return false;
            }
            func(region);
        }
        true
    }

    #[cfg(feature = "multithreading")]
    pub(crate) fn update_par(&mut self, func: impl Fn(&mut Region<'_>) + Send + Sync) {
        use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

        self.regions.par_iter_mut().for_each(func);
    }

    /// Like `update_par`, but polls `cancel` once per region from the worker
    /// threads. The first region to observe a stop flips a shared flag; every
    /// other region checks it and skips its work, so cancellation latency is
    /// bounded by the regions already executing (one per worker thread).
    /// Returns `true` if every region was rendered.
    #[cfg(feature = "multithreading")]
    pub(crate) fn update_par_cancellable(
        &mut self,
        cancel: &(dyn Fn() -> bool + Sync),
        func: impl Fn(&mut Region<'_>) + Send + Sync,
    ) -> bool {
        use core::sync::atomic::{AtomicBool, Ordering};
        use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

        let cancelled = AtomicBool::new(false);
        self.regions.par_iter_mut().for_each(|region| {
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            if cancel() {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            func(region);
        });
        !cancelled.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::Regions;
    use vello_common::pixmap::Pixmap;

    #[test]
    fn regions_with_off_target_offsets_do_not_panic() {
        for offset in [(20, 0), (0, 20)] {
            let mut pixmap = Pixmap::new(10, 10);
            let mut pixmap = pixmap.as_mut();
            let _regions = Regions::new(&mut pixmap, (4, 4), offset, 1);
        }
    }
}
