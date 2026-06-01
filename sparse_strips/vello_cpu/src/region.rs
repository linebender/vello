// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use vello_common::geometry::RectU16;
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;

/// A view into a single part of a single strip row of a pixmap.
#[derive(Default, Debug)]
pub struct Region<'a> {
    pub width: u16,
    areas: [&'a mut [u8]; Tile::HEIGHT as usize],
}

impl<'a> Region<'a> {
    pub(crate) fn new(pixmap: &'a mut PixmapMut<'_>, rect: RectU16) -> Option<Self> {
        let pixmap_bounds = RectU16::new(0, 0, pixmap.width(), pixmap.height());
        let rect = rect.intersect(pixmap_bounds);
        if rect.is_empty() {
            return None;
        }

        let row_stride = usize::from(pixmap.width()) * COLOR_COMPONENTS;
        if row_stride == 0 {
            return None;
        }

        let width = rect.width();
        let height = rect.height().min(Tile::HEIGHT);
        let start_offset =
            usize::from(rect.y0) * row_stride + usize::from(rect.x0) * COLOR_COMPONENTS;
        let row_width_bytes = usize::from(width) * COLOR_COMPONENTS;
        let buffer = pixmap.data_mut();
        let mut remaining = &mut buffer[start_offset..];
        let mut areas: [&mut [u8]; Tile::HEIGHT as usize] = [&mut [], &mut [], &mut [], &mut []];

        for area in areas.iter_mut().take(usize::from(height)) {
            let (row, rest) = remaining.split_at_mut(row_width_bytes);
            *area = row;
            let skip = (row_stride - row_width_bytes).min(rest.len());
            remaining = &mut rest[skip..];
        }

        Some(Self { areas, width })
    }

    pub(crate) fn height(&self) -> u16 {
        self.areas.len() as u16
    }

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub(crate) fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT as usize] {
        &mut self.areas
    }
}
