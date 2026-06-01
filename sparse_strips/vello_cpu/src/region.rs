// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Splitting a single mutable buffer into regions that can be accessed concurrently.

use crate::fine::COLOR_COMPONENTS;
use vello_common::tile::Tile;

/// A rectangular row-major view into a pixmap.
///
/// Most callers create regions only for the row span they are about to pack or
/// unpack. The region can be narrower than a tile-width block and can also be
/// shorter than [`Tile::HEIGHT`] at the bottom edge.
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

    pub(crate) fn row_mut(&mut self, y: u16) -> &mut [u8] {
        self.areas[usize::from(y)]
    }

    pub fn areas(&mut self) -> &mut [&'a mut [u8]; Tile::HEIGHT as usize] {
        &mut self.areas
    }
}
