// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects implementation for `vello_cpu`.

mod drop_shadow;
mod flood;
mod gaussian_blur;

pub(crate) use drop_shadow::DropShadow;
pub(crate) use flood::Flood;
pub(crate) use gaussian_blur::GaussianBlur;

use crate::fine::{Numeric, ScratchBuf};
use alloc::vec::Vec;
use vello_common::coarse::{Bbox, WideTile};
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;

/// Trait for filter effects that can be applied to layers.
///
/// Each filter implements this trait with both u8 and f32 variants
/// to support different rendering backends and precision requirements.
pub(crate) trait FilterEffect {
    /// Apply the filter to a u8 buffer.
    fn apply_u8(&self, buffer: &mut FilterBuffer<u8>);

    /// Apply the filter to an f32 buffer.
    fn apply_f32(&self, buffer: &mut FilterBuffer<f32>);
}

/// Apply a filter to a u8 layer.
///
/// This function applies the filter graph to the given layer.
/// Currently only supports single-primitive filters.
pub(crate) fn apply_filter_u8(
    filter: &Filter,
    layer: &mut Vec<ScratchBuf<u8>>,
    wtile_bbox: vello_common::coarse::Bbox,
) {
    // Convert to FilterBuffer (row-major layout)
    let mut buffer = FilterBuffer::from_layer(layer, wtile_bbox);

    // Apply filter
    if filter.graph.primitives.len() == 1 {
        match &filter.graph.primitives[0] {
            FilterPrimitive::GaussianBlur { std_deviation } => {
                let blur = GaussianBlur::new(*std_deviation);
                blur.apply_u8(&mut buffer);
            }
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                flood.apply_u8(&mut buffer);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
            } => {
                let drop_shadow = DropShadow::new(*dx, *dy, *std_deviation, *color);
                drop_shadow.apply_u8(&mut buffer);
            }
            _ => {
                // Other filter primitives not yet implemented
            }
        }
    }

    // Convert back to layer (column-major tiled layout)
    buffer.to_layer(layer);
}

/// Apply a filter to an f32 layer.
///
/// This function applies the filter graph to the given layer.
/// Currently only supports single-primitive filters.
pub(crate) fn apply_filter_f32(
    filter: &Filter,
    layer: &mut Vec<ScratchBuf<f32>>,
    wtile_bbox: vello_common::coarse::Bbox,
) {
    // Convert to FilterBuffer (row-major layout)
    let mut buffer = FilterBuffer::from_layer(layer, wtile_bbox);

    // Apply filter
    if filter.graph.primitives.len() == 1 {
        match &filter.graph.primitives[0] {
            FilterPrimitive::GaussianBlur { std_deviation } => {
                let blur = GaussianBlur::new(*std_deviation);
                blur.apply_f32(&mut buffer);
            }
            FilterPrimitive::Flood { color } => {
                let flood = Flood::new(*color);
                flood.apply_f32(&mut buffer);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_deviation,
                color,
            } => {
                let drop_shadow = DropShadow::new(*dx, *dy, *std_deviation, *color);
                drop_shadow.apply_f32(&mut buffer);
            }
            _ => {
                // Other filter primitives not yet implemented
            }
        }
    }

    // Convert back to layer (column-major tiled layout)
    buffer.to_layer(layer);
}

/// A row-major buffer for filter operations.
///
/// Pixels are stored as `[R, G, B, A]` in row-major order:
/// index = (y * width + x) * 4
///
/// This provides a simpler interface for filter operations compared to the
/// tiled column-major `ScratchBuf` layout used internally by the renderer.
pub(crate) struct FilterBuffer<T: Numeric> {
    data: Vec<T>,
    width: u16,
    height: u16,
}

impl<T: Numeric> FilterBuffer<T> {
    /// Create a new empty `FilterBuffer` with the specified dimensions.
    ///
    /// All pixels are initialized to zero (transparent).
    pub(crate) fn new(width: u16, height: u16) -> Self {
        Self {
            data: alloc::vec![T::ZERO; width as usize * height as usize * 4],
            width,
            height,
        }
    }

    /// Convert from tiled column-major `ScratchBuf` format to row-major `FilterBuffer`.
    ///
    /// # Parameters
    /// - `layer`: Slice of `ScratchBuf` tiles in column-major order (tiled layout)
    /// - `width`: Width of the image in pixels
    /// - `height`: Height of the image in pixels
    pub(crate) fn from_layer(layer: &[ScratchBuf<T>], wtile_bbox: Bbox) -> Self {
        let width_tiles = wtile_bbox.width();
        let height_tiles = wtile_bbox.height();
        let width = width_tiles * WideTile::WIDTH;
        let height = height_tiles * Tile::HEIGHT;
        let mut data = alloc::vec![T::ZERO; width as usize * height as usize * 4];

        // Convert from column-major tiles to row-major buffer
        for py in 0..height {
            for px in 0..width {
                // Determine which tile this pixel belongs to
                let tile_x = px / WideTile::WIDTH;
                let tile_y = py / Tile::HEIGHT;
                let tile_idx = (tile_y * width_tiles + tile_x) as usize;

                // Position within the tile
                let local_x = (px % WideTile::WIDTH) as usize;
                let local_y = (py % Tile::HEIGHT) as usize;

                // Column-major index within tile: 4 * (4 * x + y)
                let src_idx = 4 * (Tile::HEIGHT as usize * local_x + local_y);

                // Row-major index in output buffer
                let dst_idx = ((py * width + px) * 4) as usize;

                // Copy RGBA values
                let tile = &layer[tile_idx];
                data[dst_idx] = tile[src_idx];
                data[dst_idx + 1] = tile[src_idx + 1];
                data[dst_idx + 2] = tile[src_idx + 2];
                data[dst_idx + 3] = tile[src_idx + 3];
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Convert from row-major `FilterBuffer` back to tiled column-major `ScratchBuf` format.
    ///
    /// # Parameters
    /// - `layer`: Mutable slice of `ScratchBuf` tiles to write to
    pub(crate) fn to_layer(&self, layer: &mut [ScratchBuf<T>]) {
        let width_tiles = self.width.div_ceil(WideTile::WIDTH);

        // Convert from row-major buffer to column-major tiles
        for py in 0..self.height {
            for px in 0..self.width {
                // Determine which tile this pixel belongs to
                let tile_x = px / WideTile::WIDTH;
                let tile_y = py / Tile::HEIGHT;
                let tile_idx = (tile_y * width_tiles + tile_x) as usize;

                // Position within the tile
                let local_x = (px % WideTile::WIDTH) as usize;
                let local_y = (py % Tile::HEIGHT) as usize;

                // Column-major index within tile: 4 * (4 * x + y)
                let dst_idx = 4 * (Tile::HEIGHT as usize * local_x + local_y);

                // Row-major index in source buffer
                let src_idx = ((py * self.width + px) * 4) as usize;

                // Copy RGBA values
                let tile = &mut layer[tile_idx];
                tile[dst_idx] = self.data[src_idx];
                tile[dst_idx + 1] = self.data[src_idx + 1];
                tile[dst_idx + 2] = self.data[src_idx + 2];
                tile[dst_idx + 3] = self.data[src_idx + 3];
            }
        }
    }

    /// Get pixel at (x, y).
    #[inline]
    pub(crate) fn get_pixel(&self, x: u16, y: u16) -> [T; 4] {
        let idx = ((y * self.width + x) * 4) as usize;
        [
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ]
    }

    /// Set pixel at (x, y).
    #[inline]
    pub(crate) fn set_pixel(&mut self, x: u16, y: u16, rgba: [T; 4]) {
        let idx = ((y * self.width + x) * 4) as usize;
        self.data[idx..idx + 4].copy_from_slice(&rgba);
    }

    /// Get the width of the buffer in pixels.
    pub(crate) fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the buffer in pixels.
    pub(crate) fn height(&self) -> u16 {
        self.height
    }

    /// Get a reference to the underlying data.
    #[allow(dead_code, reason = "Part of public API, may be used in future")]
    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    #[allow(dead_code, reason = "Part of public API, may be used in future")]
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl FilterBuffer<u8> {
    /// Convert the `FilterBuffer` into a [`Pixmap`], consuming self.
    ///
    /// The u8 data is already in premultiplied RGBA8 format, so this is a direct conversion.
    #[allow(dead_code, reason = "Part of public API, will be used in future")]
    #[allow(
        clippy::wrong_self_convention,
        reason = "Consumes self intentionally to avoid cloning"
    )]
    pub(crate) fn to_pixmap(self) -> Pixmap {
        let pixels: Vec<PremulRgba8> = self
            .data
            .chunks_exact(4)
            .map(|chunk| PremulRgba8 {
                r: chunk[0],
                g: chunk[1],
                b: chunk[2],
                a: chunk[3],
            })
            .collect();

        Pixmap::from_parts(pixels, self.width, self.height)
    }
}

impl FilterBuffer<f32> {
    /// Convert the `FilterBuffer` into a [`Pixmap`], consuming self.
    ///
    /// The f32 data (0.0-1.0 range) is converted to u8 (0-255 range) premultiplied RGBA8.
    #[allow(dead_code, reason = "Part of public API, will be used in future")]
    #[allow(
        clippy::wrong_self_convention,
        reason = "Consumes self intentionally to avoid cloning"
    )]
    pub(crate) fn to_pixmap(self) -> Pixmap {
        let pixels: Vec<PremulRgba8> = self
            .data
            .chunks_exact(4)
            .map(|chunk| {
                let to_u8 = |val: f32| (val * 255.0).round().clamp(0.0, 255.0) as u8;
                PremulRgba8 {
                    r: to_u8(chunk[0]),
                    g: to_u8(chunk[1]),
                    b: to_u8(chunk[2]),
                    a: to_u8(chunk[3]),
                }
            })
            .collect();

        Pixmap::from_parts(pixels, self.width, self.height)
    }
}
