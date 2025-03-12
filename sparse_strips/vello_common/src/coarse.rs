// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use crate::color::{AlphaColor, Srgb};
use crate::strip::{STRIP_HEIGHT, Strip};
use vello_api::paint::Paint;
use vello_api::peniko::Fill;

/// The width of a wide tile.
pub const WIDE_TILE_WIDTH: usize = WideTile::WIDTH as usize;

/// A container for wide tiles.
#[derive(Debug)]
pub struct Wide {
    /// The width of the container.
    pub width: usize,
    /// The height of the container.
    pub height: usize,
    /// The wide tiles in the container.
    pub tiles: Vec<WideTile>,
}

impl Wide {
    /// Create a new container for wide tiles.
    pub fn new(width: usize, height: usize) -> Self {
        let width_tiles = width.div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = height.div_ceil(STRIP_HEIGHT);
        let mut tiles = Vec::with_capacity(width_tiles * height_tiles);

        for w in 0..width_tiles {
            for h in 0..height_tiles {
                tiles.push(WideTile::new(w * WIDE_TILE_WIDTH, h * STRIP_HEIGHT));
            }
        }

        Self {
            tiles,
            width,
            height,
        }
    }

    /// Reset all tiles in the container.
    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.bg = AlphaColor::TRANSPARENT;
            tile.cmds.clear();
        }
    }

    /// Return the number of horizontal tiles.
    pub fn width_tiles(&self) -> usize {
        self.width.div_ceil(WIDE_TILE_WIDTH)
    }

    /// Return the number of vertical tiles.
    pub fn height_tiles(&self) -> usize {
        self.height.div_ceil(STRIP_HEIGHT)
    }

    /// Get the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get(&self, x: usize, y: usize) -> &WideTile {
        assert!(
            x < self.width && y < self.height,
            "attempted to access out-of-bounds wide tile"
        );

        &self.tiles[y * self.width_tiles() + x]
    }

    /// Get mutable access to the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut WideTile {
        assert!(
            x < self.width && y < self.height,
            "attempted to access out-of-bounds wide tile"
        );

        let idx = y * self.width_tiles() + x;
        &mut self.tiles[idx]
    }

    /// Return a reference to all wide tiles.
    pub fn tiles(&self) -> &[WideTile] {
        self.tiles.as_slice()
    }

    /// Generate wide tile commands from the strip buffer.
    pub fn generate(&mut self, strip_buf: &[Strip], fill_rule: Fill, paint: Paint) {
        let width_tiles = self.width_tiles();

        if strip_buf.is_empty() {
            return;
        }

        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];

            if strip.x >= self.width as i32 {
                // Don't render strips that are outside the viewport.
                continue;
            }

            if strip.y >= self.height as u16 {
                // Since strips are sorted by location, any subsequent strips will also be
                // outside the viewport, so we can abort entirely.
                break;
            }

            let next_strip = &strip_buf[i + 1];
            // Currently, strips can also start at a negative x position, since we don't
            // support viewport culling yet. However, when generating the commands
            // we only want to emit strips >= 0, so we calculate the adjustment
            // and then only include the alpha indices for columns where x >= 0.
            let x0_adjustment = strip.x.min(0).unsigned_abs();
            let x0 = (strip.x + x0_adjustment as i32) as u32;
            let strip_y = strip.strip_y();
            let mut col = strip.col + x0_adjustment;
            // Can potentially be 0, if the next strip's x values is also < 0.
            let strip_width = next_strip.col.saturating_sub(col);
            let x1 = x0 + strip_width;
            let tile_x0 = x0 as usize / WIDE_TILE_WIDTH;
            // It's possible that a strip extends into a new wide tile, but we don't actually
            // have as many wide tiles (e.g. because the pixmap width is only 512, but
            // strip ends at 513), so take the minimum between the rounded values and `width_tiles`.
            let tile_x1 = (x1 as usize).div_ceil(WIDE_TILE_WIDTH).min(width_tiles);
            let mut x = x0;

            for tile_x in tile_x0..tile_x1 {
                let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                let width = x1.min(((tile_x + 1) * WIDE_TILE_WIDTH) as u32) - x;
                let cmd = CmdAlphaFill {
                    x: x_tile_rel,
                    width,
                    alpha_ix: col as usize,
                    paint: paint.clone(),
                };
                x += width;
                col += width;
                self.get_mut(tile_x, strip_y as usize)
                    .push(Cmd::AlphaFill(cmd));
            }

            let active_fill = match fill_rule {
                Fill::NonZero => next_strip.winding != 0,
                Fill::EvenOdd => next_strip.winding % 2 != 0,
            };

            if active_fill
                && strip_y == next_strip.strip_y()
                // Only fill if we are actually inside the viewport.
                && next_strip.x >= 0
            {
                x = x1;
                let x2 =
                    (next_strip.x as u32).min(self.width.next_multiple_of(WIDE_TILE_WIDTH) as u32);
                let fxt0 = x1 as usize / WIDE_TILE_WIDTH;
                let fxt1 = (x2 as usize).div_ceil(WIDE_TILE_WIDTH);
                for tile_x in fxt0..fxt1 {
                    let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                    let width = x2.min(((tile_x + 1) * WIDE_TILE_WIDTH) as u32) - x;
                    x += width;
                    self.get_mut(tile_x, strip_y as usize)
                        .fill(x_tile_rel, width, paint.clone());
                }
            }
        }
    }
}

/// A wide tile.
#[derive(Debug)]
pub struct WideTile {
    /// The x coordinate of the wide tile.
    pub x: usize,
    /// The y coordinate of the wide tile.
    pub y: usize,
    /// The background of the tile.
    pub bg: AlphaColor<Srgb>,
    /// The draw commands of the tile.
    pub cmds: Vec<Cmd>,
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;

    /// Create a new wide tile.
    pub fn new(x: usize, y: usize) -> Self {
        Self {
            x,
            y,
            bg: AlphaColor::TRANSPARENT,
            cmds: vec![],
        }
    }

    pub(crate) fn fill(&mut self, x: u32, width: u32, paint: Paint) {
        let Paint::Solid(s) = &paint else {
            unimplemented!()
        };
        let can_override = x == 0 && width == WIDE_TILE_WIDTH as u32 && s.components[3] == 1.0;

        if can_override {
            self.cmds.clear();
            self.bg = *s;
        } else {
            self.cmds.push(Cmd::Fill(CmdFill { x, width, paint }));
        }
    }

    pub(crate) fn push(&mut self, cmd: Cmd) {
        self.cmds.push(cmd);
    }
}

/// A drawing command.
#[derive(Debug)]
pub enum Cmd {
    /// A fill command.
    Fill(CmdFill),
    /// A fill command with alpha mask.
    AlphaFill(CmdAlphaFill),
}

/// Fill a consecutive region of a wide tile.
#[derive(Debug)]
pub struct CmdFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}

/// Fill a consecutive region of a wide tile with an alpha mask.
#[derive(Debug)]
pub struct CmdAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
    /// The start index into the alpha buffer of the command.
    pub alpha_ix: usize,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}
