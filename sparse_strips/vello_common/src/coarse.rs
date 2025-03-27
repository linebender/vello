// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use crate::{
    color::{AlphaColor, Srgb},
    strip::Strip,
    tile::Tile,
};
use vello_api::color::PremulRgba8;
use vello_api::{paint::Paint, peniko::Fill};

/// A container for wide tiles.
#[derive(Debug)]
pub struct Wide {
    /// The width of the container.
    pub width: u16,
    /// The height of the container.
    pub height: u16,
    /// The wide tiles in the container.
    pub tiles: Vec<WideTile>,
}

impl Wide {
    /// Create a new container for wide tiles.
    pub fn new(width: u16, height: u16) -> Self {
        let width_tiles = width.div_ceil(WideTile::WIDTH);
        let height_tiles = height.div_ceil(Tile::HEIGHT);
        let mut tiles = Vec::with_capacity(usize::from(width_tiles * height_tiles));

        for w in 0..width_tiles {
            for h in 0..height_tiles {
                tiles.push(WideTile::new(w * WideTile::WIDTH, h * Tile::HEIGHT));
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
            tile.bg = AlphaColor::<Srgb>::TRANSPARENT.premultiply().to_rgba8();
            tile.cmds.clear();
        }
    }

    /// Return the number of horizontal tiles.
    pub fn width_tiles(&self) -> u16 {
        self.width.div_ceil(WideTile::WIDTH)
    }

    /// Return the number of vertical tiles.
    pub fn height_tiles(&self) -> u16 {
        self.height.div_ceil(Tile::HEIGHT)
    }

    /// Get the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get(&self, x: u16, y: u16) -> &WideTile {
        assert!(
            x < self.width && y < self.height,
            "attempted to access out-of-bounds wide tile"
        );

        &self.tiles[usize::from(y) * usize::from(self.width_tiles()) + usize::from(x)]
    }

    /// Get mutable access to the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get_mut(&mut self, x: u16, y: u16) -> &mut WideTile {
        assert!(
            x < self.width && y < self.height,
            "attempted to access out-of-bounds wide tile"
        );

        let idx = usize::from(y) * usize::from(self.width_tiles()) + usize::from(x);
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

            debug_assert!(
                strip.y < self.height,
                "Strips below the viewport should have been culled prior to this stage."
            );

            if strip.x >= self.width {
                // Don't render strips that are outside the viewport.
                continue;
            }

            let next_strip = &strip_buf[i + 1];
            let x0 = strip.x;
            let strip_y = strip.strip_y();
            let mut col = strip.alpha_idx / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx / u32::from(Tile::HEIGHT);
            // Can potentially be 0, e.g. if the strip only changes coarse winding, but this
            // depends on exact details of strip footprints.
            let strip_width = next_col.saturating_sub(col) as u16;
            let x1 = x0 + strip_width;
            let tile_x0 = x0 / WideTile::WIDTH;
            // It's possible that a strip extends into a new wide tile, but we don't actually
            // have as many wide tiles (e.g. because the pixmap width is only 512, but
            // strip ends at 513), so take the minimum between the rounded values and `width_tiles`.
            let tile_x1 = x1.div_ceil(WideTile::WIDTH).min(width_tiles);
            let mut x = x0;

            for tile_x in tile_x0..tile_x1 {
                let x_tile_rel = x % WideTile::WIDTH;
                let width = x1.min((tile_x + 1) * WideTile::WIDTH) - x;
                let cmd = CmdAlphaFill {
                    x: x_tile_rel,
                    width,
                    alpha_ix: (col * u32::from(Tile::HEIGHT)) as usize,
                    paint: paint.clone(),
                };
                x += width;
                col += u32::from(width);
                self.get_mut(tile_x, strip_y).push(Cmd::AlphaFill(cmd));
            }

            let active_fill = match fill_rule {
                Fill::NonZero => next_strip.winding != 0,
                Fill::EvenOdd => next_strip.winding % 2 != 0,
            };

            if active_fill && strip_y == next_strip.strip_y() {
                x = x1;
                let x2 = next_strip
                    .x
                    .min(self.width.next_multiple_of(WideTile::WIDTH));
                let fxt0 = x1 / WideTile::WIDTH;
                let fxt1 = x2.div_ceil(WideTile::WIDTH);
                for tile_x in fxt0..fxt1 {
                    let x_tile_rel = x % WideTile::WIDTH;
                    let width = x2.min((tile_x + 1) * WideTile::WIDTH) - x;
                    x += width;
                    self.get_mut(tile_x, strip_y)
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
    pub x: u16,
    /// The y coordinate of the wide tile.
    pub y: u16,
    /// The background of the tile.
    pub bg: PremulRgba8,
    /// The draw commands of the tile.
    pub cmds: Vec<Cmd>,
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;

    /// Create a new wide tile.
    pub fn new(x: u16, y: u16) -> Self {
        Self {
            x,
            y,
            bg: AlphaColor::<Srgb>::TRANSPARENT.premultiply().to_rgba8(),
            cmds: vec![],
        }
    }

    pub(crate) fn fill(&mut self, x: u16, width: u16, paint: Paint) {
        let Paint::Solid(s) = &paint else {
            unimplemented!()
        };
        let can_override = x == 0 && width == Self::WIDTH && s.a == 255;

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
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}

/// Fill a consecutive region of a wide tile with an alpha mask.
#[derive(Debug)]
pub struct CmdAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The start index into the alpha buffer of the command.
    pub alpha_ix: usize,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}
