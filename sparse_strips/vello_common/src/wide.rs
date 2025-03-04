// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating wide tiles.

use crate::color::{AlphaColor, Srgb};
use crate::strip::Strip;
use vello_api::paint::Paint;
use vello_api::peniko::Fill;

/// The width of a wide tile.
pub const WIDE_TILE_WIDTH: usize = 256;

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
    /// A strip command.
    Strip(CmdStrip),
}

/// Fill a consecutive region of a wide tile.
#[derive(Debug)]
pub struct CmdFill {
    /// The horizontal start position of the command.
    pub x: u32,
    /// The width of the command.
    pub width: u32,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}

/// Strip a consecutive region of a wide tile.
#[derive(Debug)]
pub struct CmdStrip {
    /// The horizontal start position of the command.
    pub x: u32,
    /// The width of the command.
    pub width: u32,
    /// The start index in the alpha buffer of the command.
    pub alpha_ix: usize,
    /// The paint that should be used to strip the area.
    pub paint: Paint,
}

/// Generate the commands for wide tiles for the
pub fn generate(
    strip_buf: &[Strip],
    wide_tiles: &mut [WideTile],
    fill_rule: Fill,
    paint: Paint,
    width: usize,
    height: usize,
) {
    let width_tiles = width.div_ceil(WIDE_TILE_WIDTH);

    if strip_buf.is_empty() {
        return;
    }

    for i in 0..strip_buf.len() - 1 {
        let strip = &strip_buf[i];

        if strip.x >= width as i32 {
            // Don't render strips that are outside the viewport.
            continue;
        }

        if strip.y >= height as u16 {
            // Since strips are sorted by location, any subsequent strips will also be
            // outside the viewport, so we can abort entirely.
            break;
        }

        let next_strip = &strip_buf[i + 1];
        // Currently, strips can also start at a negative x position, since we don't
        // support viewport culling yet. However, when generating the commands
        // we only want to emit strips >= 0, so we calculate the adjustment
        // and then only include the alpha indices for columns where x >= 0.
        let x0_adjustment = (strip.x).min(0).unsigned_abs();
        let x0 = (strip.x + x0_adjustment as i32) as u32;
        let y = strip.strip_y();
        let row_start = y as usize * width_tiles;
        let mut col = strip.col + x0_adjustment;
        // Can potentially be 0, if the next strip's x values is also < 0.
        let strip_width = next_strip.col.saturating_sub(col);
        let x1 = x0 + strip_width;
        let xtile0 = x0 as usize / WIDE_TILE_WIDTH;
        // It's possible that a strip extends into a new wide tile, but we don't actually
        // have as many wide tiles (e.g. because the pixmap width is only 512, but
        // strip ends at 513), so take the minimum between the rounded values and `width_tiles`.
        let xtile1 = (x1 as usize).div_ceil(WIDE_TILE_WIDTH).min(width_tiles);
        let mut x = x0;

        for xtile in xtile0..xtile1 {
            let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
            let cmd_width = x1.min(((xtile + 1) * WIDE_TILE_WIDTH) as u32) - x;
            let cmd = CmdStrip {
                x: x_tile_rel,
                width: cmd_width,
                alpha_ix: col as usize,
                paint: paint.clone(),
            };
            x += cmd_width;
            col += cmd_width;
            wide_tiles[row_start + xtile].push(Cmd::Strip(cmd));
        }

        let active_fill = match fill_rule {
            Fill::NonZero => next_strip.winding != 0,
            Fill::EvenOdd => next_strip.winding % 2 != 0,
        };

        if active_fill
            && y == next_strip.strip_y()
            // Only fill if we are actually inside the viewport.
            && next_strip.x >= 0
        {
            x = x1;
            let x2 = next_strip.x as u32;
            let fxt0 = x1 as usize / WIDE_TILE_WIDTH;
            let fxt1 = (x2 as usize).div_ceil(WIDE_TILE_WIDTH);
            for xtile in fxt0..fxt1 {
                let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                let cmd_width = x2.min(((xtile + 1) * WIDE_TILE_WIDTH) as u32) - x;
                x += cmd_width;
                wide_tiles[row_start + xtile].fill(x_tile_rel, cmd_width, paint.clone());
            }
        }
    }
}
