// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Remove when all methods are implemented.
#![allow(unused, reason = "lots of unused arguments from todo methods")]

use std::collections::BTreeMap;

use peniko::{
    BrushRef,
    color::{AlphaColor, Srgb, palette},
    kurbo::Affine,
};
use vello_cpu::Pixmap;

use crate::{
    fine::Fine,
    strip::{self, Strip, Tile},
    tiling::{self, FlatLine},
    wide_tile::{Cmd, CmdStrip, STRIP_HEIGHT, WIDE_TILE_WIDTH, WideTile},
};

pub struct RenderContext {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) tiles: Vec<WideTile>,
    pub(crate) alphas: Vec<u32>, /*  */

    /// These are all scratch buffers, to be used for path rendering. They're here solely
    /// so the allocations can be reused.
    line_buf: Vec<FlatLine>,
    tile_buf: Vec<Tile>,
    strip_buf: Vec<Strip>,
}

impl RenderContext {
    pub fn new(width: usize, height: usize) -> Self {
        let width_tiles = width.div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = height.div_ceil(STRIP_HEIGHT);
        let tiles = (0..width_tiles * height_tiles)
            .map(|_| WideTile::default())
            .collect();
        let alphas = vec![];
        let line_buf = vec![];
        let tile_buf = vec![];
        let strip_buf = vec![];
        Self {
            width,
            height,
            tiles,
            alphas,
            line_buf,
            tile_buf,
            strip_buf,
        }
    }

    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.bg = AlphaColor::TRANSPARENT;
            tile.cmds.clear();
        }
    }

    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        let mut fine = Fine::new(
            pixmap.width as usize,
            pixmap.height as usize,
            &mut pixmap.buf,
        );
        let width_tiles = (self.width).div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = (self.height).div_ceil(STRIP_HEIGHT);
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let tile = &self.tiles[y * width_tiles + x];
                fine.clear_scalar(tile.bg.components);
                for cmd in &tile.cmds {
                    fine.run_cmd(cmd, &self.alphas);
                }
                fine.pack_scalar(x, y);
            }
        }
    }

    pub fn tile_stats(&self) {
        let mut histo = BTreeMap::new();
        let mut total = 0;
        for tile in &self.tiles {
            let count = tile.cmds.len();
            total += count;
            *histo.entry(count).or_insert(0) += 1;
        }
        println!("total = {total}, {histo:?}");
    }

    /// Render a path, which has already been flattened into `line_buf`.
    fn render_path(&mut self, brush: BrushRef<'_>) {
        // TODO: need to make sure tiles contained in viewport - we'll likely
        // panic otherwise.
        tiling::make_tiles(&self.line_buf, &mut self.tile_buf);
        self.tile_buf.sort_unstable_by(Tile::cmp);
        strip::render_strips_scalar(&self.tile_buf, &mut self.strip_buf, &mut self.alphas);
        let color = brush_to_color(brush);
        let width_tiles = self.width.div_ceil(WIDE_TILE_WIDTH);
        for i in 0..self.strip_buf.len() - 1 {
            let strip = &self.strip_buf[i];
            let next_strip = &self.strip_buf[i + 1];
            let x0 = strip.x();
            let y = strip.strip_y();
            let row_start = y as usize * width_tiles;
            let strip_width = next_strip.col - strip.col;
            let x1 = x0 + strip_width;
            let xtile0 = x0 as usize / WIDE_TILE_WIDTH;
            let xtile1 = (x1 as usize).div_ceil(WIDE_TILE_WIDTH);
            let mut x = x0;
            let mut col = strip.col;
            for xtile in xtile0..xtile1 {
                let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                let width = x1.min(((xtile + 1) * WIDE_TILE_WIDTH) as u32) - x;
                let cmd = CmdStrip {
                    x: x_tile_rel,
                    width,
                    alpha_ix: col as usize,
                    color,
                };
                x += width;
                col += width;
                let tile_index = row_start + xtile;
                if tile_index < self.tiles.len() {
                    self.tiles[tile_index].push(Cmd::Strip(cmd));
                }
            }
            if next_strip.winding != 0 && y == next_strip.strip_y() {
                x = x1;
                let x2 = next_strip.x();
                let fxt0 = x1 as usize / WIDE_TILE_WIDTH;
                let fxt1 = (x2 as usize).div_ceil(WIDE_TILE_WIDTH);
                for xtile in fxt0..fxt1 {
                    let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                    let width = x2.min(((xtile + 1) * WIDE_TILE_WIDTH) as u32) - x;
                    x += width;
                    let tile_index = row_start + xtile;
                    if tile_index < self.tiles.len() {
                        self.tiles[tile_index].fill(x_tile_rel, width, color);
                    }
                }
            }
        }
    }

    pub fn debug_dump(&self) {
        let width_tiles = self.width.div_ceil(WIDE_TILE_WIDTH);
        for (i, tile) in self.tiles.iter().enumerate() {
            if !tile.cmds.is_empty() || tile.bg.components[3] != 0.0 {
                let x = i % width_tiles;
                let y = i / width_tiles;
                println!("tile {x}, {y} bg {}", tile.bg.to_rgba8());
                for cmd in &tile.cmds {
                    println!("{cmd:?}");
                }
            }
        }
    }

    fn get_affine(&self) -> Affine {
        // TODO: get from graphics state
        Affine::scale(5.0)
    }
}

impl RenderContext {
    pub fn fill(&mut self, path: &crate::common::Path, brush: BrushRef<'_>) {
        let affine = self.get_affine();
        crate::flatten::fill(&path.path, affine, &mut self.line_buf);
        self.render_path(brush);
    }

    pub fn stroke(
        &mut self,
        path: &crate::common::Path,
        stroke: &peniko::kurbo::Stroke,
        brush: BrushRef<'_>,
    ) {
        let affine = self.get_affine();
        crate::flatten::stroke(&path.path, stroke, affine, &mut self.line_buf);
        self.render_path(brush);
    }

    pub fn draw_image(&mut self) {
        todo!()
    }

    pub fn clip(&mut self) {
        todo!()
    }

    pub fn save(&mut self) {
        todo!()
    }

    pub fn restore(&mut self) {
        todo!()
    }

    pub fn transform(&mut self, affine: peniko::kurbo::Affine) {
        todo!()
    }

    pub fn begin_draw_glyphs(&mut self, font: &peniko::Font) {
        todo!()
    }

    pub fn font_size(&mut self, size: f32) {
        todo!()
    }

    pub fn hint(&mut self, hint: bool) {
        todo!()
    }

    pub fn glyph_brush(&mut self, brush: BrushRef<'_>) {
        todo!()
    }

    pub fn draw_glyphs(&mut self) {
        todo!()
    }

    pub fn end_draw_glyphs(&mut self) {
        todo!()
    }
}

/// Get the color from the brush.
///
/// This is a hacky function that will go away when we implement
/// other brushes. The general form is to match on whether it's a
/// solid color. If not, then issue a cmd to render the brush into
/// a brush buffer, then fill/strip as needed to composite into
/// the main buffer.
fn brush_to_color(brush: BrushRef<'_>) -> AlphaColor<Srgb> {
    match brush {
        BrushRef::Solid(c) => c,
        _ => palette::css::MAGENTA,
    }
}
