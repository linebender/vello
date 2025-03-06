// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Remove when all methods are implemented.
#![allow(unused, reason = "lots of unused arguments from todo methods")]

use std::collections::BTreeMap;

use peniko::{
    color::{palette, AlphaColor, Srgb},
    kurbo::Affine,
    BrushRef,
};
use vello_cpu::Pixmap;

use crate::{
    fine::Fine,
    strip::{self, Strip, Tile},
    tiling::{self, FlatLine},
    wide_tile::{Cmd, CmdClipStrip, CmdStrip, WideTile, STRIP_HEIGHT, WIDE_TILE_WIDTH},
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

    state_stack: Vec<GfxState>,
    clip_stack: Vec<Clip>,
}

struct GfxState {
    // TODO: transform goes here (there's logic in piet-ts to copy)
    n_clip: usize,
}

struct Clip {
    // should probably be a bounding box type
    /// The intersected bounding box after clip
    clip_bbox: [usize; 4],
    /// The rendered path in sparse strip representation
    strips: Vec<Strip>,
}

impl RenderContext {
    pub fn new(width: usize, height: usize) -> Self {
        let width_tiles = width.div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = height.div_ceil(STRIP_HEIGHT);
        let tiles = (0..width_tiles * height_tiles)
            .map(|_| WideTile::default())
            .collect();
        let state = GfxState { n_clip: 0 };
        Self {
            width,
            height,
            tiles,
            alphas: vec![],
            line_buf: vec![],
            tile_buf: vec![],
            strip_buf: vec![],
            state_stack: vec![state],
            clip_stack: vec![],
        }
    }

    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.bg = AlphaColor::TRANSPARENT;
            tile.cmds.clear();
        }
    }

    /// Finish the coarse rasterization prior to fine rendering.
    ///
    /// At the moment, this mostly involves resolving any open clips, but
    /// might extend to other things.
    pub(crate) fn finish(&mut self) {
        self.pop_clips();
    }

    pub fn render_to_pixmap(&mut self, pixmap: &mut Pixmap) {
        self.finish();
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

    /// Render a path to the strip buffer.
    fn render_path_common(&mut self) {
        tiling::make_tiles(&self.line_buf, &mut self.tile_buf);
        self.tile_buf.sort_unstable_by(Tile::cmp);
        crate::strip::render_strips_scalar(&self.tile_buf, &mut self.strip_buf, &mut self.alphas);
    }

    /// Render a path, which has already been flattened into `line_buf`.
    fn render_path(&mut self, brush: BrushRef<'_>) {
        // TODO: need to make sure tiles contained in viewport - we'll likely
        // panic otherwise.
        self.render_path_common();
        let color = brush_to_color(brush);
        let width_tiles = self.width.div_ceil(WIDE_TILE_WIDTH);
        let bbox = self.get_bbox();
        for i in 0..self.strip_buf.len() - 1 {
            let strip = &self.strip_buf[i];
            let next_strip = &self.strip_buf[i + 1];
            let x0 = strip.x();
            let y = strip.strip_y() as usize;
            if y < bbox[1] {
                continue;
            }
            if y >= bbox[3] {
                break;
            }
            let row_start = y * width_tiles;
            let strip_width = next_strip.col - strip.col;
            let x1 = x0 + strip_width;
            let xtile0 = (x0 as usize / WIDE_TILE_WIDTH).max(bbox[0]);
            let xtile1 = (x1 as usize).div_ceil(WIDE_TILE_WIDTH).min(bbox[2]);
            let mut x = x0;
            let mut col = strip.col;
            if (bbox[0] * WIDE_TILE_WIDTH) as u32 > x {
                col += (bbox[0] * WIDE_TILE_WIDTH) as u32 - x;
                x = (bbox[0] * WIDE_TILE_WIDTH) as u32;
            }
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
                self.tiles[row_start + xtile].strip(cmd);
            }
            if next_strip.winding != 0 && y == next_strip.strip_y() as usize {
                x = x1;
                let x2 = next_strip.x();
                let fxt0 = (x1 as usize / WIDE_TILE_WIDTH).max(bbox[0]);
                let fxt1 = (x2 as usize).div_ceil(WIDE_TILE_WIDTH).min(bbox[2]);
                for xtile in fxt0..fxt1 {
                    let x_tile_rel = x % WIDE_TILE_WIDTH as u32;
                    let width = x2.min(((xtile + 1) * WIDE_TILE_WIDTH) as u32) - x;
                    x += width;
                    self.tiles[row_start + xtile].fill(x_tile_rel, width, color);
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

    fn get_bbox(&self) -> [usize; 4] {
        if let Some(tos) = self.clip_stack.last() {
            tos.clip_bbox
        } else {
            let width_tiles = (self.width).div_ceil(WIDE_TILE_WIDTH);
            let height_tiles = (self.height).div_ceil(STRIP_HEIGHT);
            [0, 0, width_tiles, height_tiles]
        }
    }

    fn pop_clip(&mut self) {
        self.state_stack.last_mut().unwrap().n_clip -= 1;
        let Clip { clip_bbox, strips } = self.clip_stack.pop().unwrap();
        let n_strips = strips.len();
        // The next bit of code accomplishes the following. For each tile in
        // the intersected bounding box, it does one of three things depending
        // on the contents of the clip path in that tile.
        // If all-zero: pop a zero_clip.
        // If all-one: do nothing.
        // If contains one or more strips: render strips and fills, then pop a clip.
        // This logic is the inverse of the push logic in `clip()`, and the stack
        // should be balanced after running both.
        let mut tile_x = clip_bbox[0];
        let mut tile_y = clip_bbox[1];
        let width_tiles = (self.width).div_ceil(WIDE_TILE_WIDTH);
        let mut pop_pending = false;
        for i in 0..n_strips - 1 {
            let strip = &strips[i];
            let y = strip.strip_y() as usize;
            if y < tile_y {
                continue;
            }
            while tile_y < y.min(clip_bbox[3]) {
                if core::mem::take(&mut pop_pending) {
                    self.tiles[tile_y * width_tiles + tile_x].pop_clip();
                    tile_x += 1;
                }
                for x in tile_x..clip_bbox[2] {
                    self.tiles[tile_y * width_tiles + x].pop_zero_clip();
                }
                tile_x = clip_bbox[0];
                tile_y += 1;
            }
            if tile_y == clip_bbox[3] {
                break;
            }
            let x0 = strip.x() as usize;
            let x_clamped = (x0 / WIDE_TILE_WIDTH).min(clip_bbox[2]);
            if tile_x < x_clamped {
                if core::mem::take(&mut pop_pending) {
                    self.tiles[tile_y * width_tiles + tile_x].pop_clip();
                    tile_x += 1;
                }
                // The winding check is probably not needed; if there was a fill,
                // the logic below should have advanced tile_x.
                if strip.winding == 0 {
                    for x in tile_x..x_clamped {
                        self.tiles[tile_y * width_tiles + x].pop_zero_clip();
                    }
                }
                tile_x = x_clamped;
            }
            let next_strip = &strips[i + 1];
            let strip_width = (next_strip.col - strip.col) as usize;
            let x1 = x0 + strip_width;
            let xtile0 = (x0 / WIDE_TILE_WIDTH).max(clip_bbox[0]);
            let xtile1 = x1.div_ceil(WIDE_TILE_WIDTH).min(clip_bbox[2]);
            let mut x = x0;
            let mut alpha_ix = strip.col as usize;
            if clip_bbox[0] * WIDE_TILE_WIDTH > x {
                alpha_ix += clip_bbox[0] * WIDE_TILE_WIDTH - x;
                x = clip_bbox[0] * WIDE_TILE_WIDTH;
            }
            for xtile in xtile0..xtile1 {
                if xtile > tile_x && core::mem::take(&mut pop_pending) {
                    self.tiles[tile_y * width_tiles + tile_x].pop_clip();
                }
                let x_tile_rel = (x % WIDE_TILE_WIDTH) as u32;
                let width = x1.min((xtile + 1) * WIDE_TILE_WIDTH) - x;
                let cmd = CmdClipStrip {
                    x: x_tile_rel,
                    width: width as u32,
                    alpha_ix,
                };
                x += width;
                alpha_ix += width;
                self.tiles[tile_y * width_tiles + xtile].clip_strip(cmd);
                tile_x = xtile;
                pop_pending = true;
            }
            if next_strip.winding != 0 && y == next_strip.strip_y() as usize {
                let x2 = next_strip.x() as usize;
                let tile_x2 = x2.min((tile_x + 1) * WIDE_TILE_WIDTH);
                let width = tile_x2 - x1;
                if width > 0 {
                    let x_tile_rel = (x1 % WIDE_TILE_WIDTH) as u32;
                    self.tiles[tile_y * width_tiles + tile_x].clip_fill(x_tile_rel, width as u32);
                }
                if x2 > (tile_x + 1) * WIDE_TILE_WIDTH {
                    self.tiles[tile_y * width_tiles + tile_x].pop_clip();
                    let width2 = x2 % WIDE_TILE_WIDTH;
                    tile_x = x2 / WIDE_TILE_WIDTH;
                    if width2 > 0 {
                        self.tiles[tile_y * width_tiles + tile_x].clip_fill(0, width2 as u32);
                    }
                }
            }
        }
        if core::mem::take(&mut pop_pending) {
            self.tiles[tile_y * width_tiles + tile_x].pop_clip();
            tile_x += 1;
        }
        while tile_y < clip_bbox[3] {
            for x in tile_x..clip_bbox[2] {
                self.tiles[tile_y * width_tiles + x].pop_zero_clip();
            }
            tile_x = clip_bbox[0];
            tile_y += 1;
        }
    }

    fn pop_clips(&mut self) {
        while self.state_stack.last().unwrap().n_clip > 0 {
            self.pop_clip();
        }
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

    pub fn clip(&mut self, path: &crate::common::Path) {
        let affine = self.get_affine();
        crate::flatten::fill(&path.path, affine, &mut self.line_buf);
        self.render_path_common();
        let strips = core::mem::take(&mut self.strip_buf);
        let n_strips = strips.len();
        let path_bbox = if n_strips <= 1 {
            [0, 0, 0, 0]
        } else {
            let y0 = strips[0].strip_y() as usize;
            let y1 = strips[n_strips - 1].strip_y() as usize + 1;
            let mut x0 = strips[0].x() as usize / WIDE_TILE_WIDTH;
            let mut x1 = x0;
            for i in 0..n_strips - 1 {
                let strip = &strips[i];
                let next_strip = &strips[i + 1];
                let width = next_strip.col - strip.col;
                let x = strip.x() as usize;
                x0 = x0.min(x / WIDE_TILE_WIDTH);
                x1 = x1.max((x + width as usize).div_ceil(WIDE_TILE_WIDTH));
            }
            [x0, x1, y0, y1]
        };
        let parent_bbox = self.get_bbox();
        // intersect clip bounding box
        let clip_bbox = [
            parent_bbox[0].max(path_bbox[0]),
            parent_bbox[1].max(path_bbox[1]),
            parent_bbox[2].min(path_bbox[2]),
            parent_bbox[3].min(path_bbox[3]),
        ];
        // The next bit of code accomplishes the following. For each tile in
        // the intersected bounding box, it does one of three things depending
        // on the contents of the clip path in that tile.
        // If all-zero: push a zero_clip
        // If all-one: do nothing
        // If contains one or more strips: push a clip
        let mut tile_x = clip_bbox[0];
        let mut tile_y = clip_bbox[1];
        let width_tiles = (self.width).div_ceil(WIDE_TILE_WIDTH);
        for i in 0..n_strips - 1 {
            let strip = &strips[i];
            let y = strip.strip_y() as usize;
            if y < tile_y {
                continue;
            }
            while tile_y < y.min(clip_bbox[3]) {
                for x in tile_x..clip_bbox[2] {
                    self.tiles[tile_y * width_tiles + x].push_zero_clip();
                }
                tile_x = clip_bbox[0];
                tile_y += 1;
            }
            if tile_y == clip_bbox[3] {
                break;
            }
            let x_pixels = strip.x() as usize;
            let x_clamped = (x_pixels / WIDE_TILE_WIDTH).min(clip_bbox[2]);
            if tile_x < x_clamped {
                if strip.winding == 0 {
                    for x in tile_x..x_clamped {
                        self.tiles[tile_y * width_tiles + x].push_zero_clip();
                    }
                }
                // If winding is nonzero, then wide tiles covered entirely
                // by sparse fill are no-op (no clipping is applied).
                tile_x = x_clamped;
            }
            let next_strip = &strips[i + 1];
            let width = (next_strip.col - strip.col) as usize;
            let x1 = (x_pixels + width)
                .div_ceil(WIDE_TILE_WIDTH)
                .min(clip_bbox[2]);
            if tile_x < x1 {
                for x in tile_x..x1 {
                    self.tiles[tile_y * width_tiles + x].push_clip();
                }
                tile_x = x1;
            }
        }
        while tile_y < clip_bbox[3] {
            for x in tile_x..clip_bbox[2] {
                self.tiles[tile_y * width_tiles + x].push_zero_clip();
            }
            tile_x = clip_bbox[0];
            tile_y += 1;
        }
        let clip = Clip { clip_bbox, strips };
        self.clip_stack.push(clip);
        self.state_stack.last_mut().unwrap().n_clip += 1;
    }

    pub fn save(&mut self) {
        self.state_stack.push(GfxState { n_clip: 0 });
    }

    pub fn restore(&mut self) {
        self.pop_clips();
        self.state_stack.pop();
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
