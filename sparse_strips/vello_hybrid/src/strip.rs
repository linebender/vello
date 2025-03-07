// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU implementation of sparse strip rendering
//!
//! This is copied from the most recent GPU implementation, but has
//! `path_id` stripped out, as on CPU we'll be doing one path at a time.
//! That decision makes sense to some extent even when uploading to
//! GPU, though some mechanism is required to tie the strips to paint.
//!
//! If there becomes a single, unified code base for this, then the
//! `path_id` type should probably become a generic parameter.

use crate::{tiling::Vec2, wide_tile::STRIP_HEIGHT};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Loc {
    x: u16,
    y: u16,
}

pub(crate) struct Footprint(pub(crate) u32);

pub(crate) struct Tile {
    pub x: u16,
    pub y: u16,
    pub p0: u32, // packed
    pub p1: u32, // packed
}

impl std::fmt::Debug for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p0 = Vec2::unpack(self.p0);
        let p1 = Vec2::unpack(self.p1);
        write!(
            f,
            "Tile {{ xy: ({}, {}), p0: ({:.4}, {:.4}), p1: ({:.4}, {:.4}) }}",
            self.x, self.y, p0.x, p0.y, p1.x, p1.y
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Strip {
    pub xy: u32, // this could be u16's on the Rust side
    pub col: u32,
    pub winding: i32,
}

impl Loc {
    pub(crate) fn same_strip(&self, other: &Self) -> bool {
        self.same_row(other) && (other.x - self.x) / 2 == 0
    }

    pub(crate) fn same_row(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

impl Tile {
    #[allow(unused, reason = "only used for synthetic data")]
    /// Create a tile from synthetic data.
    fn new(loc: Loc, footprint: Footprint, delta: i32) -> Self {
        let p0 = (delta == -1) as u32 * 65536 + footprint.0.trailing_zeros() * 8192;
        let p1 = (delta == 1) as u32 * 65536 + (32 - footprint.0.leading_zeros()) * 8192;
        Self {
            x: loc.x,
            y: loc.y,
            p0,
            p1,
        }
    }

    pub(crate) fn loc(&self) -> Loc {
        Loc {
            x: self.x,
            y: self.y,
        }
    }

    pub(crate) fn footprint(&self) -> Footprint {
        let x0 = (self.p0 & 0xffff) as f32 * (1.0 / 8192.0);
        let x1 = (self.p1 & 0xffff) as f32 * (1.0 / 8192.0);
        // On CPU, might be better to do this as fixed point
        let xmin = x0.min(x1).floor() as u32;
        let xmax = (xmin + 1).max(x0.max(x1).ceil() as u32);
        Footprint((1 << xmax) - (1 << xmin))
    }

    pub(crate) fn delta(&self) -> i32 {
        ((self.p1 >> 16) == 0) as i32 - ((self.p0 >> 16) == 0) as i32
    }

    // Comparison function for sorting. Only compares loc, doesn't care
    // about points. Unpacking code has been validated to be efficient in
    // Godbolt.
    pub(crate) fn cmp(&self, b: &Self) -> std::cmp::Ordering {
        let xya = ((self.y as u32) << 16) + (self.x as u32);
        let xyb = ((b.y as u32) << 16) + (b.x as u32);
        xya.cmp(&xyb)
    }
}

pub(crate) fn render_strips_scalar(
    tiles: &[Tile],
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u32>,
) {
    strip_buf.clear();
    let mut strip_start = true;
    let mut cols = alpha_buf.len() as u32;
    let mut prev_tile = &tiles[0];
    let mut fp = prev_tile.footprint().0;
    let mut seg_start = 0;
    let mut delta = 0;
    // Note: the input should contain a sentinel tile, to avoid having
    // logic here to process the final strip.
    for i in 1..tiles.len() {
        let tile = &tiles[i];
        if prev_tile.loc() != tile.loc() {
            let start_delta = delta;
            let same_strip = prev_tile.loc().same_strip(&tile.loc());
            if same_strip {
                fp |= 8;
            }
            let x0 = fp.trailing_zeros();
            let x1 = (32 - fp.leading_zeros()).min(4);
            let mut areas = [[start_delta as f32; 4]; 4];
            for this_tile in &tiles[seg_start..i] {
                delta += this_tile.delta();
                let p0 = Vec2::unpack(this_tile.p0);
                let p1 = Vec2::unpack(this_tile.p1);
                let slope = (p1.x - p0.x) / (p1.y - p0.y);
                for x in x0..x1 {
                    let startx = p0.x - x as f32;
                    for y in 0..4 {
                        let starty = p0.y - y as f32;
                        let y0 = starty.clamp(0.0, 1.0);
                        let y1 = (p1.y - y as f32).clamp(0.0, 1.0);
                        let dy = y0 - y1;
                        // Note: getting rid of this predicate might help with
                        // auto-vectorization. That said, just getting rid of
                        // it causes artifacts (which may be divide by zero).
                        if dy != 0.0 {
                            let xx0 = startx + (y0 - starty) * slope;
                            let xx1 = startx + (y1 - starty) * slope;
                            let xmin0 = xx0.min(xx1);
                            let xmax = xx0.max(xx1);
                            let xmin = xmin0.min(1.0) - 1e-6;
                            let b = xmax.min(1.0);
                            let c = b.max(0.0);
                            let d = xmin.max(0.0);
                            let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                            areas[x as usize][y] += a * dy;
                        }
                        if p0.x == 0.0 {
                            areas[x as usize][y] += (y as f32 - p0.y + 1.0).clamp(0.0, 1.0);
                        } else if p1.x == 0.0 {
                            areas[x as usize][y] -= (y as f32 - p1.y + 1.0).clamp(0.0, 1.0);
                        }
                    }
                }
            }
            for x in x0..x1 {
                let mut alphas = 0_u32;
                for y in 0..4 {
                    let area = areas[x as usize][y];
                    // nonzero winding number rule
                    let area_u8 = (area.abs().min(1.0) * 255.0).round() as u32;
                    alphas += area_u8 << (y * 8);
                }
                alpha_buf.push(alphas);
            }

            if strip_start {
                let xy = (1 << 18) * prev_tile.y as u32 + 4 * prev_tile.x as u32 + x0;
                let strip = Strip {
                    xy,
                    col: cols,
                    winding: start_delta,
                };
                strip_buf.push(strip);
            }
            cols += x1 - x0;
            fp = if same_strip { 1 } else { 0 };
            strip_start = !same_strip;
            seg_start = i;
            if !prev_tile.loc().same_row(&tile.loc()) {
                delta = 0;
            }
        }
        fp |= tile.footprint().0;
        prev_tile = tile;
    }
}

impl Strip {
    pub(crate) fn x(&self) -> u32 {
        self.xy & 0xffff
    }

    pub(crate) fn strip_y(&self) -> u32 {
        self.xy / ((1 << 16) * STRIP_HEIGHT as u32)
    }
}
