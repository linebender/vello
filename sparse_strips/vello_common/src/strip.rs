// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering strips.
use crate::flatten::{Line, Point};
use crate::peniko::Fill;
use crate::tile::{Tile, Tiles};
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use core::{f32, panic};
use fearless_simd::*;
use std::format;
use std::ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Div, Mul, Sub, SubAssign};
use std::println;
use std::string::String;
use std::vec;

// TODO make the tile simply hold the extra data

/// A strip.
#[derive(Debug, Clone, Copy)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: u16,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// Packed alpha index and fill gap flag.
    ///
    /// Bit layout (u32):
    /// - bit 31: `fill_gap` (See `Strip::fill_gap()`).
    /// - bits 0..=30: `alpha_idx` (See `Strip::alpha_idx()`).
    packed_alpha_idx_fill_gap: u32,
}

impl Strip {
    /// The bit mask for `fill_gap` packed into `packed_alpha_idx_fill_gap`.
    const FILL_GAP_MASK: u32 = 1 << 31;

    /// Creates a new strip.
    pub fn new(x: u16, y: u16, alpha_idx: u32, fill_gap: bool) -> Self {
        // Ensure `alpha_idx` does not collide with the fill flag bit.
        assert!(
            alpha_idx & Self::FILL_GAP_MASK == 0,
            "`alpha_idx` too large"
        );
        let fill_gap = u32::from(fill_gap) << 31;
        Self {
            x,
            y,
            packed_alpha_idx_fill_gap: alpha_idx | fill_gap,
        }
    }

    pub fn is_sentinel(&self) -> bool {
        self.x == u16::MAX
    }

    /// Returns the y coordinate of the strip, in strip units.
    pub fn strip_y(&self) -> u16 {
        self.y / Tile::HEIGHT
    }

    /// Returns the alpha index.
    #[inline(always)]
    pub fn alpha_idx(&self) -> u32 {
        self.packed_alpha_idx_fill_gap & !Self::FILL_GAP_MASK
    }

    /// Sets the alpha index.
    ///
    /// Note that the largest value that can be stored in the alpha index is `u32::MAX << 1`, as the
    /// highest bit is reserved for `fill_gap`.
    #[inline(always)]
    pub fn set_alpha_idx(&mut self, alpha_idx: u32) {
        // Ensure `alpha_idx` does not collide with the fill flag bit.
        assert!(
            alpha_idx & Self::FILL_GAP_MASK == 0,
            "`alpha_idx` too large"
        );
        let fill_gap = self.packed_alpha_idx_fill_gap & Self::FILL_GAP_MASK;
        self.packed_alpha_idx_fill_gap = alpha_idx | fill_gap;
    }

    /// Returns whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn fill_gap(&self) -> bool {
        (self.packed_alpha_idx_fill_gap & Self::FILL_GAP_MASK) != 0
    }

    /// Sets whether the gap that lies between this strip and the previous in the same row should be filled.
    #[inline(always)]
    pub fn set_fill_gap(&mut self, fill: bool) {
        let fill = u32::from(fill) << 31;
        self.packed_alpha_idx_fill_gap =
            (self.packed_alpha_idx_fill_gap & !Self::FILL_GAP_MASK) | fill;
    }
}

impl PartialEq for Strip {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x &&
        self.y == other.y &&
        self.packed_alpha_idx_fill_gap == other.packed_alpha_idx_fill_gap
    }
}

impl Eq for Strip {}

// Intersection Layout (6 bits)
// MSB                   LSB
// | 5 | 4 | 3 | 2 | 1 | 0 |
// | W | P | R | L | B | T |
pub const INTERSECTS_TOP_MASK: u32 = 1 << 0;
pub const INTERSECTS_BOTTOM_MASK: u32 = 1 << 1;
pub const INTERSECTS_LEFT_MASK: u32 = 1 << 2;
pub const INTERSECTS_RIGHT_MASK: u32 = 1 << 3;
pub const PERFECT_MASK: u32 = 1 << 4; // Formerly "Ambiguous" or Perfect Corner
pub const WINDING_MASK: u32 = 1 << 5; // Winding direction bit

// Masks for packed_winding_line_idx
const INTERSECTION_DATA_WIDTH: u32 = 6;
const INTERSECTION_DATA_MASK: u32 = (1 << INTERSECTION_DATA_WIDTH) - 1;

// Masks for packed_scanned_winding
const IS_END_TILE_BIT: u32 = 0;
const FILL_RULE_BIT: u32 = 1;
// Scanned winding moved to bits 8-15 to make room for 9-bit ID at top
const SCANNED_WINDING_SHIFT: u32 = 8;
const SCANNED_WINDING_MASK: u32 = 0xFF; // i8
// Start ID moved to upper 9 bits (23-31)
const TILE_START_ID_SHIFT: u32 = 23;
const TILE_START_ID_MASK: u32 = 0x1FF; // 9 bits (0-511)

/// Pre-Merge-Tile:
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreMergeTile {
    /// The index into the alpha buffer that an "end tile" should write to.
    pub alpha_index: u32,

    /// Contains the Intersection Data and the Line Buffer Index.
    /// MSB                                                           LSB
    /// 31------------------------------------------------------6|5----0|
    /// |             Line Buffer Index (26 bits)                | Ints |
    ///
    /// - **Ints**: Intersection Data (6 bits)
    pub packed_winding_line_idx: u32,

    /// Contains the Scanned Winding, Tile Start ID, and Flags.
    /// MSB                                                          LSB
    /// 31-----------------23|22-------16|15--------8|7-----------2|1|0|
    /// |   Start ID (9b)    |   Unused  | Scan Wind |   Unused    |F|E|
    ///
    /// - **Start ID**: u16 (Tile Start ID, 9 bits)
    /// - **Scan Wind**: i8 (Exclusive prefix sum)
    /// - **F**: Fill Rule (bool)
    /// - **E**: Is End Tile (bool)
    pub packed_scanned_winding: u32,

    /// Contains the packed u16 tile x and y coordinates.
    /// MSB                  LSB
    /// 31-------16|15--------0|
    /// |  Tile Y  |  Tile X   |
    pub packed_xy: u32,

    pub line: Line,
}

impl PreMergeTile {
    pub fn set_tile_xy(&mut self, x: u16, y: u16) {
        self.packed_xy = ((y as u32) << 16) | (x as u32);
    }

    #[inline]
    pub const fn tile_x(&self) -> u16 {
        (self.packed_xy & 0xFFFF) as u16
    }

    #[inline]
    pub const fn tile_y(&self) -> u16 {
        (self.packed_xy >> 16) as u16
    }

    pub fn pack_winding_and_line(&mut self, line_idx: u32, intersection_data: u32) {
        self.packed_winding_line_idx =
            (line_idx << INTERSECTION_DATA_WIDTH) | (intersection_data & INTERSECTION_DATA_MASK);
    }

    pub fn set_line_index(&mut self, idx: u32) {
        let intersection = self.packed_winding_line_idx & INTERSECTION_DATA_MASK;
        self.packed_winding_line_idx = (idx << INTERSECTION_DATA_WIDTH) | intersection;
    }

    #[inline]
    pub const fn line_index(&self) -> u32 {
        self.packed_winding_line_idx >> INTERSECTION_DATA_WIDTH
    }

    pub fn set_intersection_data(&mut self, data: u32) {
        let line_idx = self.packed_winding_line_idx & !INTERSECTION_DATA_MASK;
        self.packed_winding_line_idx = line_idx | (data & INTERSECTION_DATA_MASK);
    }

    #[inline]
    pub const fn intersection_data(&self) -> u32 {
        self.packed_winding_line_idx & INTERSECTION_DATA_MASK
    }

    #[inline]
    pub const fn intersects_top(&self) -> bool {
        (self.intersection_data() & INTERSECTS_TOP_MASK) != 0
    }

    #[inline]
    pub const fn intersects_bottom(&self) -> bool {
        (self.intersection_data() & INTERSECTS_BOTTOM_MASK) != 0
    }

    #[inline]
    pub const fn intersects_left(&self) -> bool {
        (self.intersection_data() & INTERSECTS_LEFT_MASK) != 0
    }

    #[inline]
    pub const fn intersects_right(&self) -> bool {
        (self.intersection_data() & INTERSECTS_RIGHT_MASK) != 0
    }

    #[inline]
    pub const fn is_perfect(&self) -> bool {
        (self.intersection_data() & PERFECT_MASK) != 0
    }

    #[inline]
    pub const fn winding_bit(&self) -> bool {
        (self.intersection_data() & WINDING_MASK) != 0
    }

    pub fn set_is_end_tile(&mut self, is_end: bool) {
        if is_end {
            self.packed_scanned_winding |= 1 << IS_END_TILE_BIT;
        } else {
            self.packed_scanned_winding &= !(1 << IS_END_TILE_BIT);
        }
    }

    #[inline]
    pub const fn is_end_tile(&self) -> bool {
        (self.packed_scanned_winding & (1 << IS_END_TILE_BIT)) != 0
    }

    pub fn set_fill_rule(&mut self, is_non_zero: bool) {
        if is_non_zero {
            self.packed_scanned_winding |= 1 << FILL_RULE_BIT;
        } else {
            self.packed_scanned_winding &= !(1 << FILL_RULE_BIT);
        }
    }

    #[inline]
    pub const fn fill_rule(&self) -> bool {
        (self.packed_scanned_winding & (1 << FILL_RULE_BIT)) != 0
    }

    pub fn set_tile_start_id(&mut self, id: u16) {
        self.packed_scanned_winding |= ((id as u32) & TILE_START_ID_MASK) << TILE_START_ID_SHIFT;
    }

    #[inline]
    pub const fn tile_start_id(&self) -> u16 {
        ((self.packed_scanned_winding >> TILE_START_ID_SHIFT) & TILE_START_ID_MASK) as u16
    }

    pub fn set_scanned_winding(&mut self, winding: i8) {
        // Clear only the scanned winding bits
        self.packed_scanned_winding |= (winding as u8 as u32) << SCANNED_WINDING_SHIFT;
    }

    #[inline]
    pub const fn scanned_winding(&self) -> u32 {
        ((self.packed_scanned_winding >> SCANNED_WINDING_SHIFT) & SCANNED_WINDING_MASK) as i8 as u32
    }
}

impl PartialEq for PreMergeTile {
    fn eq(&self, other: &Self) -> bool {
        let valid_bits_mask =
            (1 << IS_END_TILE_BIT) |
            (1 << FILL_RULE_BIT) |
            (SCANNED_WINDING_MASK << SCANNED_WINDING_SHIFT) |
            (TILE_START_ID_MASK << TILE_START_ID_SHIFT);

        self.alpha_index == other.alpha_index &&
        self.packed_winding_line_idx == other.packed_winding_line_idx &&
        (self.packed_scanned_winding & valid_bits_mask) == (other.packed_scanned_winding & valid_bits_mask) &&
        self.packed_xy == other.packed_xy &&
        self.line == other.line
    }
}

impl Eq for PreMergeTile {}

const MASK_WIDTH: u32 = 64;
const MASK_HEIGHT: u32 = 64;
const SAMPLE_COUNT: usize = 8;
const PACKING_SCALE: f32 = 0.5;

fn algorithm_lut_mask_halfplane(line_p0: [f32; 2], line_p1: [f32; 2], lut: &[u8]) -> u8 {
    let p0 = line_p0;
    let p1 = line_p1;

    let dir = (p1[0] - p0[0], p1[1] - p0[1]);
    let n_unnormalized = (dir.1, -dir.0);

    let mut len = n_unnormalized.0.hypot(n_unnormalized.1);
    if len < 1e-9 {
        len = 1e-9;
    }

    let n = (n_unnormalized.0 / len, n_unnormalized.1 / len);
    let mut c = n.0 * p0[0] + n.1 * p0[1];
    c -= 0.5 * (n.0 + n.1);

    let c_lookup = c;
    let sign = if c < 0.0 { -1.0 } else { 1.0 };

    let c2 = (1.0 - c_lookup * PACKING_SCALE * sign).max(0.0);

    let n_rev = (c2 * n.0 * sign, c2 * n.1 * sign);

    let mut uv = (n_rev.0 * 0.5 + 0.5, n_rev.1 * 0.5 + 0.5);

    if sign < 0.0 && uv.0 == 0.5 {
        uv.0 = 0.5f32.next_down();
    }

    let u_f = (uv.0 * MASK_WIDTH as f32).floor();
    let v_f = (uv.1 * MASK_HEIGHT as f32).floor();

    let mask_width_m1 = (MASK_WIDTH - 1) as i32;
    let mask_height_m1 = (MASK_HEIGHT - 1) as i32;

    let u = (u_f as i32).max(0).min(mask_width_m1) as u32;
    let v = (v_f as i32).max(0).min(mask_height_m1) as u32;

    let index = (v * MASK_WIDTH + u) as usize;
    lut[index]
}

/// Render the tiles stored in `tiles` into the strip and alpha buffer.
pub fn render(
    _level: Level,
    tiles: &Tiles,
    strip_buf: &mut Vec<Strip>,
    pmt_buf: &mut Vec<PreMergeTile>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
    lines: &[Line],
    lut: &Vec<u8>,
) {
    let start_index = pmt_buf.len();
    let fr = Fill::NonZero;
    //let fr = fill_rule;
    prepare_gpu_inputs(
        tiles,
        lines,
        strip_buf,
        pmt_buf,
        alpha_buf,
        fr,
    );

    // This enables the CPU side generation.
    // let lut_u32: &[u32] = bytemuck::cast_slice(&lut);
    // gpu_non_zero(pmt_buf, alpha_buf, lut_u32, start_index);
}

// For emulating the behavior of the shader
const BLOCK_DIM: u32 = 256;
// The number of tiles processed per workgroup, 1 thread per row, 4 rows per tile
const PART_SIZE: u32 = BLOCK_DIM / 4;
const PART_SIZE_LG: u32 = 6;
const INVALID_ID: u32 = 256;
const EPSILON: f32 = 1e-6;

/// Prepares gpu inputs
fn prepare_gpu_inputs(
    tiles: &Tiles,
    lines: &[Line],
    strip_buf: &mut Vec<Strip>,
    pre_merge_buf: &mut Vec<PreMergeTile>,
    alpha_buf: &mut Vec<u8>,
    fill_rule: Fill,
) {
    if tiles.is_empty() {
        return;
    }

    let should_fill = |winding: i8| match fill_rule {
        Fill::NonZero => winding != 0,
        Fill::EvenOdd => winding % 2 != 0,
    };

    let mut winding_delta: i8 = 0;
    let mut prev_tile = *tiles.get(0);

    let mut alpha_offset: u32 = 0;
    let initial_alpha_len = alpha_buf.len() as u32;

    const SENTINEL: Tile = Tile::new(u16::MAX, u16::MAX, 0, 0);

    let mut start_tile_idx = 0;
    let mut strip = Strip::new(
        prev_tile.x * Tile::WIDTH,
        prev_tile.y * Tile::HEIGHT,
        initial_alpha_len,
        should_fill(winding_delta),
    );
    for (tile_idx, tile) in tiles.iter().copied().chain([SENTINEL]).enumerate() {
        let is_start_tile = tile_idx == 0 || !prev_tile.same_loc(&tile);
        let is_start_segment = tile_idx == 0 || (is_start_tile && !prev_tile.prev_loc(&tile));

        if is_start_tile {
            start_tile_idx = pre_merge_buf.len();
            if tile_idx > 0 {
                alpha_offset += (Tile::WIDTH * Tile::HEIGHT) as u32;
            }
        }

        if is_start_segment {
            strip_buf.push(strip);

            let is_sentinel = tile_idx == tiles.len() as usize;
            if !prev_tile.same_row(&tile) {
                if winding_delta != 0 || is_sentinel {
                    strip_buf.push(Strip::new(
                        u16::MAX,
                        prev_tile.y * Tile::HEIGHT,
                        initial_alpha_len + alpha_offset,
                        should_fill(winding_delta),
                    ));
                }
                winding_delta = 0;
            }

            if is_sentinel {
                break;
            }

            strip = Strip::new(
                tile.x * Tile::WIDTH,
                tile.y * Tile::HEIGHT,
                initial_alpha_len + alpha_offset,
                should_fill(winding_delta),
            );
        }

        // TODO: Try making tile and pmt the same?
        // TODO: Move line off the pmt
        let line = lines[tile.line_idx() as usize];
        let mut pmt = PreMergeTile {
            alpha_index: initial_alpha_len + alpha_offset,
            packed_winding_line_idx: tile.packed_winding_line_idx,
            packed_scanned_winding: 0,
            packed_xy: 0,
            line,
        };
        pmt.set_tile_xy(tile.x, tile.y);
        pmt.set_scanned_winding(winding_delta);

        let sign = if line.p1.y >= line.p0.y { 1i8 } else { -1i8 };

        let signed_winding = sign * tile.winding() as i8;
        winding_delta += signed_winding;

        pmt.set_is_end_tile(
            tile_idx == tiles.len() as usize - 1 || !tiles.get(tile_idx as u32 + 1).same_loc(&tile),
        );

        let block_idx = pre_merge_buf.len() >> PART_SIZE_LG << PART_SIZE_LG;
        let start_tile_pack = if start_tile_idx < block_idx {
            INVALID_ID
        } else {
            (start_tile_idx - block_idx) as u32
        };
        pmt.set_tile_start_id(start_tile_pack as u16);
        pmt.set_fill_rule(is_start_tile);

        pre_merge_buf.push(pmt);
        prev_tile = tile;
    }

    alpha_buf.resize(alpha_buf.len() + alpha_offset as usize, 0);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2f {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vec2u {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vec2bool {
    pub x: bool,
    pub y: bool,
}

fn vec2f(x: f32, y: f32) -> Vec2f {
    Vec2f { x, y }
}
fn vec2f_s(s: f32) -> Vec2f {
    Vec2f { x: s, y: s }
} // splat

fn vec2u(x: u32, y: u32) -> Vec2u {
    Vec2u { x, y }
}

fn vec2bool(val: bool) -> Vec2bool {
    Vec2bool { x: val, y: val }
}

impl Vec2f {
    fn yx(&self) -> Vec2f {
        Vec2f {
            x: self.y,
            y: self.x,
        }
    }
    fn xy(&self) -> Vec2f {
        *self
    }
}

trait Selectable<Cond> {
    fn select(false_val: Self, true_val: Self, cond: Cond) -> Self;
}

impl Selectable<bool> for f32 {
    fn select(f: f32, t: f32, cond: bool) -> f32 {
        if cond { t } else { f }
    }
}

impl Selectable<bool> for Vec2f {
    fn select(f: Vec2f, t: Vec2f, cond: bool) -> Vec2f {
        if cond { t } else { f }
    }
}

impl Selectable<Vec2bool> for Vec2f {
    fn select(f: Vec2f, t: Vec2f, cond: Vec2bool) -> Vec2f {
        Vec2f {
            x: if cond.x { t.x } else { f.x },
            y: if cond.y { t.y } else { f.y },
        }
    }
}

impl Selectable<Vec2bool> for Vec2u {
    fn select(f: Vec2u, t: Vec2u, cond: Vec2bool) -> Vec2u {
        Vec2u {
            x: if cond.x { t.x } else { f.x },
            y: if cond.y { t.y } else { f.y },
        }
    }
}

fn select<T, C>(f: T, t: T, c: C) -> T
where
    T: Selectable<C>,
{
    T::select(f, t, c)
}

fn clamp(val: Vec2f, min: Vec2f, max: Vec2f) -> Vec2f {
    Vec2f {
        x: val.x.clamp(min.x, max.x),
        y: val.y.clamp(min.y, max.y),
    }
}

fn any(v: Vec2bool) -> bool {
    v.x || v.y
}

impl Add for Vec2f {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        vec2f(self.x + rhs.x, self.y + rhs.y)
    }
}
impl Sub for Vec2f {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        vec2f(self.x - rhs.x, self.y - rhs.y)
    }
}
impl Mul for Vec2f {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        vec2f(self.x * rhs.x, self.y * rhs.y)
    }
}
impl Mul<f32> for Vec2f {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        vec2f(self.x * rhs, self.y * rhs)
    }
}
impl SubAssign for Vec2f {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Vec2f {
    fn lt(&self, rhs: Vec2f) -> Vec2bool {
        Vec2bool {
            x: self.x < rhs.x,
            y: self.y < rhs.y,
        }
    }
    fn gt(&self, rhs: Vec2f) -> Vec2bool {
        Vec2bool {
            x: self.x > rhs.x,
            y: self.y > rhs.y,
        }
    }
}

impl AddAssign for Vec2u {
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x.wrapping_add(rhs.x);
        self.y = self.y.wrapping_add(rhs.y);
    }
}

impl Add for Vec2u {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        vec2u(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2u {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        vec2u(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2u {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

fn get_msaa_mask(p0: Vec2f, p1: Vec2f, lut: &[u32]) -> u32 {
    let dir = p1 - p0;
    let n_unnorm = (dir.y, -dir.x);
    let len = n_unnorm.0.hypot(n_unnorm.1).max(1e-9);
    let n = (n_unnorm.0 / len, n_unnorm.1 / len);

    let c_raw = (n.0 * p0.x + n.1 * p0.y) - 0.5 * (n.0 + n.1);
    let sign_c = if c_raw < 0.0 { -1.0 } else { 1.0 };
    let c2 = (1.0 - c_raw * PACKING_SCALE * sign_c).max(0.0);
    let n_rev_scale = c2 * sign_c;
    let n_rev = (n.0 * n_rev_scale, n.1 * n_rev_scale);

    let mut uv = (n_rev.0 * 0.5 + 0.5, n_rev.1 * 0.5 + 0.5);
    if sign_c < 0.0 && uv.0 == 0.5 {
        uv.0 = 0.5 - 1e-7;
    }

    let mask_w = MASK_WIDTH as i32;
    let mask_h = MASK_HEIGHT as i32;
    let u_raw = (uv.0 * mask_w as f32).floor() as i32;
    let v_raw = (uv.1 * mask_h as f32).floor() as i32;
    let u = u_raw.max(0).min(mask_w - 1) as u32;
    let v = v_raw.max(0).min(mask_h - 1) as u32;

    let index = v * MASK_WIDTH + u;
    let shift = (index & 3) * 8;
    (lut[(index >> 2) as usize] >> shift) & 0xff
}

fn gpu_non_zero(
    pmt_buf: &mut Vec<PreMergeTile>,
    alpha_buf: &mut Vec<u8>,
    lut: &[u32],
    start_index: usize,
) {
    if pmt_buf.is_empty() || start_index >= pmt_buf.len() {
        return;
    }

    let aligned_start = start_index & !(PART_SIZE as usize - 1);

    let pmt_count = pmt_buf.len() - aligned_start;
    let wg_count = ((pmt_count as u32) + PART_SIZE - 1) / PART_SIZE;
    let total_threads = (wg_count * BLOCK_DIM) as usize;

    let mut temp_mask = vec![
        [[[0x80808080u32; PACK as usize]; Tile::HEIGHT as usize];
            Tile::WIDTH as usize];
        total_threads
    ];

    for wgid in 0..wg_count {
        for tid in 0..BLOCK_DIM {
            let gid = tid + wgid * BLOCK_DIM;
            let pmt_index = (gid >> 2) as usize;
            if pmt_index + aligned_start < start_index || pmt_index >= pmt_count {
                continue;
            }

            let row: u32 = gid & 3;
            let row_index = row as usize;
            let pmt = if pmt_index < pmt_count {
                pmt_buf[pmt_index + aligned_start]
            } else {
                PreMergeTile {
                    alpha_index: 0,
                    packed_winding_line_idx: 0,
                    packed_scanned_winding: 0,
                    packed_xy: 0,
                    line: Line {
                        p0: Point { x: 0.0, y: 0.0 },
                        p1: Point { x: 0.0, y: 0.0 },
                    },
                }
            };

            let line = pmt.line;
            let p0_x = line.p0.x;
            let p0_y = line.p0.y;
            let p1_x = line.p1.x;
            let p1_y = line.p1.y;

            let dx = p1_x - p0_x;
            let dy = p1_y - p0_y;
            let is_vertical = dx.abs() <= f32::EPSILON;
            let is_horizontal = dy.abs() <= f32::EPSILON;
            let idx = select(1.0 / dx, 0.0, is_vertical);
            let idy = select(1.0 / dy, 0.0, is_horizontal);
            let dxdy = dx * idy;
            let dydx = dy * idx;

            let canonical_x_dir = p1_x >= p0_x;
            let canonical_y_dir = p1_y >= p0_y;

            // If the rightmost point of a line is left of the viewport, no further processing is
            // required. The coarse mask is the only dependency which need be passed on.
            let right = if canonical_x_dir { p1_x } else { p0_x };
            let right_in_viewport = right >= 0.0;

            let mut clipped_top = vec2f(0f32, 0f32);
            let mut clipped_bot = vec2f(0f32, 0f32);
            let intersection_data = pmt.intersection_data();
            if right_in_viewport {
                let p0 = vec2f(p0_x, p0_y);
                let p1 = vec2f(p1_x, p1_y);
                let mut p_entry = p0;
                let mut p_exit = p1;

                let tile_min_x_u32 = (pmt.tile_x() * Tile::WIDTH) as u32;
                let tile_min_y_u32 = (pmt.tile_y() * Tile::HEIGHT) as u32;

                let tile_min = vec2f(tile_min_x_u32 as f32, tile_min_y_u32 as f32);
                let tile_max = vec2f(
                    (tile_min_x_u32 + (Tile::WIDTH as u32)) as f32,
                    (tile_min_y_u32 + (Tile::HEIGHT as u32)) as f32,
                );

                let x_canon = vec2bool(canonical_x_dir);
                let v_masks = select(
                    vec2u(INTERSECTS_RIGHT_MASK, INTERSECTS_LEFT_MASK),
                    vec2u(INTERSECTS_LEFT_MASK, INTERSECTS_RIGHT_MASK),
                    x_canon,
                );
                let v_bounds = select(
                    vec2f(tile_max.x, tile_min.x),
                    vec2f(tile_min.x, tile_max.x),
                    x_canon,
                );
                let mask_v_in = v_masks.x;
                let mask_v_out = v_masks.y;
                let bound_v_in = v_bounds.x;
                let bound_v_out = v_bounds.y;

                let y_canon = vec2bool(canonical_y_dir);
                let h_masks = select(
                    vec2u(INTERSECTS_BOTTOM_MASK, INTERSECTS_TOP_MASK),
                    vec2u(INTERSECTS_TOP_MASK, INTERSECTS_BOTTOM_MASK),
                    y_canon,
                );
                let h_bounds = select(
                    vec2f(tile_max.y, tile_min.y),
                    vec2f(tile_min.y, tile_max.y),
                    y_canon,
                );
                let mask_h_in = h_masks.x;
                let mask_h_out = h_masks.y;
                let bound_h_in = h_bounds.x;
                let bound_h_out = h_bounds.y;

                // Entry intersection
                let entry_hits = intersection_data & (mask_v_in | mask_h_in);
                {
                    let use_h = (intersection_data & mask_h_in) != 0;
                    let p_oriented = select(p0.yx(), p0.xy(), use_h);
                    let bound = select(bound_v_in, bound_h_in, use_h);
                    let slope = select(dydx, dxdy, use_h);
                    let calculated = p_oriented.x + (bound - p_oriented.y) * slope;
                    let candidate =
                        select(vec2f(bound, calculated), vec2f(calculated, bound), use_h);
                    p_entry = select(p_entry, candidate, entry_hits != 0);
                }

                // Exit intersection
                let exit_hits = intersection_data & (mask_v_out | mask_h_out);
                {
                    let use_h = (intersection_data & mask_h_out) != 0;
                    let p_oriented = select(p0.yx(), p0.xy(), use_h);
                    let bound = select(bound_v_out, bound_h_out, use_h);
                    let slope = select(dydx, dxdy, use_h);
                    let calculated = p_oriented.x + (bound - p_oriented.y) * slope;
                    let candidate =
                        select(vec2f(bound, calculated), vec2f(calculated, bound), use_h);
                    p_exit = select(p_exit, candidate, exit_hits != 0);
                }

                // This is redundant, see shader comment.
                // Perfect corner logic
                // if (intersection_data & PERFECT_MASK) != 0 && (exit_hits ^ entry_hits) != 0 {
                //     let entry_is_empty = entry_hits == 0;

                //     let target_val = select(p_exit, p_entry, entry_is_empty);
                //     let out_of_bounds =
                //         any(target_val.lt(tile_min)) || any(target_val.gt(tile_max));

                //     p_entry = select(p_entry, p_exit, out_of_bounds && entry_is_empty);
                //     p_exit = select(p_exit, p_entry, out_of_bounds && !entry_is_empty);
                // }

                {
                    let exit_is_min = p_exit.y < p_entry.y;
                    clipped_top = select(p_entry, p_exit, exit_is_min);
                    clipped_bot = select(p_exit, p_entry, exit_is_min);

                    clipped_top -= tile_min;
                    clipped_bot -= tile_min;

                    let tile_size = vec2f(Tile::WIDTH as f32, Tile::HEIGHT as f32);
                    let zero_vec = vec2f_s(0.0);

                    // Clamping has a dual purpose here:
                    // 1) Points which are slightly outside the tile due to floating point error are
                    //    coerced inside.
                    // 2) More subtly, perfectly horizontal or vertical lines have their reciprocal
                    //    derivatives set to 0. This causes the intersection calculation to return
                    //    the original coordinate. While the coordinate fixed to the tile edge is
                    //    explicitly set (and guaranteed valid), clamping forces the coordinate
                    //    along that edge to be in bounds and watertight.
                    clipped_top = clamp(clipped_top, zero_vec, tile_size);
                    clipped_bot = clamp(clipped_bot, zero_vec, tile_size);
                }
            }

            let mut mask = [[0x80808080u32; PACK as usize]; Tile::WIDTH as usize];
            if pmt.tile_start_id() as u32 == (tid >> 2) {
                let winding = pmt.scanned_winding();
                for x in 0..Tile::WIDTH as usize {
                    mask[x][0] += winding * 0x1010101u32;
                    mask[x][1] += winding * 0x1010101u32;
                }
            }

            if !right_in_viewport {
                for x in 0..Tile::WIDTH as usize {
                    temp_mask[pmt_index][x][row_index][0] = mask[x][0];
                    temp_mask[pmt_index][x][row_index][1] = mask[x][1];
                }
                continue;
            }

            let left = (intersection_data & INTERSECTS_LEFT_MASK) != 0;
            if left {
                let y_edge = if clipped_top.x <= clipped_bot.x {
                    clipped_top.y
                } else {
                    clipped_bot.y
                };
                let v = if canonical_x_dir {
                    0xfefefeffu32 // = 0u32.wrapping_sub(0x1010101u32)
                } else {
                    0x1010101u32
                };

                let y_cross = y_edge.ceil() as u32;
                if row >= y_cross {
                    for x in 0..Tile::WIDTH as usize {
                        mask[x][0] += v;
                        mask[x][1] += v;
                    }
                }
            }

            // Discard perfectly axis aligned horizontal lines as the vertical mask produces the correct
            // value. This also ensures that the ceil call for end_y will always produce a value
            // distinct from start_y.
            if is_horizontal && clipped_top.y == clipped_top.y.floor() {
                for x in 0..Tile::WIDTH as usize {
                    temp_mask[pmt_index][x][row_index][0] = mask[x][0];
                    temp_mask[pmt_index][x][row_index][1] = mask[x][1];
                }
                continue;
            }

            let start_y = clipped_top.y.floor() as u32;
            let end_y = clipped_bot.y.ceil() as u32;
            let x_dir = clipped_top.x <= clipped_bot.x;
            if row >= start_y && row < end_y {
                let row_top = row as f32;
                let row_bot = (row + 1u32) as f32;
                let x_cross_top = clipped_top.x + (row_top - clipped_top.y) * dxdy;
                let p_top = select(
                    vec2f(x_cross_top, row_top),
                    clipped_top,
                    clipped_top.y >= row_top,
                );

                let x_cross_bot = clipped_top.x + (row_bot - clipped_top.y) * dxdy;
                let p_bot = select(
                    vec2f(x_cross_bot, row_bot),
                    clipped_bot,
                    clipped_bot.y <= row_bot,
                );

                if p_top.y <= p_bot.y && p_top.y < row_bot {
                    let y = row;
                    let x_min = p_top.x.min(p_bot.x);
                    let x_max = p_top.x.max(p_bot.x);
                    let x_start = (x_min.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;
                    let x_end = (x_max.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;

                    for x in x_start..Tile::WIDTH as usize {
                        let crossed_top = y > start_y || p_top.y == p_top.y.floor();

                        if x <= x_end {
                            let px = x as f32;
                            let py = y as f32;

                            let mut msaa_mask = get_msaa_mask(
                                vec2f(p_top.x - px, p_top.y - py),
                                vec2f(p_bot.x - px, p_bot.y - py),
                                &lut,
                            );

                            let is_start = x == x_start;
                            let is_end = x == x_end;
                            let cannonical_start = (x_dir && is_start) || (!x_dir && is_end);
                            let cannonical_end = (x_dir && is_end) || (!x_dir && is_start);
                            let line_top = cannonical_start && y == start_y;

                            let pixel_top_touch = p_top.y != p_top.y.floor();

                            let bumped = (line_top && p_top.x == 0.0 && pixel_top_touch)
                                || (!line_top && pixel_top_touch && x_dir);

                            if line_top && !bumped {
                                let mask_shift = (8.0 * (p_top.y - py)).round() as u32;
                                msaa_mask &= 0xffu32 << mask_shift;
                            }

                            if cannonical_end && y == end_y - 1 && p_bot.x != 0.0 {
                                let mask_shift = (8.0 * (p_bot.y - py)).round() as u32;
                                msaa_mask &= !(0xffu32 << mask_shift);
                            }

                            let mask_a = msaa_mask ^ (msaa_mask << 7u32);
                            let mask_b = mask_a ^ (mask_a << 14u32);
                            let mut mask_lo = mask_b & 0x1010101u32;
                            let mut mask_hi = mask_b >> 4 & 0x1010101u32;

                            if bumped {
                                let ones = 0x1010101u32;
                                mask_lo = mask_lo.wrapping_sub(ones);
                                mask_hi = mask_hi.wrapping_sub(ones);
                            }

                            if canonical_y_dir {
                                mask[x][0] += mask_lo;
                                mask[x][1] += mask_hi;
                            } else {
                                mask[x][0] -= mask_lo;
                                mask[x][1] -= mask_hi;
                            };
                        } else {
                            if crossed_top {
                                if canonical_y_dir {
                                    mask[x][0] += 0x1010101u32;
                                    mask[x][1] += 0x1010101u32;
                                } else {
                                    mask[x][0] += 0xfefefeffu32;
                                    mask[x][1] += 0xfefefeffu32;
                                }
                            }
                        }
                    }
                }
            }
            for x in 0..Tile::WIDTH as usize {
                temp_mask[pmt_index][x][row_index][0] = mask[x][0];
                temp_mask[pmt_index][x][row_index][1] = mask[x][1];
            }
        }
    }

    let mut part_indicator = vec![false; wg_count as usize];
    let mut part_acc = vec![
        [[[0x80808080u32; PACK as usize]; Tile::HEIGHT as usize];
            Tile::WIDTH as usize];
        wg_count as usize
    ];

    // wasteful? yes
    let mut final_mask = vec![
        [[[0x80808080u32; PACK as usize]; Tile::HEIGHT as usize];
            Tile::WIDTH as usize];
        total_threads
    ];

    // TODO if guards are added to prevent carry smearing across packed values, a separate trick can
    // be used where the value is only dibiased once at the end. However, since we assume that
    // winding values can never exceed [-128, 127], simply debiasing with a wrapping sub is fastest.
    for wgid in 0..wg_count {
        for tid in 0..PART_SIZE {
            let gidu = (tid + wgid * PART_SIZE) as usize;
            if gidu >= pmt_count || gidu + aligned_start < start_index {
                continue;
            }

            if tid != 0 {
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        let prev_lo = temp_mask[gidu - 1][x][y][0];
                        let delta_lo = prev_lo.wrapping_sub(0x80808080u32);
                        temp_mask[gidu][x][y][0] = temp_mask[gidu][x][y][0].wrapping_add(delta_lo);

                        let prev_hi = temp_mask[gidu - 1][x][y][1];
                        let delta_hi = prev_hi.wrapping_sub(0x80808080u32);
                        temp_mask[gidu][x][y][1] = temp_mask[gidu][x][y][1].wrapping_add(delta_hi);
                    }
                }
            }

            let pmt = pmt_buf[gidu + aligned_start];

            // Tiles which are not end tiles have no need for further processing
            if !(pmt.is_end_tile() || tid == PART_SIZE - 1) {
                continue;
            }

            let tile_start_id = pmt.tile_start_id() as u32;
            // If the tile start is inside our partition, we must subtract the reduction at the
            // tile before it.
            // A tile start at 0 has no predecessors, so there is no need to subtract predecessors.
            if tile_start_id != INVALID_ID && tile_start_id != 0 {
                let tile_id = (tile_start_id + wgid * PART_SIZE - 1) as usize;
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        let prev_lo = temp_mask[tile_id][x][y][0];
                        let delta_lo = prev_lo.wrapping_sub(0x80808080u32);
                        final_mask[gidu][x][y][0] = temp_mask[gidu][x][y][0].wrapping_sub(delta_lo);

                        let prev_hi = temp_mask[tile_id][x][y][1];
                        let delta_hi = prev_hi.wrapping_sub(0x80808080u32);
                        final_mask[gidu][x][y][1] = temp_mask[gidu][x][y][1].wrapping_sub(delta_hi);
                    }
                }
            } else {
                final_mask[gidu] = temp_mask[gidu];
            }

            // The final thread in a workgroup writes out state necessary for successor workgroups
            if tid == PART_SIZE - 1 {
                // If there is a tile start in this workgroup, the last thread must see a valid id.
                part_indicator[wgid as usize] = pmt.tile_start_id() as u32 != INVALID_ID;
                part_acc[wgid as usize] = final_mask[gidu];
            }

            // Now, the "stitching process," if the start tile is not within the workgroup, we need
            // to traverse backwards until we reach it.
            if tile_start_id == INVALID_ID && pmt.is_end_tile() {
                let mut lookback_id = wgid - 1;
                loop {
                    for x in 0..Tile::WIDTH as usize {
                        for y in 0..Tile::HEIGHT as usize {
                            let prev_lo = part_acc[lookback_id as usize][x][y][0];
                            let delta_lo = prev_lo.wrapping_sub(0x80808080u32);
                            final_mask[gidu][x][y][0] =
                                final_mask[gidu][x][y][0].wrapping_add(delta_lo);

                            let prev_hi = part_acc[lookback_id as usize][x][y][1];
                            let delta_hi = prev_hi.wrapping_sub(0x80808080u32);
                            final_mask[gidu][x][y][1] =
                                final_mask[gidu][x][y][1].wrapping_add(delta_hi);
                        }
                    }

                    if part_indicator[lookback_id as usize] || lookback_id == 0 {
                        break;
                    } else {
                        lookback_id -= 1;
                    }
                }
            }
        }
    }

    for gid in 0..pmt_count {
        if gid + aligned_start < start_index {
            continue;
        }
        let pmt = pmt_buf[gid + aligned_start];
        if pmt.is_end_tile() {
            let mut temp_fine = [[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize];
            let bias_mask = 0x80808080u32;
            let ones_mask = 0x01010101u32;

            if pmt.tile_x() > 0 {
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        let mut active_count = 0u32;
                        for &pack in &[final_mask[gid][x][y][0], final_mask[gid][x][y][1]] {
                            let diff = pack ^ bias_mask;
                            let subbed = diff.wrapping_sub(ones_mask);
                            let zero_byte_markers = subbed & (!diff) & bias_mask;
                            let active_byte_markers = zero_byte_markers ^ bias_mask;
                            let count_pack =
                                ((active_byte_markers >> 7).wrapping_mul(ones_mask)) >> 24;
                            active_count += count_pack;
                        }

                        if active_count > 0 {
                            temp_fine[x][y] += (active_count as f32) * 31.875;
                        }
                    }
                }
            }

            let mut u8_vals = [0u8; 16];
            let mut i = 0;
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    u8_vals[i] = temp_fine[x][y].round() as u8;
                    i += 1;
                }
            }

            let dst_idx = pmt.alpha_index as usize;
            if dst_idx + 16 <= alpha_buf.len() {
                alpha_buf[dst_idx..dst_idx + 16].copy_from_slice(&u8_vals);
            }
        }
    }
}

//CPU Style

/// Clips a line segment to a tile.
///
/// Returns the start and end points of the clipped line segment relative to the tile origin.
pub fn clip_to_tile(
    line: &Line,
    bounds: &[f32; 4],
    derivatives: &[f32; 4],
    intersection_data: u32,
    cannonical_x_dir: bool,
    cannonical_y_dir: bool,
) -> [[f32; 2]; 2] {
    const INTERSECTS_TOP_MASK: u32 = 1;
    const INTERSECTS_BOTTOM_MASK: u32 = 2;
    const INTERSECTS_LEFT_MASK: u32 = 4;
    const INTERSECTS_RIGHT_MASK: u32 = 8;
    const PERFECT_MASK: u32 = 16;

    let mut p_entry = [line.p0.x, line.p0.y];
    let mut p_exit = [line.p1.x, line.p1.y];

    let tile_min_x = bounds[0];
    let tile_min_y = bounds[1];
    let tile_max_x = bounds[2];
    let tile_max_y = bounds[3];

    let dx = derivatives[0];
    let dy = derivatives[1];

    let (mask_v_in, bound_v_in, mask_v_out, bound_v_out) = if cannonical_x_dir {
        (
            INTERSECTS_LEFT_MASK,
            tile_min_x,
            INTERSECTS_RIGHT_MASK,
            tile_max_x,
        )
    } else {
        (
            INTERSECTS_RIGHT_MASK,
            tile_max_x,
            INTERSECTS_LEFT_MASK,
            tile_min_x,
        )
    };

    let (mask_h_in, bound_h_in, mask_h_out, bound_h_out) = if cannonical_y_dir {
        (
            INTERSECTS_TOP_MASK,
            tile_min_y,
            INTERSECTS_BOTTOM_MASK,
            tile_max_y,
        )
    } else {
        (
            INTERSECTS_BOTTOM_MASK,
            tile_max_y,
            INTERSECTS_TOP_MASK,
            tile_min_y,
        )
    };

    // let slope_v = dy * derivatives[2];
    // let slope_h = dx * derivatives[3];
    // let entry_hits = intersection_data & (mask_v_in | mask_h_in);
    // if entry_hits != 0 {
    //     let use_h = (intersection_data & mask_h_in) != 0;

    //     let base = if use_h { line.p0.x } else { line.p0.y };
    //     let ortho = if use_h { line.p0.y } else { line.p0.x };
    //     let bound = if use_h { bound_h_in } else { bound_v_in };
    //     let slope = if use_h { slope_h } else { slope_v };

    //     let calculated = base + (bound - ortho) * slope;
    //     let axis = if use_h { 0 } else { 1 };

    //     p_entry[axis] = calculated;
    //     p_entry[1 - axis] = bound;
    // }

    // let exit_hits = intersection_data & (mask_v_out | mask_h_out);
    // if exit_hits != 0 {
    //     let use_h = (intersection_data & mask_h_out) != 0;

    //     let base = if use_h { line.p0.x } else { line.p0.y };
    //     let ortho = if use_h { line.p0.y } else { line.p0.x };
    //     let bound = if use_h { bound_h_out } else { bound_v_out };
    //     let slope = if use_h { slope_h } else { slope_v };

    //     let calculated = base + (bound - ortho) * slope;
    //     let axis = if use_h { 0 } else { 1 };

    //     p_exit[axis] = calculated;
    //     p_exit[1 - axis] = bound;
    // }

    let idx = derivatives[2];
    let idy = derivatives[3];

    let entry_hits = intersection_data & (mask_v_in | mask_h_in);
    if entry_hits != 0 {
        let use_h = (intersection_data & mask_h_in) != 0;

        let bound = if use_h { bound_h_in } else { bound_v_in };
        let start = if use_h { line.p0.y } else { line.p0.x };
        let inv_d = if use_h { idy } else { idx };

        let t = (bound - start) * inv_d;

        p_entry[0] = line.p0.x + t * dx;
        p_entry[1] = line.p0.y + t * dy;
        p_entry[if use_h { 1 } else { 0 }] = bound;
    }

    let exit_hits = intersection_data & (mask_v_out | mask_h_out);
    if exit_hits != 0 {
        let use_h = (intersection_data & mask_h_out) != 0;

        let bound = if use_h { bound_h_out } else { bound_v_out };
        let start = if use_h { line.p0.y } else { line.p0.x };
        let inv_d = if use_h { idy } else { idx };

        let t = (bound - start) * inv_d;

        p_exit[0] = line.p0.x + t * dx;
        p_exit[1] = line.p0.y + t * dy;
        p_exit[if use_h { 1 } else { 0 }] = bound;
    }

    // TODO: The perfect bit can be removed; this means less cpu side work, but this check becomes
    // less specific, so EVERY single intersection hits the bounds checks
    //
    // If we have a perfect corner intersection (PERFECT_MASK is set) AND the intersection has been
    // tie-broken to a single edge (only 1 bit set in 0-3), we duplicate the intersection point.
    // Otherwise, the raw endpoint would be returned. In these single intersection cases, this
    // creates valid non-deleterious logic.
    let is_perfect = (intersection_data & PERFECT_MASK) != 0;
    let single_hit = (exit_hits ^ entry_hits) != 0;
    if is_perfect && single_hit {
        let (target, source) = if entry_hits == 0 {
            (&mut p_entry, p_exit)
        } else {
            (&mut p_exit, p_entry)
        };
        if target[0] < tile_min_x
            || target[0] > tile_max_x
            || target[1] < tile_min_y
            || target[1] > tile_max_y
        {
            *target = source;
        }
    }

    let mut result = if p_exit[1] >= p_entry[1] {
        [p_entry, p_exit]
    } else {
        [p_exit, p_entry]
    };

    // if result[0][1] > result[1][1] {
    //     panic!("Not ordered correctly! Intersected: {} {} Original Line ({},{}) ({},{})",
    //     result[0][1], result[1][1], p0[0], p0[1], p1[0], p1[1]);
    // }

    result[0][0] -= tile_min_x;
    result[0][1] -= tile_min_y;
    result[1][0] -= tile_min_x;
    result[1][1] -= tile_min_y;

    // Clamping has a dual purpose here:
    // 1) Points which are slightly outside the tile due to floating point error are coerced inside.
    // 2) More subtly, perfectly horizontal or vertical lines have their reciprocal derivatives
    //    set to 0. This causes the intersection calculation to return the original coordinate.
    //    While the coordinate fixed to the tile edge is explicitly set (and guaranteed valid),
    //    clamping forces the coordinate along that edge to be in bounds and watertight.
    let width = Tile::WIDTH as f32;
    let height = Tile::HEIGHT as f32;
    result[0][0] = result[0][0].clamp(0.0, width);
    result[0][1] = result[0][1].clamp(0.0, height);
    result[1][0] = result[1][0].clamp(0.0, width);
    result[1][1] = result[1][1].clamp(0.0, height);

    result
}

fn msaa_merge_even_odd(pmt_buf: &mut Vec<PreMergeTile>, alpha_buf: &mut Vec<u8>, lut: &Vec<u8>) {
    if pmt_buf.is_empty() {
        return;
    }

    let pmt_count = pmt_buf.len();
    let mut t_vert_mask = vec![[0u8; Tile::HEIGHT as usize]; pmt_count];
    let mut temp_mask = vec![[[0u8; Tile::HEIGHT as usize]; Tile::WIDTH as usize]; pmt_count];

    for gid in 0..pmt_count {
        let pmt = pmt_buf[gid];

        let line = pmt.line;
        let p0_x = line.p0.x;
        let p0_y = line.p0.y;
        let p1_x = line.p1.x;
        let p1_y = line.p1.y;

        let tile_min_x_u32 = (pmt.tile_x() as u32) * (Tile::WIDTH as u32);
        let tile_min_y_u32 = (pmt.tile_y() as u32) * (Tile::HEIGHT as u32);

        let tile_min_x_px = tile_min_x_u32 as f32;
        let tile_min_y_px = tile_min_y_u32 as f32;
        let tile_max_x_px = (tile_min_x_u32 + (Tile::WIDTH as u32)) as f32;
        let tile_max_y_px = (tile_min_y_u32 + (Tile::HEIGHT as u32)) as f32;

        let bounds = [tile_min_x_px, tile_min_y_px, tile_max_x_px, tile_max_y_px];
        let intersection_data = pmt.intersection_data();

        let cannonical_x_dir = p1_x >= p0_x;
        let cannonical_y_dir = p1_y >= p0_y;

        let dx = p1_x - p0_x;
        let dy = p1_y - p0_y;

        let is_vertical = dx.abs() <= f32::EPSILON;
        let is_horizontal = dy.abs() <= f32::EPSILON;

        let idx = if !is_vertical { 1.0 / dx } else { 0.0 };
        let idy = if !is_horizontal { 1.0 / dy } else { 0.0 };

        let derivatives = [dx, dy, idx, idy];
        let clipped_result = clip_to_tile(
            &line,
            &bounds,
            &derivatives,
            intersection_data,
            cannonical_x_dir,
            cannonical_y_dir,
        );

        let [local_p0, local_p1] = clipped_result;

        let left = (intersection_data & INTERSECTS_LEFT_MASK) != 0;

        // A edge is considered as intersecting the left edge if:
        //  1) It passes through the left tile, i.e. a point that only touches the left edge does
        //  not count.
        //  2) It does not also intersect the top edge at the corner. i.e. perfect intersection
        //  3) It is not a vertical line. This is a property of the tiling algorithm, so no check is
        //  needed here.
        if left {
            let y_edge = if local_p0[0] <= local_p1[0] {
                local_p0[1]
            } else {
                local_p1[1]
            };
            let y_cross = y_edge.ceil() as u16;
            for y in y_cross as usize..Tile::HEIGHT as usize {
                t_vert_mask[gid][y] = 0xffu8;
            }
        }

        // Start tiles get the coarse winding number
        if pmt.fill_rule() {
            let fill = (pmt.scanned_winding() % 2) != 0;
            let coarse_mask = if fill { 0xffu8 } else { 0u8 };
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_mask[gid][x][y] = coarse_mask;
                }
            }
        }

        for x in 0..Tile::WIDTH as usize {
            for y in 0..Tile::HEIGHT as usize {
                temp_mask[gid][x][y] ^= t_vert_mask[gid][y];
            }
        }

        // Discard perfectly axis aligned horizontal lines as the vertical mask produces the correct
        // value. This also ensures that the ceil call for end_y will always produce a value
        // distinct from start_y.
        if is_horizontal && local_p0[1] == local_p0[1].floor() {
            continue;
        }

        let start_y = local_p0[1].floor() as usize;
        let end_y = local_p1[1].ceil() as usize;
        let mut top_row = [[f32::NAN; 2]; (Tile::HEIGHT + 1) as usize];
        {
            // if end_y > Tile::HEIGHT as usize {
            //     println!("BAD TILE at ({}, {})", pmt.tile_x(), pmt.tile_y());
            //     println!("Line: ({}, {}) to ({}, {})", p0_x, p0_y, p1_x, p1_y);
            //     println!("Intersection Data bits: {:05b}", pmt.intersection_data());
            //     println!("Calculated local_p1: {:?}", local_p1);
            //     println!("startY EndY {}, {}", start_y, end_y);
            // }
            debug_assert!(end_y <= (Tile::HEIGHT as usize) && start_y <= end_y);
            let dydx = if is_horizontal { 0.0 } else { dx / dy };
            top_row[start_y] = local_p0;
            for y_idx in (start_y + 1)..end_y {
                let grid_y = y_idx as f32;
                let grid_x = local_p0[0] + (grid_y - local_p0[1]) * dydx;
                top_row[y_idx] = [grid_x, grid_y];
            }
            top_row[end_y] = local_p1;
        }

        let x_dir = local_p0[0] <= local_p1[0];
        for y in start_y..end_y {
            let p_top = top_row[y];
            let p_bottom = top_row[y + 1];

            if p_top[0].is_nan() || p_bottom[0].is_nan() {
                continue;
            }

            let x_min = p_top[0].min(p_bottom[0]);
            let x_max = p_top[0].max(p_bottom[0]);
            let x_start = (x_min.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;
            let x_end =
                (x_max.next_down().floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;
            let crossed_top = y > start_y || p_top[1] == p_top[1].floor();
            for x in x_start..Tile::WIDTH as usize {
                if x <= x_end {
                    let px = x as f32;
                    let py = y as f32;

                    let mut mask = algorithm_lut_mask_halfplane(
                        [(p_top[0] - px), (p_top[1] - py)],
                        [(p_bottom[0] - px), (p_bottom[1] - py)],
                        &lut,
                    );

                    let is_start = x == x_start;
                    let is_end = x == x_end;
                    let cannonical_start = (x_dir && is_start) || (!x_dir && is_end);
                    let cannonical_end = (x_dir && is_end) || (!x_dir && is_start);
                    let line_top = cannonical_start && y == start_y;
                    let bumped = (line_top && p_top[0] == 0.0 && left)
                        || (!line_top && p_top[1] != p_top[1].floor() && x_dir);
                    if line_top && !bumped {
                        let mask_shift = (8.0 * (p_top[1] - py)).round() as u32;
                        mask &= (0xffu32 << mask_shift) as u8;
                    }

                    if cannonical_end && y == end_y - 1 && p_bottom[0] != 0.0 {
                        let mask_shift = (8.0 * (p_bottom[1] - py)).round() as u32;
                        mask &= !((0xffu32 << mask_shift) as u8);
                    }

                    if bumped {
                        mask ^= 0xffu8;
                    }

                    temp_mask[gid][x][y] ^= mask;
                } else {
                    if crossed_top {
                        temp_mask[gid][x][y] ^= 0xffu8;
                    }
                }
            }
        }
    }

    for gid in 1..pmt_count {
        let pmt = pmt_buf[gid];
        if !pmt.fill_rule() {
            // If i am NOT a start tile . . . layer
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_mask[gid][x][y] ^= temp_mask[gid - 1][x][y];
                }
            }
        }
    }

    for gid in 0..(pmt_count - 1) {
        let pmt = pmt_buf[gid];
        if pmt.is_end_tile() {
            let mut temp_fine = [[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize];
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_fine[x][y] =
                        (temp_mask[gid][x][y].count_ones() as f32 * 31.875).min(255.0);
                }
            }

            let mut u8_vals = [0u8; 16];
            let mut i = 0;
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    u8_vals[i] = temp_fine[x][y].round() as u8;
                    i += 1;
                }
            }

            // this will be indexed into
            alpha_buf.extend_from_slice(&u8_vals);
        }
    }
}

const PACK: u32 = 2;
fn msaa_merge_non_zero(pmt_buf: &mut Vec<PreMergeTile>, alpha_buf: &mut Vec<u8>, lut: &Vec<u8>) {
    if pmt_buf.is_empty() {
        return;
    }

    let pmt_count = pmt_buf.len();
    let mut t_vert_mask = vec![[0u32; Tile::HEIGHT as usize]; pmt_count];
    let mut temp_mask = vec![
        [[[0x80808080u32; PACK as usize]; Tile::HEIGHT as usize];
            Tile::WIDTH as usize];
        pmt_count
    ];

    for gid in 0..pmt_count {
        let pmt = pmt_buf[gid];

        let line = pmt.line;
        let p0_x = line.p0.x;
        let p0_y = line.p0.y;
        let p1_x = line.p1.x;
        let p1_y = line.p1.y;

        let cannonical_x_dir = p1_x >= p0_x;
        let cannonical_y_dir = p1_y >= p0_y;

        // Start tiles get the coarse winding number
        if pmt.fill_rule() {
            let winding = pmt.scanned_winding();
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    temp_mask[gid][x][y][0] += winding * 0x1010101u32;
                    temp_mask[gid][x][y][1] += winding * 0x1010101u32;
                }
            }
        }

        // If the rightmost point of a line is left of the viewport, no further processing is
        // required. The coarse mask is the only dependency which need be passed on.
        let right = if cannonical_x_dir { p1_x } else { p0_x };

        if right < 0.0 {
            continue;
        }

        let tile_min_x_u32 = (pmt.tile_x() as u32) * (Tile::WIDTH as u32);
        let tile_min_y_u32 = (pmt.tile_y() as u32) * (Tile::HEIGHT as u32);

        let tile_min_x_px = tile_min_x_u32 as f32;
        let tile_min_y_px = tile_min_y_u32 as f32;
        let tile_max_x_px = (tile_min_x_u32 + (Tile::WIDTH as u32)) as f32;
        let tile_max_y_px = (tile_min_y_u32 + (Tile::HEIGHT as u32)) as f32;

        let bounds = [tile_min_x_px, tile_min_y_px, tile_max_x_px, tile_max_y_px];
        let intersection_data = pmt.intersection_data();

        let dx = p1_x - p0_x;
        let dy = p1_y - p0_y;

        let is_vertical = dx.abs() <= f32::EPSILON;
        let is_horizontal = dy.abs() <= f32::EPSILON;

        let idx = if !is_vertical { 1.0 / dx } else { 0.0 };
        let idy = if !is_horizontal { 1.0 / dy } else { 0.0 };

        let derivatives = [dx, dy, idx, idy];
        let clipped_result = clip_to_tile(
            &line,
            &bounds,
            &derivatives,
            intersection_data,
            cannonical_x_dir,
            cannonical_y_dir,
        );

        let [local_p0, local_p1] = clipped_result;
        //println!("({}, {}) ({}, {})", local_p0[0], local_p0[1], local_p1[0], local_p1[1]);

        let left = (intersection_data & INTERSECTS_LEFT_MASK) != 0;
        if left {
            let y_edge = if local_p0[0] <= local_p1[0] {
                local_p0[1]
            } else {
                local_p1[1]
            };
            let v = if cannonical_x_dir {
                0xfefefeffu32 // = 0u32.wrapping_sub(0x1010101u32)
            } else {
                0x1010101u32
            };

            let y_cross = y_edge.ceil() as u16;
            for y in y_cross as usize..Tile::HEIGHT as usize {
                t_vert_mask[gid][y] = v;
            }
        }

        for x in 0..Tile::WIDTH as usize {
            for y in 0..Tile::HEIGHT as usize {
                temp_mask[gid][x][y][0] += t_vert_mask[gid][y];
                temp_mask[gid][x][y][1] += t_vert_mask[gid][y];
            }
        }

        // Discard perfectly axis aligned horizontal lines as the vertical mask produces the correct
        // value. This also ensures that the ceil call for end_y will always produce a value
        // distinct from start_y.
        if is_horizontal && local_p0[1] == local_p0[1].floor() {
            continue;
        }

        let start_y = local_p0[1].floor() as usize;
        let end_y = local_p1[1].ceil() as usize;
        let mut top_row = [[f32::NAN; 2]; (Tile::HEIGHT + 1) as usize];
        {
            // if end_y > Tile::HEIGHT as usize {
            //     println!("BAD TILE at ({}, {})", pmt.tile_x(), pmt.tile_y());
            //     println!("Line: ({}, {}) to ({}, {})", p0_x, p0_y, p1_x, p1_y);
            //     println!("Intersection Data bits: {:05b}", pmt.intersection_data());
            //     println!("Calculated local_p1: {:?}", local_p1);
            //     println!("startY EndY {}, {}", start_y, end_y);
            // }
            debug_assert!(end_y <= (Tile::HEIGHT as usize) && start_y <= end_y);
            let dydx = if is_horizontal { 0.0 } else { dx / dy };
            top_row[start_y] = local_p0;
            for y_idx in (start_y + 1)..end_y {
                let grid_y = y_idx as f32;
                let grid_x = local_p0[0] + (grid_y - local_p0[1]) * dydx;
                top_row[y_idx] = [grid_x, grid_y];
            }
            top_row[end_y] = local_p1;
        }

        let x_dir = local_p0[0] <= local_p1[0];
        for y in start_y..end_y {
            let p_top = top_row[y];
            let p_bottom = top_row[y + 1];

            if p_top[0].is_nan() || p_bottom[0].is_nan() {
                continue;
            }

            let x_min = p_top[0].min(p_bottom[0]);
            let x_max = p_top[0].max(p_bottom[0]);
            let x_start = (x_min.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;
            let x_end = (x_max.floor() as i32).clamp(0, Tile::WIDTH as i32 - 1) as usize;

            //println!("({}, {}) ({}, {})", p_top[0], p_top[1], p_bottom[1], p_bottom[1]);

            for x in x_start..Tile::WIDTH as usize {
                let crossed_top = y > start_y || p_top[1] == p_top[1].floor();

                // Does this matter?
                //let crossed_top = y > start_y || (y == 0 && pmt.intersects_top()) || p_top[1] == p_top[1].floor();

                if x <= x_end {
                    let px = x as f32;
                    let py = y as f32;

                    let mut mask = algorithm_lut_mask_halfplane(
                        [(p_top[0] - px), (p_top[1] - py)],
                        [(p_bottom[0] - px), (p_bottom[1] - py)],
                        &lut,
                    ) as u32;

                    let is_start = x == x_start;
                    let is_end = x == x_end;
                    let cannonical_start = (x_dir && is_start) || (!x_dir && is_end);
                    let cannonical_end = (x_dir && is_end) || (!x_dir && is_start);
                    let line_top = cannonical_start && y == start_y;

                    let pixel_top_touch = p_top[1] != p_top[1].floor();
                    let bumped = (line_top && p_top[0] == 0.0 && pixel_top_touch)
                        || (!line_top && pixel_top_touch && x_dir);
                    if line_top && !bumped {
                        let mask_shift = (8.0 * (p_top[1] - py)).round() as u32;
                        mask &= 0xffu32 << mask_shift;
                    }

                    if cannonical_end && y == end_y - 1 && p_bottom[0] != 0.0 {
                        let mask_shift = (8.0 * (p_bottom[1] - py)).round() as u32;
                        mask &= !(0xffu32 << mask_shift);
                    }

                    let mask_a = mask ^ (mask << 7u32);
                    let mask_b = mask_a ^ (mask_a << 14u32);
                    let mut mask_lo = mask_b & 0x1010101u32;
                    let mut mask_hi = mask_b >> 4 & 0x1010101u32;

                    if bumped {
                        let ones = 0x1010101u32;
                        mask_lo = mask_lo.wrapping_sub(ones);
                        mask_hi = mask_hi.wrapping_sub(ones);
                    }

                    if cannonical_y_dir {
                        temp_mask[gid][x][y][0] += mask_lo;
                        temp_mask[gid][x][y][1] += mask_hi;
                    } else {
                        temp_mask[gid][x][y][0] -= mask_lo;
                        temp_mask[gid][x][y][1] -= mask_hi;
                    };
                } else {
                    if crossed_top {
                        if cannonical_y_dir {
                            temp_mask[gid][x][y][0] += 0x1010101u32;
                            temp_mask[gid][x][y][1] += 0x1010101u32;
                        } else {
                            temp_mask[gid][x][y][0] += 0xfefefeffu32; // = 0u32.wrapping_sub(0x1010101u32)
                            temp_mask[gid][x][y][1] += 0xfefefeffu32;
                        }
                    }
                }
            }
        }
    }

    for gid in 1..pmt_count {
        let pmt = pmt_buf[gid];
        if !pmt.fill_rule() {
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    let prev_lo = temp_mask[gid - 1][x][y][0];
                    let delta_lo = prev_lo.wrapping_sub(0x80808080u32);
                    temp_mask[gid][x][y][0] = temp_mask[gid][x][y][0].wrapping_add(delta_lo);

                    let prev_hi = temp_mask[gid - 1][x][y][1];
                    let delta_hi = prev_hi.wrapping_sub(0x80808080u32);
                    temp_mask[gid][x][y][1] = temp_mask[gid][x][y][1].wrapping_add(delta_hi);
                }
            }
        }
    }

    for gid in 0..(pmt_count - 1) {
        let pmt = pmt_buf[gid];
        if pmt.is_end_tile() {
            let mut temp_fine = [[0f32; Tile::HEIGHT as usize]; Tile::WIDTH as usize];
            let bias_mask = 0x80808080u32;
            let ones_mask = 0x01010101u32;

            if pmt.tile_x() > 0 {
                for x in 0..Tile::WIDTH as usize {
                    for y in 0..Tile::HEIGHT as usize {
                        let mut active_count = 0u32;
                        for &pack in &[temp_mask[gid][x][y][0], temp_mask[gid][x][y][1]] {
                            let diff = pack ^ bias_mask;
                            let subbed = diff.wrapping_sub(ones_mask);
                            let zero_byte_markers = subbed & (!diff) & bias_mask;
                            let active_byte_markers = zero_byte_markers ^ bias_mask;
                            let count_pack =
                                ((active_byte_markers >> 7).wrapping_mul(ones_mask)) >> 24;
                            active_count += count_pack;
                        }
                        if active_count > 0 {
                            temp_fine[x][y] += (active_count as f32) * 31.875;
                        }
                    }
                }
            }

            let mut u8_vals = [0u8; 16];
            let mut i = 0;
            for x in 0..Tile::WIDTH as usize {
                for y in 0..Tile::HEIGHT as usize {
                    u8_vals[i] = temp_fine[x][y].round() as u8;
                    i += 1;
                }
            }

            // this will be indexed into
            alpha_buf.extend_from_slice(&u8_vals);
        }
    }
}
