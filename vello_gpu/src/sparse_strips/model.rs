// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU model of the GPU sparse strip algorithm.
//!
//! Every function here corresponds to a piece of the shaders: [`leaf`] and [`combine`] are the
//! allocation monoid, [`emit`] is the alloc pass for one entry, [`line_payload`] is the
//! per-entry rasterization of the merge pass and [`resolve`] is the resolve pass. The model
//! processes entries sequentially; the shaders evaluate the same monoid as a parallel scan,
//! which is valid because [`combine`] is associative (tested below).

use super::{
    ACC_WORDS, FIXED_ONE, GpuPathInfo, GpuStripDesc, STRIP_FILL_GAP, STRIP_REAL, STRIP_SENTINEL,
    STRIP_SOLID, STRIP_TERMINATOR, StripInput, TILE_PIXELS,
};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::tile::Tile;

const AGG_NONEMPTY: u32 = 1;
const AGG_MULTI_ROW: u32 = 2;

/// Aggregate of the allocation monoid over a contiguous range of entries.
///
/// A row is identified by `(path, tile y)`. Entries are sorted by path, then y, then x, so the
/// rows of a range are contiguous and only the first and last row can extend beyond the range.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Agg {
    /// `(path, y)` of the first row in the range.
    pub first_row: [u32; 2],
    /// `(path, y)` of the last row in the range.
    pub last_row: [u32; 2],
    /// Winding sum of the entries in the first row (includes the row's coarse winding if the
    /// range contains the row's first entry).
    pub w_first: i32,
    /// Winding sum of the entries in the last row (equal to `w_first` for a single-row range).
    pub w_last: i32,
    /// Strips emitted by the entries of the range, plus the terminators of the rows that are
    /// known to end inside the range, except the first row's, whose total is not yet known.
    pub strips: u32,
    /// Alpha blocks allocated by the entries of the range.
    pub blocks: u32,
    /// `AGG_NONEMPTY` and `AGG_MULTI_ROW`.
    pub flags: u32,
}

impl Agg {
    /// Whether the aggregate covers at least one entry.
    pub fn nonempty(&self) -> bool {
        self.flags & AGG_NONEMPTY != 0
    }

    /// Whether the aggregate covers more than one row.
    pub fn multi_row(&self) -> bool {
        self.flags & AGG_MULTI_ROW != 0
    }
}

/// Per-entry facts that do not depend on the scan prefix.
#[derive(Debug, Copy, Clone, Default)]
pub struct Local {
    /// Path index.
    pub path: u32,
    /// Tile x.
    pub x: u32,
    /// Tile y.
    pub y: u32,
    /// First entry of its path.
    pub path_head: bool,
    /// Last entry of its path.
    pub path_last: bool,
    /// First entry of its row.
    pub row_head: bool,
    /// Last entry of its row.
    pub row_last: bool,
    /// First entry of its run (tile location).
    pub run_head: bool,
    /// First entry of its strip (chain of adjacent runs).
    pub strip_head: bool,
    /// Winding contribution of the entry's line (`sign * crosses_top`).
    pub d: i32,
    /// Coarse winding seeded at the row start (culled paths only, row heads only).
    pub coarse: i32,
    /// Index into the path's filled-row list of the first culled background row this entry
    /// emits before its own row.
    pub bg_lo: u32,
    /// Number of culled background rows this entry emits before its own row.
    pub n_bg: u32,
    /// Whether this entry emits a captive strip before its own strip.
    pub captive: bool,
    /// Index into the filled-row list of the first culled background row after the path's
    /// last row (path tail only).
    pub trail_lo: u32,
    /// Number of culled background rows after the path's last row (path tail only).
    pub n_trail: u32,
}

/// Decode an entry.
fn entry_xy(entry: [u32; 2]) -> (u32, u32) {
    (entry[1] & 0xffff, entry[1] >> 16)
}

/// The line's winding contribution: zero for horizontal lines (which `strip::render` skips
/// entirely), otherwise the sign of the line's direction if it crosses the tile's top edge.
fn winding_delta(line: [f32; 4], entry: [u32; 2]) -> i32 {
    let (y0, y1) = (line[1], line[3]);
    if y0 == y1 {
        return 0;
    }
    let crosses_top = entry[0] & 0x10 != 0;
    if !crosses_top {
        return 0;
    }
    if y0 > y1 { 1 } else { -1 }
}

/// Compute the leaf aggregate and local facts of entry `i`.
pub fn leaf(input: &StripInput, i: u32) -> (Agg, Local) {
    let path_ix = input.path_of_entry(i);
    let path = &input.paths[path_ix as usize];
    let entry = input.entries[i as usize];
    let (x, y) = entry_xy(entry);
    let line = input.lines[(path.line_base + (entry[0] >> 5)) as usize];

    let path_head = i == path.entry_start;
    let path_last = i + 1 == path.entry_end;
    let (prev_x, prev_y) = if path_head {
        (u32::MAX, u32::MAX)
    } else {
        entry_xy(input.entries[i as usize - 1])
    };
    let row_head = path_head || prev_y != y;
    let run_head = row_head || prev_x != x;
    let strip_head = run_head && (row_head || prev_x + 1 != x);
    let row_last = path_last || entry_xy(input.entries[i as usize + 1]).1 != y;

    let mut local = Local {
        path: path_ix,
        x,
        y,
        path_head,
        path_last,
        row_head,
        row_last,
        run_head,
        strip_head,
        d: winding_delta(line, entry),
        ..Default::default()
    };

    if path.culled() {
        if row_head {
            let row = input.row(path, y);
            local.coarse = row.coarse;
            // Culled background rows between the previous tile row and this one.
            let lo = if path_head {
                0
            } else {
                input.row(path, prev_y + 1).bg_prefix
            };
            local.bg_lo = lo;
            local.n_bg = row.bg_prefix - lo;
            local.captive = path.should_fill(row.coarse) && x != 0;
        }
        if path_last {
            let lo = if y + 1 < input.rows() {
                input.row(path, y + 1).bg_prefix
            } else {
                path.n_fill_rows
            };
            local.trail_lo = lo;
            local.n_trail = path.n_fill_rows - lo;
        }
    }

    let mut strips = 2 * local.n_bg + u32::from(local.captive) + u32::from(strip_head);
    let mut blocks = local.n_bg + u32::from(local.captive) + u32::from(run_head);
    if path_last {
        strips += 2 * local.n_trail + 1;
        blocks += local.n_trail;
    }
    let w = local.d + local.coarse;
    let agg = Agg {
        first_row: [path_ix, y],
        last_row: [path_ix, y],
        w_first: w,
        w_last: w,
        strips,
        blocks,
        flags: AGG_NONEMPTY,
    };
    (agg, local)
}

/// The monoid operation: the aggregate of `l` followed by `r`.
pub fn combine(l: &Agg, r: &Agg) -> Agg {
    if !l.nonempty() {
        return *r;
    }
    if !r.nonempty() {
        return *l;
    }
    let same_row = l.last_row == r.first_row;
    let (w_first, w_last, extra, multi) = if same_row {
        let merged = l.w_last + r.w_first;
        let w_last = if r.multi_row() { r.w_last } else { merged };
        if l.multi_row() {
            // The merged row is an interior row of the result if it ends inside `r`.
            let extra = u32::from(r.multi_row() && merged != 0);
            (l.w_first, w_last, extra, true)
        } else {
            // The merged row is the result's first row; its terminator stays pending.
            (merged, w_last, 0, r.multi_row())
        }
    } else {
        // `l`'s last row ends at the boundary and `r`'s first row starts there. Each is
        // complete inside the result unless it is also the result's first row.
        let extra =
            u32::from(l.multi_row() && l.w_last != 0) + u32::from(r.multi_row() && r.w_first != 0);
        (l.w_first, r.w_last, extra, true)
    };
    Agg {
        first_row: l.first_row,
        last_row: r.last_row,
        w_first,
        w_last,
        strips: l.strips + r.strips + extra,
        blocks: l.blocks + r.blocks,
        flags: AGG_NONEMPTY | if multi { AGG_MULTI_ROW } else { 0 },
    }
}

/// What the exclusive prefix tells an entry.
#[derive(Debug, Copy, Clone)]
pub struct Before {
    /// Strips emitted before the entry, including all terminators of earlier rows.
    pub strips: u32,
    /// Alpha blocks allocated before the entry.
    pub blocks: u32,
    /// Winding at the left edge of the entry's tile, excluding fractional parts.
    pub winding: i32,
}

/// Resolve the exclusive prefix `p` against the entry's local facts.
pub fn before(p: &Agg, local: &Local) -> Before {
    let mut terminators = 0;
    if p.nonempty() {
        if p.multi_row() {
            // The first row ended inside `p`; the last row ends here if this entry starts a row.
            terminators += u32::from(p.w_first != 0);
            terminators += u32::from(local.row_head && p.w_last != 0);
        } else {
            terminators += u32::from(local.row_head && p.w_first != 0);
        }
    }
    let winding = if local.row_head {
        local.coarse
    } else {
        p.w_last
    };
    Before {
        strips: p.strips + terminators,
        blocks: p.blocks,
        winding,
    }
}

/// Output of the alloc pass.
#[derive(Debug, Default)]
pub struct Alloc {
    /// Strip descriptors in strip order.
    pub descs: Vec<GpuStripDesc>,
    /// Alpha block of each entry's run.
    pub entry_block: Vec<u32>,
    /// Per-path `(strip_start, strip_end, block_start, block_end)`.
    pub path_ranges: Vec<[u32; 4]>,
    /// Total number of alpha blocks.
    pub n_blocks: u32,
}

fn desc(x: u32, y: u32, block_start: u32, info: u32, winding: i32, path: u32) -> GpuStripDesc {
    GpuStripDesc {
        xy: x | (y << 16),
        block_start,
        info,
        winding,
        path,
    }
}

/// The alloc pass for entry `i` with exclusive prefix `p`: write the strips the entry is
/// responsible for into `out.descs` at their indices, and its run's block.
pub fn emit(input: &StripInput, i: u32, p: &Agg, local: &Local, out: &mut Alloc) {
    let path = &input.paths[local.path as usize];
    let b = before(p, local);
    let tw = u32::from(Tile::WIDTH);
    let th = u32::from(Tile::HEIGHT);
    let vw = input.viewport_width();
    let mut s = b.strips;
    let mut blk = b.blocks;

    if local.path_head {
        out.path_ranges[local.path as usize][0] = s;
        out.path_ranges[local.path as usize][2] = blk;
    }
    if local.row_head && path.culled() {
        for k in 0..local.n_bg {
            let row = input.fill_row(path, local.bg_lo + k);
            out.descs[s as usize] = desc(0, row * th, blk, STRIP_SOLID, 0, local.path);
            out.descs[s as usize + 1] = desc(
                vw,
                row * th,
                blk + 1,
                STRIP_TERMINATOR | STRIP_FILL_GAP,
                0,
                local.path,
            );
            s += 2;
            blk += 1;
        }
        if local.captive {
            out.descs[s as usize] = desc(0, local.y * th, blk, STRIP_SOLID, 0, local.path);
            s += 1;
            blk += 1;
        }
    }
    if local.strip_head {
        let fill_gap = path.should_fill(b.winding) && local.x != 0;
        let info = STRIP_REAL | if fill_gap { STRIP_FILL_GAP } else { 0 };
        out.descs[s as usize] = desc(local.x * tw, local.y * th, blk, info, b.winding, local.path);
        s += 1;
    }
    if local.run_head {
        out.entry_block[i as usize] = blk;
        blk += 1;
    } else {
        out.entry_block[i as usize] = b.blocks - 1;
    }
    if local.row_last {
        let total = b.winding + local.d;
        if total != 0 {
            let info = STRIP_TERMINATOR
                | if path.should_fill(total) {
                    STRIP_FILL_GAP
                } else {
                    0
                };
            out.descs[s as usize] = desc(vw, local.y * th, blk, info, total, local.path);
            s += 1;
        }
    }
    if local.path_last {
        let mut last_y = local.y;
        for k in 0..local.n_trail {
            let row = input.fill_row(path, local.trail_lo + k);
            out.descs[s as usize] = desc(0, row * th, blk, STRIP_SOLID, 0, local.path);
            out.descs[s as usize + 1] = desc(
                vw,
                row * th,
                blk + 1,
                STRIP_TERMINATOR | STRIP_FILL_GAP,
                0,
                local.path,
            );
            s += 2;
            blk += 1;
            last_y = row;
        }
        out.descs[s as usize] = desc(0xffff, last_y * th, blk, STRIP_SENTINEL, 0, local.path);
        s += 1;
        out.path_ranges[local.path as usize][1] = s;
        out.path_ranges[local.path as usize][3] = blk;
        if i + 1 == input.entries.len() as u32 {
            out.descs.truncate(s as usize);
            out.n_blocks = blk;
        }
    }
}

/// Run the alloc pass sequentially over all entries.
pub fn alloc(input: &StripInput) -> Alloc {
    let n = input.entries.len();
    let mut out = Alloc {
        descs: vec![GpuStripDesc::default(); input.max_strips() as usize],
        entry_block: vec![0; n],
        path_ranges: vec![[0; 4]; input.paths.len()],
        n_blocks: 0,
    };
    let mut prefix = Agg::default();
    for i in 0..n as u32 {
        let (agg, local) = leaf(input, i);
        emit(input, i, &prefix, &local, &mut out);
        prefix = combine(&prefix, &agg);
    }
    out
}

/// The per-entry rasterization of the merge pass: the line's contribution to the 16 pixel
/// windings of its tile (column-major, `col * 4 + row`) and to the 4 per-scanline windings at
/// the tile's right edge. Returns `None` for horizontal lines.
///
/// This follows `strip::render` pixel by pixel: the trapezoid between the line and the pixel's
/// right edge, plus the line's y extent in the pixels to the left.
pub fn line_payload(line: [f32; 4], tile_x: u32, tile_y: u32) -> Option<([f32; 16], [f32; 4])> {
    let tw = Tile::WIDTH as f32;
    let th = Tile::HEIGHT as f32;
    let p0x = line[0] - tile_x as f32 * tw;
    let p0y = line[1] - tile_y as f32 * th;
    let p1x = line[2] - tile_x as f32 * tw;
    let p1y = line[3] - tile_y as f32 * th;
    if p0y == p1y {
        return None;
    }
    // Lines moving up (decreasing y) add winding.
    let sign = if p0y > p1y { 1.0_f32 } else { -1.0 };
    let (top_y, top_x, bot_y, bot_x) = if p0y < p1y {
        (p0y, p0x, p1y, p1x)
    } else {
        (p1y, p1x, p0y, p0x)
    };
    let dx = bot_x - top_x;
    let dy = bot_y - top_y;
    let vertical = dx == 0.0;
    let y_slope = dy / dx;
    let x_slope = dx / dy;
    let mut a = [0.0_f32; 16];
    let mut r = [0.0_f32; 4];
    for row in 0..Tile::HEIGHT as usize {
        let ymin = top_y.max(row as f32);
        let ymax = bot_y.min(row as f32 + 1.0);
        let mut acc = 0.0_f32;
        for col in 0..Tile::WIDTH as usize {
            let px_left = col as f32;
            let px_right = col as f32 + 1.0;
            // y where the line meets the pixel's left and right edges, clamped to the pixel
            // row and the line's extent. A vertical line belongs to the pixel whose left edge
            // it lies on.
            let edge_y = |edge_x: f32| {
                let t = edge_x - top_x;
                let raw = if vertical {
                    if t <= 0.0 { f32::MIN } else { f32::MAX }
                } else {
                    top_y + t * y_slope
                };
                raw.max(ymin).min(ymax)
            };
            let yl = edge_y(px_left);
            let yr = edge_y(px_right);
            let xl = top_x + (yl - top_y) * x_slope;
            let xr = top_x + (yr - top_y) * x_slope;
            let h = (yr - yl).abs();
            let area = h * (px_right - 0.5 * (xl + xr));
            a[col * 4 + row] += sign * area + acc;
            acc += sign * h;
        }
        r[row] = acc;
    }
    Some((a, r))
}

/// The merge pass: accumulate every entry's payload into its run's block, in fixed point as
/// the GPU does (`round(x * 65536)` per entry, summed in `i32`).
pub fn merge(input: &StripInput, alloc: &Alloc) -> Vec<i32> {
    let mut acc = vec![0_i32; (alloc.n_blocks * ACC_WORDS) as usize];
    for (i, entry) in input.entries.iter().enumerate() {
        let path = &input.paths[input.path_of_entry(i as u32) as usize];
        let (x, y) = entry_xy(*entry);
        let line = input.lines[(path.line_base + (entry[0] >> 5)) as usize];
        let Some((a, r)) = line_payload(line, x, y) else {
            continue;
        };
        let base = (alloc.entry_block[i] * ACC_WORDS) as usize;
        for (k, v) in a.iter().enumerate() {
            acc[base + k] += (v * FIXED_ONE).round() as i32;
        }
        for (k, v) in r.iter().enumerate() {
            acc[base + TILE_PIXELS as usize + k] += (v * FIXED_ONE).round() as i32;
        }
    }
    acc
}

/// Convert a per-pixel winding to an 8-bit alpha, as `strip::render` does.
pub fn resolve_alpha(path: &GpuPathInfo, w: f32) -> u8 {
    let coverage = if path.even_odd() {
        let im1 = (w * 0.5 + 0.5).floor();
        (w - 2.0 * im1).abs()
    } else {
        w.abs()
    };
    let alpha = (coverage * 255.0 + 0.5).min(255.0) as u8;
    match path.aliasing_threshold() {
        Some(threshold) => {
            if alpha >= threshold {
                255
            } else {
                0
            }
        }
        None => alpha,
    }
}

/// The resolve pass: per strip, walk its blocks carrying the per-scanline winding, and write
/// alphas and strips.
pub fn resolve(
    input: &StripInput,
    alloc: &Alloc,
    acc: &[i32],
) -> (Vec<vello_common::strip::Strip>, Vec<u8>) {
    let mut alphas = vec![0_u8; (alloc.n_blocks * TILE_PIXELS) as usize];
    let strips = alloc.descs.iter().map(GpuStripDesc::to_strip).collect();
    for (s, d) in alloc.descs.iter().enumerate() {
        match d.kind() {
            STRIP_SOLID => {
                let base = (d.block_start * TILE_PIXELS) as usize;
                alphas[base..base + TILE_PIXELS as usize].fill(255);
            }
            STRIP_REAL => {
                let path = &input.paths[d.path as usize];
                let n_blocks = alloc.descs[s + 1].block_start - d.block_start;
                let mut carry = [d.winding as f32; 4];
                if d.x() == 0 && path.culled() {
                    let row = input.row(path, u32::from(d.y()) / u32::from(Tile::HEIGHT));
                    for (c, p) in carry.iter_mut().zip(row.partial) {
                        *c += p;
                    }
                }
                for block in d.block_start..d.block_start + n_blocks {
                    let acc_base = (block * ACC_WORDS) as usize;
                    let alpha_base = (block * TILE_PIXELS) as usize;
                    for k in 0..TILE_PIXELS as usize {
                        let w = carry[k % 4] + acc[acc_base + k] as f32 / FIXED_ONE;
                        alphas[alpha_base + k] = resolve_alpha(path, w);
                    }
                    for (k, c) in carry.iter_mut().enumerate() {
                        *c += acc[acc_base + TILE_PIXELS as usize + k] as f32 / FIXED_ONE;
                    }
                }
            }
            _ => {}
        }
    }
    (strips, alphas)
}

/// Run the whole model: strips and alphas for the input, in the same layout as
/// `strip::render` produces them for the paths in order.
pub fn render(input: &StripInput) -> (Vec<vello_common::strip::Strip>, Vec<u8>) {
    if input.entries.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let alloc = alloc(input);
    let acc = merge(input, &alloc);
    resolve(input, &alloc, &acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_strips::test_util::{Scene, compare, tiger_scene};
    use alloc::format;
    use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape};
    use vello_common::peniko::Fill;

    /// Compare the model against `strip::render`: strips exactly, alphas within one unit.
    fn check(scene: &Scene, what: &str) {
        let (input, cpu_strips, cpu_alphas) = scene.build();
        let (strips, alphas) = render(&input);
        compare(what, (&strips, &alphas), (&cpu_strips, &cpu_alphas), 1);
    }

    #[test]
    fn combine_is_associative() {
        // Random aggregates over a small row space; `combine` must be associative on every
        // triple whose row sequences are consistent (sorted, contiguous).
        let mut state = 12345_u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % m) as u32
        };
        for _ in 0..20000 {
            let mut rows = [next(4), next(4), next(4), next(4)];
            rows.sort_unstable();
            let mk = |first: u32, last: u32, next: &mut dyn FnMut(u64) -> u32| {
                let w_first = next(5) as i32 - 2;
                let multi = first != last;
                Agg {
                    first_row: [0, first],
                    last_row: [0, last],
                    w_first,
                    w_last: if multi { next(5) as i32 - 2 } else { w_first },
                    strips: next(10),
                    blocks: next(10),
                    flags: AGG_NONEMPTY | if multi { AGG_MULTI_ROW } else { 0 },
                }
            };
            let a = mk(rows[0], rows[1], &mut next);
            let b = mk(rows[1], rows[2], &mut next);
            let c = mk(rows[2], rows[3], &mut next);
            let left = combine(&combine(&a, &b), &c);
            let right = combine(&a, &combine(&b, &c));
            assert_eq!(left, right, "a={a:?} b={b:?} c={c:?}");
            let e = Agg::default();
            assert_eq!(combine(&e, &a), a);
            assert_eq!(combine(&a, &e), a);
        }
    }

    #[test]
    fn single_shapes() {
        let circle = Circle::new((50.0, 40.0), 30.0).to_path(0.1);
        let star = Scene::star(Point::new(50.0, 40.0), 5, 15.0, 38.0);
        for (path, fill, threshold, what) in [
            (&circle, Fill::NonZero, None, "circle"),
            (&star, Fill::NonZero, None, "star nonzero"),
            (&star, Fill::EvenOdd, None, "star evenodd"),
            (&circle, Fill::NonZero, Some(128), "circle aliased"),
        ] {
            let mut scene = Scene::new(100, 80);
            scene.fill(path.clone(), Affine::IDENTITY, fill, threshold);
            check(&scene, what);
        }
    }

    #[test]
    fn culled_shapes() {
        // Extending past the left edge: culled windings and captive strips.
        let circle = Circle::new((10.0, 40.0), 35.0).to_path(0.1);
        let mut scene = Scene::new(100, 80);
        scene.fill(circle.clone(), Affine::IDENTITY, Fill::NonZero, None);
        check(&scene, "circle off left");
        // Surrounding the viewport horizontally: rows with winding but no tiles.
        let mut scene = Scene::new(100, 80);
        scene.fill(
            Rect::new(-20.0, 20.5, 130.0, 60.5).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        check(&scene, "wide rect");
        // Fully off to the left plus a shifted copy inside: culled rows and gaps.
        let mut scene = Scene::new(100, 80);
        scene.fill(
            Circle::new((-30.0, 40.0), 20.0).to_path(0.1),
            Affine::IDENTITY,
            Fill::NonZero,
            None,
        );
        scene.fill(circle, Affine::translate((40.0, 0.0)), Fill::EvenOdd, None);
        check(&scene, "far and shifted");
    }

    #[test]
    fn random_polygons() {
        for seed in 0..40_u64 {
            let fill = if seed % 2 == 0 {
                Fill::NonZero
            } else {
                Fill::EvenOdd
            };
            let mut scene = Scene::new(100, 80);
            scene.fill(
                Scene::polygon(seed, 3 + (seed % 9) as usize, 140.0, 110.0),
                Affine::translate((-20.0, -15.0)),
                fill,
                None,
            );
            check(&scene, &format!("polygon {seed}"));
        }
    }

    #[test]
    fn several_paths() {
        let mut scene = Scene::new(100, 80);
        for seed in 0..12_u64 {
            scene.fill(
                Scene::polygon(seed + 100, 4 + (seed % 5) as usize, 130.0, 100.0),
                Affine::translate((-15.0, -10.0)),
                if seed % 3 == 0 {
                    Fill::EvenOdd
                } else {
                    Fill::NonZero
                },
                if seed % 4 == 3 { Some(100) } else { None },
            );
        }
        check(&scene, "several paths");
    }

    #[test]
    fn tiger() {
        check(&tiger_scene(400, 300, Affine::scale(0.75)), "tiger");
        check(
            &tiger_scene(
                400,
                300,
                Affine::translate((-150.0, -100.0)) * Affine::scale(1.5),
            ),
            "tiger culled",
        );
    }
}
