// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU sparse strip generation (experimental).
//!
//! This module moves the strip rendering stage of the sparse strips pipeline, which is
//! [`vello_common::strip::render`] on the CPU, to the GPU. Flattening, tiling and sorting still
//! happen on the CPU; the GPU consumes the sorted tiles of one or more paths and produces the
//! same strips and alphas that `strip::render` would produce for each path.
//!
//! The GPU pipeline has these stages, each a compute dispatch:
//!
//! 1. **reduce**: per workgroup of entries, the aggregate of the allocation monoid
//!    ([`model::Agg`]).
//! 2. **scan**: one workgroup scans the workgroup aggregates into exclusive prefixes.
//! 3. **alloc**: every entry knows its exclusive prefix and emits strip descriptors
//!    ([`GpuStripDesc`]), the alpha block of its run, and per-path strip and block ranges.
//! 4. **merge**: every entry rasterizes its line into fixed-point per-pixel winding deltas and
//!    these are summed per run (tile location), first in workgroup memory and then with global
//!    atomics for runs that span workgroups.
//! 5. **resolve**: one thread per strip walks the strip's runs left to right, carrying the
//!    per-scanline winding across tiles, applies the fill rule and writes alphas and strips.
//!
//! The strip geometry (positions, fill gaps, alpha indices, terminator and sentinel strips) is
//! integer and matches the CPU exactly. Alpha values are accumulated in a different order and in
//! fixed point, so they may differ from the CPU by one unit.
//!
//! [`model`] is a CPU implementation of exactly this algorithm. It documents the GPU logic, is
//! tested against `strip::render` without a GPU, and is what the GPU output is diffed against.
//!
//! ## Limitations
//!
//! - A path with no tiles but with geometry culled to the left of the viewport produces strips
//!   on the CPU (solid rows). It produces no entries here, so [`StripInput::add_path`] reports
//!   it and the caller must fall back to the CPU for that path.

#![allow(
    clippy::cast_possible_truncation,
    reason = "counts and coordinates are bounded by the GPU formats they are cast to"
)]

pub mod model;

#[cfg(feature = "wgpu")]
pub mod gpu;

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::flatten::Line;
use vello_common::peniko::Fill;
use vello_common::tile::{Tile, Tiles};

/// Number of pixels in a tile.
pub const TILE_PIXELS: u32 = Tile::WIDTH as u32 * Tile::HEIGHT as u32;
/// `u32` words per alpha block (one tile's worth of 8-bit alphas).
pub const BLOCK_WORDS: u32 = TILE_PIXELS / 4;
/// `i32` words per accumulation block: 16 per-pixel windings and 4 per-scanline carries.
pub const ACC_WORDS: u32 = TILE_PIXELS + Tile::HEIGHT as u32;
/// Fixed-point scale of the accumulated windings.
pub const FIXED_ONE: f32 = 65536.0;
/// `u32` words per row table entry ([`GpuRow`]).
pub const ROW_WORDS: u32 = 8;

/// Workgroup size of the reduce and alloc passes (entries per workgroup).
pub const SCAN_WG: u32 = 256;
/// Workgroup size of the merge pass (entries per workgroup).
pub const MERGE_WG: u32 = 128;
/// Workgroup size of the resolve pass (strips per workgroup).
pub const RESOLVE_WG: u32 = 64;

/// Set in [`GpuPathInfo::flags`] for even-odd fills.
pub const PATH_EVEN_ODD: u32 = 1;
/// Set in [`GpuPathInfo::flags`] when the path has geometry culled left of the viewport, in
/// which case its row table is valid.
pub const PATH_CULLED: u32 = 2;
/// Set in [`GpuPathInfo::flags`] when an aliasing threshold applies; the threshold is in
/// bits 8..16.
pub const PATH_THRESHOLD: u32 = 4;

/// Strip descriptor kinds, in bits 0..2 of [`GpuStripDesc::info`].
pub const STRIP_REAL: u32 = 0;
/// A solid tile-wide strip (captive or culled-background strip) with a block of 255 alphas.
pub const STRIP_SOLID: u32 = 1;
/// A zero-width row terminator strip at the viewport width.
pub const STRIP_TERMINATOR: u32 = 2;
/// The sentinel strip ending a path.
pub const STRIP_SENTINEL: u32 = 3;
/// Set in [`GpuStripDesc::info`] when the strip's gap to the previous strip is filled.
pub const STRIP_FILL_GAP: u32 = 4;

/// Uniform configuration of the GPU passes.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
pub struct GpuConfig {
    /// Number of entries (tiles) over all paths.
    pub n_entries: u32,
    /// Number of paths.
    pub n_paths: u32,
    /// Number of tile rows in the viewport.
    pub rows: u32,
    /// Viewport width in pixels, rounded up to a multiple of the tile width.
    pub viewport_width: u32,
    /// Number of workgroups of the reduce pass.
    pub n_wg: u32,
    /// Capacity of the strip descriptor and strip buffers.
    pub max_strips: u32,
    /// Capacity of the accumulation and alpha buffers, in blocks.
    pub max_blocks: u32,
    /// Padding to 16 bytes.
    pub _pad: u32,
}

/// Per-path input to the GPU passes. The last four fields are written by the alloc pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
pub struct GpuPathInfo {
    /// First entry of this path.
    pub entry_start: u32,
    /// One past the last entry of this path.
    pub entry_end: u32,
    /// Offset of this path's lines in the line buffer.
    pub line_base: u32,
    /// Word offset of this path's row table in the aux buffer (valid when `PATH_CULLED`).
    pub row_base: u32,
    /// Word offset of this path's list of filled culled rows in the aux buffer.
    pub fill_row_base: u32,
    /// Number of filled culled rows.
    pub n_fill_rows: u32,
    /// `PATH_*` flags and the aliasing threshold.
    pub flags: u32,
    /// Padding.
    pub _pad: u32,
    /// First strip of this path (written by the GPU).
    pub strip_start: u32,
    /// One past the last strip of this path (written by the GPU).
    pub strip_end: u32,
    /// First alpha block of this path (written by the GPU).
    pub block_start: u32,
    /// One past the last alpha block of this path (written by the GPU).
    pub block_end: u32,
}

impl GpuPathInfo {
    /// Whether the fill rule is even-odd.
    pub fn even_odd(&self) -> bool {
        self.flags & PATH_EVEN_ODD != 0
    }

    /// Whether the path has culled geometry (and therefore a row table).
    pub fn culled(&self) -> bool {
        self.flags & PATH_CULLED != 0
    }

    /// The aliasing threshold, if any.
    pub fn aliasing_threshold(&self) -> Option<u8> {
        (self.flags & PATH_THRESHOLD != 0).then_some((self.flags >> 8) as u8)
    }

    /// Whether `winding` fills under this path's fill rule.
    pub fn should_fill(&self, winding: i32) -> bool {
        if self.even_odd() {
            winding % 2 != 0
        } else {
            winding != 0
        }
    }
}

/// Row table entry for a culled path, stored in the aux buffer as [`ROW_WORDS`] words.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
pub struct GpuRow {
    /// Integer winding at the left viewport edge for this tile row.
    pub coarse: i32,
    /// Number of filled culled rows before this row (prefix of the filled row list).
    pub bg_prefix: u32,
    /// Padding to 32 bytes.
    pub _pad: [u32; 2],
    /// Fractional per-scanline winding at the left viewport edge.
    pub partial: [f32; 4],
}

/// Strip descriptor, emitted by the alloc pass and consumed by the resolve pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuStripDesc {
    /// Pixel x in the low half, pixel y in the high half.
    pub xy: u32,
    /// Number of alpha blocks allocated before this strip; the strip's alphas start here.
    pub block_start: u32,
    /// `STRIP_*` kind and flags.
    pub info: u32,
    /// Integer winding at the left edge of a real strip.
    pub winding: i32,
    /// Path index.
    pub path: u32,
}

impl GpuStripDesc {
    /// Pixel x.
    pub fn x(&self) -> u16 {
        self.xy as u16
    }

    /// Pixel y.
    pub fn y(&self) -> u16 {
        (self.xy >> 16) as u16
    }

    /// Strip kind.
    pub fn kind(&self) -> u32 {
        self.info & 3
    }

    /// Whether the gap before this strip is filled.
    pub fn fill_gap(&self) -> bool {
        self.info & STRIP_FILL_GAP != 0
    }

    /// Convert to a [`vello_common::strip::Strip`].
    pub fn to_strip(&self) -> vello_common::strip::Strip {
        vello_common::strip::Strip::new(
            self.x(),
            self.y(),
            self.block_start * TILE_PIXELS,
            self.fill_gap(),
        )
    }
}

/// Reports what [`StripInput::add_path`] did with a path.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AddPath {
    /// The path produces no strips at all.
    Empty,
    /// The path was added.
    Added,
    /// The path has no tiles but culled geometry; the GPU cannot represent it, so the caller
    /// must render it on the CPU.
    CpuFallback,
}

/// The concatenated, GPU-ready input of one or more paths.
#[derive(Debug, Default)]
pub struct StripInput {
    /// Viewport width in pixels.
    pub width: u16,
    /// Viewport height in pixels.
    pub height: u16,
    /// Lines of all paths, as `(x0, y0, x1, y1)`.
    pub lines: Vec<[f32; 4]>,
    /// Sorted tiles of all paths, path by path, as `(packed_winding_line_idx, x | y << 16)`.
    pub entries: Vec<[u32; 2]>,
    /// Per-path information.
    pub paths: Vec<GpuPathInfo>,
    /// Row tables and filled-row lists of culled paths.
    pub aux: Vec<u32>,
}

impl StripInput {
    /// Create an empty input for a viewport.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Number of tile rows in the viewport.
    pub fn rows(&self) -> u32 {
        u32::from(self.height).div_ceil(u32::from(Tile::HEIGHT))
    }

    /// Viewport width in pixels, rounded up to the tile width, as `strip::render` uses it.
    pub fn viewport_width(&self) -> u32 {
        u32::from(
            self.width
                .checked_next_multiple_of(Tile::WIDTH)
                .unwrap_or(u16::MAX),
        )
    }

    /// Reset the input, keeping allocations.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.entries.clear();
        self.paths.clear();
        self.aux.clear();
    }

    /// Add a path from its sorted tiles and lines.
    pub fn add_path(
        &mut self,
        tiles: &Tiles,
        lines: &[Line],
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
    ) -> AddPath {
        let culled = tiles.has_culled_tiles();
        if tiles.is_empty() {
            return if culled {
                AddPath::CpuFallback
            } else {
                AddPath::Empty
            };
        }
        let mut flags = 0;
        if fill_rule == Fill::EvenOdd {
            flags |= PATH_EVEN_ODD;
        }
        if let Some(threshold) = aliasing_threshold {
            flags |= PATH_THRESHOLD | (u32::from(threshold) << 8);
        }
        let mut info = GpuPathInfo {
            entry_start: self.entries.len() as u32,
            line_base: self.lines.len() as u32,
            flags,
            ..Default::default()
        };
        self.lines
            .extend(lines.iter().map(|l| [l.p0.x, l.p0.y, l.p1.x, l.p1.y]));
        self.entries.extend(tiles.iter().map(|t| {
            [
                t.packed_winding_line_idx,
                u32::from(t.x) | (u32::from(t.y) << 16),
            ]
        }));
        info.entry_end = self.entries.len() as u32;

        if culled {
            info.flags |= PATH_CULLED;
            let rows = self.rows() as usize;
            let windings = &tiles.windings;
            debug_assert_eq!(
                windings.coarse.len(),
                rows,
                "row table matches the viewport"
            );
            let should_fill = |w: i32| match fill_rule {
                Fill::NonZero => w != 0,
                Fill::EvenOdd => w % 2 != 0,
            };
            // The filled culled rows: active rows whose coarse winding fills. These are the
            // rows `strip::render` emits solid strips for when they lie between tile rows.
            let mut fill_rows = Vec::new();
            windings.for_active_rows_in_range(0, rows, |row| {
                if should_fill(i32::from(windings.coarse[row])) {
                    fill_rows.push(row as u32);
                }
            });
            info.row_base = self.aux.len() as u32;
            let mut next_fill = 0;
            for row in 0..rows {
                while next_fill < fill_rows.len() && (fill_rows[next_fill] as usize) < row {
                    next_fill += 1;
                }
                let gpu_row = GpuRow {
                    coarse: i32::from(windings.coarse[row]),
                    bg_prefix: next_fill as u32,
                    _pad: [0; 2],
                    partial: windings.partial[row],
                };
                self.aux.extend_from_slice(bytemuck::cast_slice(&[gpu_row]));
            }
            info.fill_row_base = self.aux.len() as u32;
            info.n_fill_rows = fill_rows.len() as u32;
            self.aux.extend_from_slice(&fill_rows);
        }
        self.paths.push(info);
        AddPath::Added
    }

    /// Upper bound on the number of strips the input produces.
    pub fn max_strips(&self) -> u32 {
        let rows = self.rows();
        self.paths
            .iter()
            .map(|p| {
                let n = p.entry_end - p.entry_start;
                // Per row with tiles: at most one terminator and one captive strip; per filled
                // culled row: two background strips; plus the sentinel.
                let row_strips = if p.culled() {
                    2 * n.min(rows) + 2 * p.n_fill_rows
                } else {
                    n.min(rows)
                };
                n + row_strips + 1
            })
            .sum()
    }

    /// Upper bound on the number of alpha blocks the input produces.
    pub fn max_blocks(&self) -> u32 {
        let rows = self.rows();
        self.paths
            .iter()
            .map(|p| {
                let n = p.entry_end - p.entry_start;
                let row_blocks = if p.culled() {
                    n.min(rows) + p.n_fill_rows
                } else {
                    0
                };
                n + row_blocks
            })
            .sum()
    }

    /// The uniform configuration for this input.
    pub fn config(&self) -> GpuConfig {
        GpuConfig {
            n_entries: self.entries.len() as u32,
            n_paths: self.paths.len() as u32,
            rows: self.rows(),
            viewport_width: self.viewport_width(),
            n_wg: (self.entries.len() as u32).div_ceil(SCAN_WG),
            max_strips: self.max_strips(),
            max_blocks: self.max_blocks(),
            _pad: 0,
        }
    }

    /// The row table entry of a culled path.
    pub fn row(&self, path: &GpuPathInfo, row: u32) -> GpuRow {
        let base = (path.row_base + row * ROW_WORDS) as usize;
        bytemuck::pod_read_unaligned(bytemuck::cast_slice(
            &self.aux[base..base + ROW_WORDS as usize],
        ))
    }

    /// The `k`-th filled culled row of a culled path.
    pub fn fill_row(&self, path: &GpuPathInfo, k: u32) -> u32 {
        self.aux[(path.fill_row_base + k) as usize]
    }

    /// The path index of an entry.
    pub fn path_of_entry(&self, entry: u32) -> u32 {
        // Binary search for the last path whose `entry_start <= entry`, as the shaders do.
        let mut lo = 0_u32;
        let mut hi = self.paths.len() as u32;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.paths[mid as usize].entry_start <= entry {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Test scaffolding shared by the model and GPU tests: build a [`StripInput`] and the CPU
/// reference output for a set of paths, and compare outputs.
#[cfg(test)]
pub(crate) mod test_util {
    extern crate std;

    use super::{AddPath, StripInput};
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use vello_common::fearless_simd::Level;
    use vello_common::flatten::{self, FlattenCtx, Line};
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, BezPath, Point, Stroke, StrokeCtx};
    use vello_common::peniko::Fill;
    use vello_common::pico_svg::{Item, PicoSvg};
    use vello_common::strip::{self, Strip};
    use vello_common::tile::Tiles;

    /// A path to render: filled or stroked.
    pub(crate) struct ScenePath {
        pub path: BezPath,
        pub transform: Affine,
        pub fill: Fill,
        pub stroke: Option<Stroke>,
        pub aliasing_threshold: Option<u8>,
    }

    /// A viewport and a list of paths.
    pub(crate) struct Scene {
        pub width: u16,
        pub height: u16,
        pub paths: Vec<ScenePath>,
    }

    impl Scene {
        pub(crate) fn new(width: u16, height: u16) -> Self {
            Self {
                width,
                height,
                paths: Vec::new(),
            }
        }

        pub(crate) fn fill(
            &mut self,
            path: BezPath,
            transform: Affine,
            fill: Fill,
            aliasing_threshold: Option<u8>,
        ) {
            self.paths.push(ScenePath {
                path,
                transform,
                fill,
                stroke: None,
                aliasing_threshold,
            });
        }

        pub(crate) fn stroke(&mut self, path: BezPath, transform: Affine, stroke: Stroke) {
            self.paths.push(ScenePath {
                path,
                transform,
                fill: Fill::NonZero,
                stroke: Some(stroke),
                aliasing_threshold: None,
            });
        }

        /// Flatten, tile and sort every path, add it to a [`StripInput`], and render the CPU
        /// reference strips and alphas for the same paths into shared buffers.
        pub(crate) fn build(&self) -> (StripInput, Vec<Strip>, Vec<u8>) {
            let level = Level::baseline();
            let (width, height) = (self.width, self.height);
            let mut input = StripInput::new(width, height);
            let mut cpu_strips: Vec<Strip> = Vec::new();
            let mut cpu_alphas: Vec<u8> = Vec::new();
            let mut lines: Vec<Line> = Vec::new();
            let mut ctx = FlattenCtx::default();
            let mut stroke_ctx = StrokeCtx::default();
            let mut tiles = Tiles::new(level, width, height);
            let bbox = RectU16::new(0, 0, width, height);
            for p in &self.paths {
                let els = p.path.elements().iter().copied();
                if let Some(stroke) = &p.stroke {
                    flatten::stroke(
                        level,
                        els,
                        stroke,
                        p.transform,
                        &mut lines,
                        &mut ctx,
                        &mut stroke_ctx,
                        bbox,
                    );
                } else {
                    flatten::fill(level, els, p.transform, &mut lines, &mut ctx, bbox);
                }
                tiles.reset(width, height);
                tiles.make_tiles_analytic_aa(level, &lines, width, height);
                tiles.sort_tiles();
                if input.add_path(&tiles, &lines, p.fill, p.aliasing_threshold)
                    == AddPath::CpuFallback
                {
                    continue;
                }
                strip::render(
                    level,
                    &tiles,
                    &mut cpu_strips,
                    &mut cpu_alphas,
                    p.fill,
                    p.aliasing_threshold,
                    &lines,
                );
            }
            (input, cpu_strips, cpu_alphas)
        }

        pub(crate) fn star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
            let mut path = BezPath::new();
            for i in 0..2 * n {
                let r = if i % 2 == 0 { outer } else { inner };
                let a = i as f64 * core::f64::consts::PI / n as f64;
                let p = center + (r * a.cos(), r * a.sin());
                if i == 0 {
                    path.move_to(p);
                } else {
                    path.line_to(p);
                }
            }
            path.close_path();
            path
        }

        /// Deterministic pseudo-random polygon.
        pub(crate) fn polygon(seed: u64, n: usize, w: f64, h: f64) -> BezPath {
            let mut state = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1_u64 << 53) as f64
            };
            let mut path = BezPath::new();
            for i in 0..n {
                let p = Point::new(next() * w, next() * h);
                if i == 0 {
                    path.move_to(p);
                } else {
                    path.line_to(p);
                }
            }
            path.close_path();
            path
        }
    }

    /// The Ghostscript tiger from the repository assets, fills and strokes.
    pub(crate) fn tiger_scene(width: u16, height: u16, transform: Affine) -> Scene {
        let svg = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/Ghostscript_Tiger.svg"
        ))
        .expect("tiger asset");
        let parsed = PicoSvg::load(&svg, 1.0).expect("tiger parses");
        let mut scene = Scene::new(width, height);
        fn add(scene: &mut Scene, items: &[Item], transform: Affine) {
            for item in items {
                match item {
                    Item::Fill(f) => scene.fill(f.path.clone(), transform, Fill::NonZero, None),
                    Item::Stroke(s) => {
                        scene.stroke(s.path.clone(), transform, Stroke::new(s.width));
                    }
                    Item::Group(g) => add(scene, &g.children, transform * g.affine),
                }
            }
        }
        add(&mut scene, &parsed.items, transform);
        scene
    }

    /// Strips must match exactly; alphas within `tolerance`.
    pub(crate) fn compare(
        what: &str,
        got: (&[Strip], &[u8]),
        want: (&[Strip], &[u8]),
        tolerance: u32,
    ) {
        let (strips, alphas) = got;
        let (want_strips, want_alphas) = want;
        assert_eq!(
            strips.len(),
            want_strips.len(),
            "{what}: {} strips, expected {}",
            strips.len(),
            want_strips.len()
        );
        for (i, (g, w)) in strips.iter().zip(want_strips).enumerate() {
            assert_eq!(g, w, "{what}: strip {i} differs");
        }
        assert_eq!(alphas.len(), want_alphas.len(), "{what}: alpha length");
        let mut worst = 0;
        let mut first_bad: Option<String> = None;
        for (i, (g, w)) in alphas.iter().zip(want_alphas).enumerate() {
            let diff = (i32::from(*g) - i32::from(*w)).unsigned_abs();
            worst = worst.max(diff);
            if diff > tolerance && first_bad.is_none() {
                first_bad = Some(format!("alpha {i}: got {g}, expected {w}"));
            }
        }
        assert!(
            worst <= tolerance,
            "{what}: worst alpha difference {worst}; {first_bad:?}"
        );
    }
}
