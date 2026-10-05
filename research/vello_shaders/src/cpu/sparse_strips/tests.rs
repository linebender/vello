// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the sparse strips pipeline, plus helpers shared by the port's tests.

use super::make_strips::Rasterizer;
use super::reference::rasterize_path_reference;
use super::tile::{Line, Point};
use super::{Backend, MASK_WORDS_PER_TILE, StripRecord, StripSink, TILE_SIZE};

/// Closed polygon through `points`.
pub(super) fn polygon(points: &[(f32, f32)]) -> Vec<Line> {
    (0..points.len())
        .map(|i| {
            let (x0, y0) = points[i];
            let (x1, y1) = points[(i + 1) % points.len()];
            Line::new(Point::new(x0, y0), Point::new(x1, y1))
        })
        .filter(|l| l.p0 != l.p1)
        .collect()
}

/// Axis-aligned rectangle, clockwise in y-down coordinates.
pub(super) fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Line> {
    polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

/// Rasterize one path (as path 0) with the given backend.
pub(super) fn rasterize(
    lines: &[Line],
    even_odd: bool,
    width: u32,
    height: u32,
    backend: Backend,
) -> StripSink {
    let mut sink = StripSink::new(width.div_ceil(TILE_SIZE as u32));
    sink.begin_path(0);
    match backend {
        Backend::Reference => rasterize_path_reference(lines, even_odd, width, height, &mut sink),
        Backend::Skia => Rasterizer::new(width, height).rasterize_path(lines, even_odd, &mut sink),
    }
    sink
}

/// Decode a path's records into per-pixel masks over the full tile grid, row-major with stride
/// `width_in_tiles * 16`. Solid spans decode as `0xffff`.
pub(super) fn decode(
    records: &[StripRecord],
    masks: &[u32],
    path_ix: u32,
    width: u32,
    height: u32,
) -> Vec<u16> {
    let ts = TILE_SIZE as usize;
    let wt = width.div_ceil(TILE_SIZE as u32) as usize;
    let ht = height.div_ceil(TILE_SIZE as u32) as usize;
    let stride = wt * ts;
    let mut px = vec![0_u16; stride * ht * ts];
    let mut backdrop = vec![0_i32; wt * ht];
    for r in records.iter().filter(|r| r.path_ix == path_ix) {
        let (tx, ty) = (r.tile_x() as usize, r.tile_y() as usize);
        if r.is_delta() {
            if tx < wt {
                backdrop[ty * wt + tx] += r.delta_value() as i32;
            }
            continue;
        }
        let block = &masks[r.payload as usize * MASK_WORDS_PER_TILE..][..MASK_WORDS_PER_TILE];
        for y in 0..ts {
            for x in 0..ts {
                let w = block[y * (ts / 2) + x / 2];
                let m = if x % 2 == 0 { w & 0xffff } else { w >> 16 };
                px[(ty * ts + y) * stride + tx * ts + x] = m as u16;
            }
        }
    }
    for ty in 0..ht {
        let mut sum = 0;
        for tx in 0..wt {
            sum += backdrop[ty * wt + tx];
            assert!(
                (0..=1).contains(&sum),
                "backdrop out of range at {tx},{ty}: {sum}"
            );
            if sum == 1 {
                for y in 0..ts {
                    for x in 0..ts {
                        let p = &mut px[(ty * ts + y) * stride + tx * ts + x];
                        assert_eq!(*p, 0, "solid span overlaps a mask tile at {tx},{ty}");
                        *p = 0xffff;
                    }
                }
            }
        }
        // A span reaching the last column has no closing delta, so `sum` may end at 1.
    }
    px
}

/// Total coverage in pixels: sum of popcount / 16 over the viewport.
pub(super) fn coverage(px: &[u16], width: u32, height: u32) -> f64 {
    let stride = (width.div_ceil(TILE_SIZE as u32) * TILE_SIZE as u32) as usize;
    let mut total = 0_u64;
    for y in 0..height as usize {
        for x in 0..width as usize {
            total += px[y * stride + x].count_ones() as u64;
        }
    }
    total as f64 / 16.0
}

fn render(lines: &[Line], even_odd: bool, w: u32, h: u32, backend: Backend) -> Vec<u16> {
    let sink = rasterize(lines, even_odd, w, h, backend);
    decode(&sink.records, &sink.masks, 0, w, h)
}

/// Tiny deterministic PRNG for randomized tests.
pub(super) struct XorShift(pub u64);

impl XorShift {
    pub(super) fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1_u64 << 24) as f32
    }
}

/// Random star-shaped (hence simple) polygon around `(cx, cy)`, with its exact area.
pub(super) fn star_polygon(
    rng: &mut XorShift,
    cx: f32,
    cy: f32,
    r: f32,
    n: usize,
) -> (Vec<Line>, f64) {
    let mut pts: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let a = (i as f32 + 0.8 * rng.next_f32()) / n as f32 * std::f32::consts::TAU;
            let rr = r * (0.3 + 0.7 * rng.next_f32());
            (cx + rr * a.cos(), cy + rr * a.sin())
        })
        .collect();
    pts.dedup();
    let mut area = 0.0_f64;
    for i in 0..pts.len() {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % pts.len()];
        area += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
    }
    (polygon(&pts), (area / 2.0).abs())
}

fn check_rect_exact(backend: Backend) {
    let (w, h) = (64, 64);
    let px = render(&rect(16.0, 16.0, 48.0, 40.0), false, w, h, backend);
    assert!(px.iter().all(|&m| m == 0 || m == 0xffff));
    assert_eq!(coverage(&px, w, h), 32.0 * 24.0);
}

fn check_half_pixel_edge(backend: Backend) {
    let (w, h) = (48, 32);
    let px = render(&rect(10.5, 0.0, 30.0, 16.0), false, w, h, backend);
    let stride = 48;
    for y in 0..16 {
        assert_eq!(px[y * stride + 10].count_ones(), 8, "row {y}");
        assert_eq!(px[y * stride + 9], 0);
        assert_eq!(px[y * stride + 11], 0xffff);
    }
}

fn check_fill_rules(backend: Backend) {
    let (w, h) = (64, 64);
    let mut lines = rect(8.0, 8.0, 40.0, 40.0);
    lines.extend(rect(24.0, 24.0, 56.0, 56.0));
    let stride = 64;
    let nz = render(&lines, false, w, h, backend);
    let eo = render(&lines, true, w, h, backend);
    assert_eq!(nz[30 * stride + 30], 0xffff);
    assert_eq!(eo[30 * stride + 30], 0);
    assert_eq!(eo[12 * stride + 12], 0xffff);
    assert_eq!(coverage(&nz, w, h), 32.0 * 32.0 * 2.0 - 16.0 * 16.0);
    assert_eq!(coverage(&eo, w, h), 32.0 * 32.0 * 2.0 - 2.0 * 16.0 * 16.0);
}

fn check_left_of_viewport(backend: Backend) {
    let (w, h) = (64, 48);
    let px = render(&rect(-100.0, 4.0, 20.0, 44.0), false, w, h, backend);
    assert_eq!(coverage(&px, w, h), 20.0 * 40.0);
    // Extends past every edge of the viewport.
    let px = render(&rect(-10.0, -10.0, 100.0, 100.0), false, w, h, backend);
    assert_eq!(coverage(&px, w, h), (w * h) as f64);
}

fn check_ragged_viewport(backend: Backend) {
    // Viewport not a multiple of the tile size.
    let (w, h) = (37, 21);
    let px = render(&rect(-5.0, -5.0, 50.0, 50.0), false, w, h, backend);
    assert_eq!(coverage(&px, w, h), (w * h) as f64);
}

fn check_random_polygon_area(backend: Backend) {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    let (w, h) = (256, 256);
    for i in 0..40 {
        let n = 3 + (i % 12);
        let radius = 20.0 + 100.0 * rng.next_f32();
        let (lines, area) = star_polygon(&mut rng, 128.0, 128.0, radius, n);
        let even_odd = i % 2 == 1;
        let px = render(&lines, even_odd, w, h, backend);
        let cov = coverage(&px, w, h);
        let perimeter: f64 = lines
            .iter()
            .map(|l| ((l.p1.x - l.p0.x) as f64).hypot((l.p1.y - l.p0.y) as f64))
            .sum();
        // Sampling error is bounded by roughly one sample-row's worth per unit of edge length.
        let tol = perimeter / 16.0 + 1.0;
        assert!(
            (cov - area).abs() <= tol,
            "polygon {i}: coverage {cov} vs area {area} (tol {tol})"
        );
    }
}

#[test]
fn reference_rect_exact() {
    check_rect_exact(Backend::Reference);
}

#[test]
fn reference_half_pixel_edge() {
    check_half_pixel_edge(Backend::Reference);
}

#[test]
fn reference_fill_rules() {
    check_fill_rules(Backend::Reference);
}

#[test]
fn reference_left_of_viewport() {
    check_left_of_viewport(Backend::Reference);
}

#[test]
fn reference_ragged_viewport() {
    check_ragged_viewport(Backend::Reference);
}

#[test]
fn reference_random_polygon_area() {
    check_random_polygon_area(Backend::Reference);
}

#[test]
fn skia_rect_exact() {
    check_rect_exact(Backend::Skia);
}

#[test]
fn skia_half_pixel_edge() {
    check_half_pixel_edge(Backend::Skia);
}

#[test]
fn skia_fill_rules() {
    check_fill_rules(Backend::Skia);
}

#[test]
fn skia_left_of_viewport() {
    check_left_of_viewport(Backend::Skia);
}

#[test]
fn skia_ragged_viewport() {
    check_ragged_viewport(Backend::Skia);
}

#[test]
fn skia_random_polygon_area() {
    check_random_polygon_area(Backend::Skia);
}

#[test]
fn sink_merges_adjacent_spans() {
    let mut sink = StripSink::new(8);
    sink.begin_path(3);
    sink.add_tile_span(1, 3, 2);
    sink.add_tile_span(3, 5, 2);
    // Full tile right after: also merges.
    sink.scratch_mut().fill(u32::MAX);
    sink.push_mask_tile(5, 2);
    assert_eq!(
        sink.records,
        vec![
            StripRecord::delta(3, 1, 2, 1),
            StripRecord::delta(3, 6, 2, -1)
        ]
    );
    // Empty tiles are dropped; a span reaching the last column has no closing delta.
    sink.scratch_mut().fill(0);
    sink.push_mask_tile(7, 0);
    sink.add_tile_span(6, 9, 0);
    assert_eq!(sink.records.len(), 3);
    assert_eq!(sink.records[2], StripRecord::delta(3, 6, 0, 1));
}

/// Statistics of the per-pixel difference between two decoded outputs over the viewport.
#[derive(Clone, Copy, Debug, Default)]
struct DiffStats {
    /// Sum over pixels of |popcount difference|.
    popcount: u64,
    /// Sum over pixels of the number of differing samples.
    samples: u64,
    /// Number of differing pixels.
    pixels: u64,
    /// Largest |popcount difference| of a pixel.
    max: u32,
}

impl DiffStats {
    fn of(a: &[u16], b: &[u16], width: u32, height: u32) -> Self {
        let stride = (width.div_ceil(TILE_SIZE as u32) * TILE_SIZE as u32) as usize;
        let mut stats = Self::default();
        for y in 0..height as usize {
            for x in 0..width as usize {
                let (a, b) = (a[y * stride + x], b[y * stride + x]);
                let d = a.count_ones().abs_diff(b.count_ones());
                stats.popcount += u64::from(d);
                stats.samples += u64::from((a ^ b).count_ones());
                stats.pixels += u64::from(a != b);
                stats.max = stats.max.max(d);
            }
        }
        stats
    }

    fn add(&mut self, other: Self) {
        self.popcount += other.popcount;
        self.samples += other.samples;
        self.pixels += other.pixels;
        self.max = self.max.max(other.max);
    }
}

fn perimeter(lines: &[Line]) -> f64 {
    lines
        .iter()
        .map(|l| f64::from(l.p1.x - l.p0.x).hypot(f64::from(l.p1.y - l.p0.y)))
        .sum()
}

/// A random test path: a star polygon for even `i`, else a random (generally self-intersecting)
/// polygon. Both may extend past any edge of a `width` x `height` viewport.
fn random_path(rng: &mut XorShift, i: usize, width: u32, height: u32) -> Vec<Line> {
    let (w, h) = (width as f32, height as f32);
    let n = 3 + i % 14;
    if i.is_multiple_of(2) {
        let cx = -0.1 * w + 1.2 * w * rng.next_f32();
        let cy = -0.1 * h + 1.2 * h * rng.next_f32();
        let r = 4.0 + 0.5 * w.min(h) * rng.next_f32();
        star_polygon(rng, cx, cy, r, n).0
    } else {
        let pts: Vec<(f32, f32)> = (0..n)
            .map(|_| {
                (
                    -0.15 * w + 1.3 * w * rng.next_f32(),
                    -0.15 * h + 1.3 * h * rng.next_f32(),
                )
            })
            .collect();
        polygon(&pts)
    }
}

/// The LUT quantizes the line offset and slope, so samples very close to an edge may differ
/// from the exact reference. This bounds the summed |popcount difference| of a path relative to
/// its perimeter (which overestimates the edge length inside the viewport).
fn max_popcount_diff(perimeter: f64) -> f64 {
    0.25 * perimeter + 8.0
}

#[test]
fn skia_matches_reference() {
    let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
    // A ragged viewport.
    let (w, h) = (200, 150);
    let mut total = DiffStats::default();
    let mut total_perimeter = 0.0;
    for i in 0..80 {
        let lines = random_path(&mut rng, i, w, h);
        let perimeter = perimeter(&lines);
        for even_odd in [false, true] {
            let skia = render(&lines, even_odd, w, h, Backend::Skia);
            let reference = render(&lines, even_odd, w, h, Backend::Reference);
            let stats = DiffStats::of(&skia, &reference, w, h);
            assert!(
                stats.popcount as f64 <= max_popcount_diff(perimeter),
                "path {i} even_odd {even_odd}: {stats:?}, perimeter {perimeter}: {lines:?}"
            );
            total.add(stats);
            total_perimeter += perimeter;
        }
    }
    println!(
        "skia vs reference, per px of perimeter: |popcount diff| {:.4}, differing samples {:.4}, \
         differing pixels {:.4}; max |popcount diff| of a pixel {}",
        total.popcount as f64 / total_perimeter,
        total.samples as f64 / total_perimeter,
        total.pixels as f64 / total_perimeter,
        total.max,
    );
}

#[test]
fn skia_matches_reference_across_paths() {
    // One rasterizer for a whole scene, as in `render_sparse_strips_with`: the winding histogram
    // and tile buffers are reused across paths, and must be reset correctly in between.
    let mut rng = XorShift(0x0123_4567_89ab_cdef);
    let (w, h) = (130, 100);
    let paths: Vec<Vec<Line>> = (0..40).map(|i| random_path(&mut rng, i, w, h)).collect();
    let mut skia = StripSink::new(w.div_ceil(TILE_SIZE as u32));
    let mut reference = StripSink::new(w.div_ceil(TILE_SIZE as u32));
    let mut rasterizer = Rasterizer::new(w, h);
    for (path_ix, lines) in paths.iter().enumerate() {
        let even_odd = path_ix % 3 == 0;
        skia.begin_path(path_ix as u32);
        rasterizer.rasterize_path(lines, even_odd, &mut skia);
        reference.begin_path(path_ix as u32);
        rasterize_path_reference(lines, even_odd, w, h, &mut reference);
    }
    for (path_ix, lines) in paths.iter().enumerate() {
        let a = decode(&skia.records, &skia.masks, path_ix as u32, w, h);
        let b = decode(&reference.records, &reference.masks, path_ix as u32, w, h);
        let stats = DiffStats::of(&a, &b, w, h);
        let perimeter = perimeter(lines);
        assert!(
            stats.popcount as f64 <= max_popcount_diff(perimeter),
            "path {path_ix}: {stats:?}, perimeter {perimeter}: {lines:?}"
        );
    }
}
