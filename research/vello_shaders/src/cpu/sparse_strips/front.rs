// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Front end: run the existing CPU pathtag, `bbox_clear` and flatten stages on the packed scene,
//! and turn the line soup into per-path line slices for the tiler.

use std::ops::Range;

use vello_encoding::{BumpAllocators, ConfigUniform, LineSoup, PathBbox, PathMonoid};

use super::super::{bbox_clear::bbox_clear_main, flatten::flatten_main};
use super::super::{pathtag_reduce::pathtag_reduce_main, pathtag_scan::pathtag_scan_main};
use super::tile::{Line, Point};

/// Tags per pathtag reduce workgroup (256 threads × 4 tags per word).
const PATH_TAGS_PER_REDUCE_WG: u32 = 4 * 256;
/// Tags per flatten workgroup.
const PATH_TAGS_PER_FLATTEN_WG: u32 = 256;

/// The flattened scene.
#[derive(Debug, Default)]
pub struct FlattenedScene {
    /// Per path: bbox, `draw_flags` (fill rule) and `trans_ix`, exactly as GPU flatten writes them.
    pub path_bboxes: Vec<PathBbox>,
    /// All lines, grouped by path.
    pub lines: Vec<Line>,
    /// Per path, the range of its lines in `lines`.
    pub ranges: Vec<Range<usize>>,
    /// Number of lines dropped because they were non-finite.
    pub n_non_finite: usize,
}

impl FlattenedScene {
    pub fn path_lines(&self, path_ix: usize) -> &[Line] {
        &self.lines[self.ranges[path_ix].clone()]
    }
}

/// Run pathtag reduce/scan, `bbox_clear` and flatten on the CPU.
///
/// `scene` is the packed scene (as `u32` words) and `config` the matching uniform.
pub fn flatten_scene(config: &ConfigUniform, scene: &[u32]) -> FlattenedScene {
    let n_paths = config.layout.n_paths as usize;
    let n_path_tags = config.layout.path_tags_size();
    if n_paths == 0 || n_path_tags == 0 {
        return FlattenedScene {
            path_bboxes: vec![PathBbox::default(); n_paths],
            ranges: vec![0..0; n_paths],
            ..Default::default()
        };
    }
    let path_tag_wgs = n_path_tags.div_ceil(PATH_TAGS_PER_REDUCE_WG);
    let flatten_wgs = n_path_tags.div_ceil(PATH_TAGS_PER_FLATTEN_WG);

    let mut reduced = vec![PathMonoid::default(); path_tag_wgs as usize];
    pathtag_reduce_main(path_tag_wgs, config, scene, &mut reduced);
    let mut tag_monoids = vec![PathMonoid::default(); path_tag_wgs as usize * 256];
    pathtag_scan_main(path_tag_wgs, config, scene, &reduced, &mut tag_monoids);

    let mut path_bboxes = vec![PathBbox::default(); n_paths];
    bbox_clear_main(config, &mut path_bboxes);
    let mut bump = BumpAllocators::default();
    let mut soup: Vec<LineSoup> = Vec::new();
    flatten_main(
        flatten_wgs,
        config,
        scene,
        &tag_monoids,
        &mut path_bboxes,
        &mut bump,
        &mut soup,
    );

    let mut lines = Vec::with_capacity(soup.len());
    let mut ranges = vec![0..0; n_paths];
    let mut n_non_finite = 0;
    let mut current: Option<usize> = None;
    for l in &soup {
        let path_ix = l.path_ix as usize;
        if current != Some(path_ix) {
            debug_assert!(
                current.is_none_or(|c| c < path_ix),
                "flatten must emit lines in path order"
            );
            if let Some(c) = current {
                ranges[c].end = lines.len();
            }
            ranges[path_ix] = lines.len()..lines.len();
            current = Some(path_ix);
        }
        let [x0, y0] = l.p0;
        let [x1, y1] = l.p1;
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            n_non_finite += 1;
            continue;
        }
        // Zero-length lines never contribute coverage or winding.
        if x0 == x1 && y0 == y1 {
            continue;
        }
        lines.push(Line::new(Point::new(x0, y0), Point::new(x1, y1)));
    }
    if let Some(c) = current {
        ranges[c].end = lines.len();
    }
    FlattenedScene {
        path_bboxes,
        lines,
        ranges,
        n_non_finite,
    }
}

/// Debug check of the closed-loop invariant the tiler relies on.
///
/// Returns the number of points whose in-degree and out-degree differ, i.e. 0 for a watertight
/// path. Points are compared bitwise.
pub fn watertight_violations(lines: &[Line]) -> usize {
    use std::collections::HashMap;
    let key = |p: Point| (p.x.to_bits(), p.y.to_bits());
    let mut degree: HashMap<(u32, u32), i32> = HashMap::new();
    for l in lines {
        *degree.entry(key(l.p0)).or_default() += 1;
        *degree.entry(key(l.p1)).or_default() -= 1;
    }
    degree.values().filter(|&&d| d != 0).count()
}
