// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_encoding::{
    BinHeader, BumpAllocators, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox, DrawMonoid,
    LineSoup, Path, PathBbox, PathMonoid, PathSegment, SegmentCount, Tile,
};

use super::infra::Buffer;

pub(crate) struct Buffers {
    // TODO: Packed is a bit special, but only a little bit
    pub(crate) packed: Vec<u8>,
    pub(crate) path_reduced: Buffer<PathMonoid>,
    pub(crate) path_monoids: Buffer<PathMonoid>,
    pub(crate) path_bboxes: Buffer<PathBbox>,
    pub(crate) draw_reduced: Buffer<DrawMonoid>,
    pub(crate) draw_monoids: Buffer<DrawMonoid>,
    pub(crate) info: Buffer<u32>,
    pub(crate) clip_inps: Buffer<Clip>,
    pub(crate) clip_els: Buffer<ClipElement>,
    pub(crate) clip_bics: Buffer<ClipBic>,
    pub(crate) clip_bboxes: Buffer<ClipBbox>,
    pub(crate) draw_bboxes: Buffer<DrawBbox>,
    // bump_alloc is a bit special, because there's only one of them (they are also never write-only)
    pub(crate) bump_alloc: BumpAllocators,
    pub(crate) bin_headers: Buffer<BinHeader>,
    pub(crate) paths: Buffer<Path>,
    // Bump allocated buffers
    pub(crate) lines: Buffer<LineSoup>,
    pub(crate) bin_data: Buffer<u32>,
    pub(crate) tiles: Buffer<Tile>,
    pub(crate) seg_counts: Buffer<SegmentCount>,
    pub(crate) segments: Buffer<PathSegment>,
    pub(crate) ptcl: Buffer<u32>,
}
