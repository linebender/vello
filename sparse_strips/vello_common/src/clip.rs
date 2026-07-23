// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Managing clipping state.

use crate::geometry::RectU16;
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, BezPath, PathEl, Point};
use crate::strip::Strip;
use crate::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use crate::tile::Tile;
use crate::util::{Clear, Pool, normalized_mul_u8x16, strip_bbox};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use fearless_simd::{Level, Simd, SimdBase, dispatch, u8x16};
use peniko::Fill;

/// Maximum number of rectangles tracked in an [`IntRectSet`].
pub const MAX_INT_CLIP_RECTS: usize = 16;

/// Maximum number of path elements captured for integer-rectangle clip detection: a set of
/// [`MAX_INT_CLIP_RECTS`] rectangles needs at most `MoveTo + 5 LineTo + ClosePath` per
/// rectangle. Longer clip paths conservatively fail detection.
const MAX_DETECT_ELEMENTS: usize = MAX_INT_CLIP_RECTS * 7;

/// Tolerance for treating a device-space clip edge as lying on an integer pixel boundary.
///
/// An edge within `1/512` of an integer produces the same 0/255 anti-aliasing bytes as the
/// integer edge itself (`255 / 512 + 0.5 < 1`), so snapping is byte-exact.
const INT_EDGE_EPSILON: f64 = 1.0 / 512.0;

/// A small inline set of pairwise-disjoint, axis-aligned integer rectangles.
///
/// Used to mark clip stacks whose combined effect is exactly a union of such rectangles:
/// anti-aliasing on integer edges degenerates to a 0/255 step, so clipping content to the set
/// is byte-identical to intersecting with the rasterized clip mask.
#[derive(Debug, Clone, Copy)]
pub struct IntRectSet {
    rects: [RectU16; MAX_INT_CLIP_RECTS],
    len: u8,
}

impl IntRectSet {
    fn new() -> Self {
        Self {
            rects: [RectU16::ZERO; MAX_INT_CLIP_RECTS],
            len: 0,
        }
    }

    /// Add a rectangle to the set. Empty rectangles are ignored. Returns `false` when the set
    /// is full.
    fn push(&mut self, rect: RectU16) -> bool {
        if rect.is_empty() {
            return true;
        }
        if (self.len as usize) == MAX_INT_CLIP_RECTS {
            return false;
        }
        self.rects[self.len as usize] = rect;
        self.len += 1;
        true
    }

    /// The rectangles in the set.
    #[inline]
    pub fn as_slice(&self) -> &[RectU16] {
        &self.rects[..self.len as usize]
    }

    /// Intersect two disjoint sets. The pairwise intersections of two internally disjoint
    /// families are themselves pairwise disjoint. Returns `None` when the result would exceed
    /// [`MAX_INT_CLIP_RECTS`].
    fn intersect(&self, other: &Self) -> Option<Self> {
        let mut out = Self::new();
        for a in self.as_slice() {
            for b in other.as_slice() {
                if !out.push(a.intersect(*b)) {
                    return None;
                }
            }
        }
        Some(out)
    }
}

/// Decompose `path` (after `transform`) into a set of pairwise-disjoint, axis-aligned
/// rectangles whose edges lie on integer device coordinates once clamped to `viewport`.
///
/// Returns `None` if any subpath is not such a rectangle (curves, diagonal edges, fractional
/// in-viewport edges), if the rectangles overlap, or if there are more than
/// [`MAX_INT_CLIP_RECTS`] of them. Rectangles fully outside `viewport` are dropped.
pub fn path_as_integer_rect_set(
    path: impl IntoIterator<Item = PathEl>,
    transform: Affine,
    viewport: RectU16,
) -> Option<IntRectSet> {
    let mut set = IntRectSet::new();
    // Corners of the subpath currently being walked: the `MoveTo` point plus up to four
    // `LineTo` points (the last of which may return to the start).
    let mut pts = [Point::ZERO; 6];
    let mut n = 0_usize;
    let mut open = false;

    let finish = |pts: &[Point], set: &mut IntRectSet| -> bool {
        match subpath_as_integer_rect(pts, viewport) {
            SubpathRect::Rect(rect) => set.push(rect),
            SubpathRect::Empty => true,
            SubpathRect::NotARect => false,
        }
    };

    for el in path {
        match el {
            PathEl::MoveTo(p) => {
                if open && !finish(&pts[..n], &mut set) {
                    return None;
                }
                pts[0] = transform * p;
                n = 1;
                open = true;
            }
            PathEl::LineTo(p) => {
                if !open || n == pts.len() {
                    return None;
                }
                pts[n] = transform * p;
                n += 1;
            }
            PathEl::QuadTo(..) | PathEl::CurveTo(..) => return None,
            PathEl::ClosePath => {
                if open && !finish(&pts[..n], &mut set) {
                    return None;
                }
                open = false;
            }
        }
    }
    if open && !finish(&pts[..n], &mut set) {
        return None;
    }

    // Overlapping rectangles form a union we cannot decompose here; reject them.
    let rects = set.as_slice();
    for (i, a) in rects.iter().enumerate() {
        for b in &rects[i + 1..] {
            if !a.intersect(*b).is_empty() {
                return None;
            }
        }
    }

    Some(set)
}

/// The result of interpreting one subpath as an integer rectangle.
enum SubpathRect {
    Rect(RectU16),
    /// A valid rectangle that clamps to nothing inside the viewport.
    Empty,
    NotARect,
}

/// Interpret already-transformed subpath corners as an axis-aligned rectangle loop with
/// integer device edges (after clamping to `viewport`).
fn subpath_as_integer_rect(pts: &[Point], viewport: RectU16) -> SubpathRect {
    let mut n = pts.len();
    // Drop an explicit closing point that returns to the start.
    if n == 5 && point_eq(pts[4], pts[0]) {
        n = 4;
    }
    if n != 4 {
        return SubpathRect::NotARect;
    }

    // Consecutive edges (including the closing one) must be axis-parallel and alternate
    // between horizontal and vertical; this excludes diagonal "Z" quads whose corners still
    // span a rectangle.
    let horizontal = |a: Point, b: Point| (a.y - b.y).abs() <= INT_EDGE_EPSILON;
    let vertical = |a: Point, b: Point| (a.x - b.x).abs() <= INT_EDGE_EPSILON;
    let mut first_horizontal = false;
    for i in 0..4 {
        let (a, b) = (pts[i], pts[(i + 1) % 4]);
        let h = horizontal(a, b);
        let v = vertical(a, b);
        if h == v {
            // Degenerate (both) or diagonal (neither).
            return SubpathRect::NotARect;
        }
        if i == 0 {
            first_horizontal = h;
        } else if h != (first_horizontal == (i % 2 == 0)) {
            return SubpathRect::NotARect;
        }
    }

    let x0 = pts
        .iter()
        .take(4)
        .map(|p| p.x)
        .fold(f64::INFINITY, f64::min);
    let x1 = pts
        .iter()
        .take(4)
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let y0 = pts
        .iter()
        .take(4)
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min);
    let y1 = pts
        .iter()
        .take(4)
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);

    // Clamp to the viewport first: out-of-viewport edges land on the (integer) viewport
    // bounds, so only the surviving in-viewport edges need to be integers themselves.
    let x0 = x0.max(f64::from(viewport.x0)).min(f64::from(viewport.x1));
    let x1 = x1.max(f64::from(viewport.x0)).min(f64::from(viewport.x1));
    let y0 = y0.max(f64::from(viewport.y0)).min(f64::from(viewport.y1));
    let y1 = y1.max(f64::from(viewport.y0)).min(f64::from(viewport.y1));

    let snap = |v: f64| -> Option<u16> {
        let r = v.round();
        ((v - r).abs() <= INT_EDGE_EPSILON).then_some(r as u16)
    };
    let (Some(x0), Some(x1), Some(y0), Some(y1)) = (snap(x0), snap(x1), snap(y0), snap(y1)) else {
        return SubpathRect::NotARect;
    };

    let rect = RectU16::new(x0, y0, x1, y1);
    if rect.is_empty() {
        SubpathRect::Empty
    } else {
        SubpathRect::Rect(rect)
    }
}

#[inline]
fn point_eq(a: Point, b: Point) -> bool {
    (a.x - b.x).abs() <= INT_EDGE_EPSILON && (a.y - b.y).abs() <= INT_EDGE_EPSILON
}

#[derive(Debug)]
struct ClipData {
    alpha_start: u32,
    strip_start: u32,

    /// A coarse bounding box of the clip path in pixel coordinates.
    ///
    /// These bounds have already been intersected with the viewport.
    bbox: RectU16,

    /// When the whole clip stack up to and including this entry is equivalent to a set of
    /// pairwise-disjoint integer rectangles, that effective set (clamped to the viewport).
    /// `None` means "not representable", never "empty".
    int_rects: Option<IntRectSet>,
}

impl ClipData {
    fn to_path_data_ref<'a>(&self, storage: &'a StripStorage) -> PathDataRef<'a> {
        PathDataRef {
            strips: storage
                .strips
                .get(self.strip_start as usize..)
                .unwrap_or(&[]),
            alphas: storage
                .alphas
                .get(self.alpha_start as usize..)
                .unwrap_or(&[]),
            bbox: self.bbox,
        }
    }
}

/// A context for managing clip stacks.
#[derive(Debug)]
pub struct ClipContext {
    storage: StripStorage,
    temp_storage: StripStorage,
    clip_stack: Vec<ClipData>,
    /// Reusable capture buffer for integer-rectangle clip detection; see
    /// [`Self::push_clip`].
    detect_buf: Vec<PathEl>,
}

impl Default for ClipContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipContext {
    /// Create a new clip context.
    #[inline]
    pub fn new() -> Self {
        let mut main_storage = StripStorage::default();
        main_storage.set_generation_mode(GenerationMode::Append);
        Self {
            storage: main_storage,
            temp_storage: StripStorage::default(),
            clip_stack: vec![],
            detect_buf: vec![],
        }
    }

    /// Reset the clip context.
    #[inline]
    pub fn reset(&mut self) {
        self.clip_stack.clear();
        self.storage.clear();
        self.temp_storage.clear();
    }

    /// Get the data of the current clip path.
    #[inline]
    pub fn get(&self) -> Option<PathDataRef<'_>> {
        self.clip_stack
            .last()
            .map(|c| c.to_path_data_ref(&self.storage))
    }

    /// Whether the current clip stack (possibly empty) is exactly representable as a set of
    /// pairwise-disjoint integer rectangles.
    #[inline]
    pub fn is_int_rect_clip(&self) -> bool {
        self.clip_stack.last().is_none_or(|c| c.int_rects.is_some())
    }

    /// The effective clip region as a set of pairwise-disjoint integer rectangles, when every
    /// clip on the stack is such a set. `None` when the stack is empty or the region is not
    /// representable.
    #[inline]
    pub fn effective_int_rect_set(&self) -> Option<IntRectSet> {
        self.clip_stack.last().and_then(|c| c.int_rects)
    }

    /// Push a new clip path to the stack.
    #[inline]
    pub fn push_clip(
        &mut self,
        clip_path: impl IntoIterator<Item = PathEl>,
        strip_generator: &mut StripGenerator,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.temp_storage.clear();

        let alpha_start = self.storage.alphas.len() as u32;
        let strip_start = self.storage.strips.len() as u32;

        let existing_clip = self
            .clip_stack
            .last()
            .map(|c| c.to_path_data_ref(&self.storage));

        // Capture a bounded prefix of the path while it streams into generation, for the
        // integer-rectangle detection below (the path is only iterable once).
        self.detect_buf.clear();
        let detect_buf = &mut self.detect_buf;
        let clip_path = clip_path.into_iter().inspect(|el| {
            if detect_buf.len() <= MAX_DETECT_ELEMENTS {
                detect_buf.push(*el);
            }
        });

        strip_generator.generate_filled_path(
            clip_path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.temp_storage,
            existing_clip,
        );

        let bbox = strip_bbox(&self.temp_storage.strips).unwrap_or(RectU16::ZERO);

        // Track whether the stack remains an exact union of disjoint integer rectangles.
        let int_rects = if self.detect_buf.len() > MAX_DETECT_ELEMENTS {
            None
        } else {
            let viewport = RectU16::new(0, 0, strip_generator.width(), strip_generator.height());
            let own =
                |buf: &[PathEl]| path_as_integer_rect_set(buf.iter().copied(), transform, viewport);
            match self.clip_stack.last() {
                Some(parent) => parent
                    .int_rects
                    .and_then(|parent_set| parent_set.intersect(&own(&self.detect_buf)?)),
                None => own(&self.detect_buf),
            }
        };

        let clip_data = ClipData {
            alpha_start,
            strip_start,
            bbox,
            int_rects,
        };

        self.storage.extend(&self.temp_storage);
        self.clip_stack.push(clip_data);
    }

    /// Pop the least recent clip path.
    #[inline]
    pub fn pop_clip(&mut self) {
        let data = self.clip_stack.pop().expect("clip stack underflowed");
        self.storage.strips.truncate(data.strip_start as usize);
        self.storage.alphas.truncate(data.alpha_start as usize);
    }
}

/// Raw data of a previously pushed clip path.
#[derive(Debug)]
struct RawClip {
    /// The range of commands in [`ClipState::path_elements`] belonging to this clip path.
    path: Range<usize>,
    fill_rule: Fill,
    transform: Affine,
    aliasing_threshold: Option<u8>,
}

/// A frame containing clipping-relevant state for the root layer or a filter layer.
#[derive(Debug)]
struct ClipFrame {
    /// The clip context of the parent.
    parent_context: ClipContext,
    /// The accumulated source shift of the current layer.
    source_shift: Affine,
    /// The current revision of the clipping state.
    clip_revision: u64,
}

// This struct implements an additional piece of logic to make non-isolated clips work properly
// with filter layers. The root of all "evil" that requires us to implement this wrapper around
// [`crate::clip::ClipContext`] is that, as the user pushes new filter layers into the
// render context, we eagerly apply a shift to all subsequently rendered contents to ensure that
// everything necessary for correct filter rendering is guaranteed to be visible. However, since
// the clip stack eagerly generates strips for each clip path that is clipped to the original
// viewport, those generated clip paths cannot just be translated on demand to account for the
// source shift of the filter layer. Therefore, every time a new filter layer is pushed, we need
// to regenerate the clip context for that specific layer to ensure clips are applied correctly.
/// State for managing clip paths across multiple viewports.
#[derive(Debug)]
pub struct ClipState {
    /// The currently active clip context.
    context: ClipContext,
    /// A pool of reusable clip contexts.
    context_pool: Pool<ClipContext>,
    /// A flat factor of path elements storing the original path data of clip paths.
    path_elements: Vec<PathEl>,
    /// Raw data of the currently active stack of clip paths
    raw_clips: Vec<RawClip>,
    /// Stack of pushed clip frames.
    frames: Vec<ClipFrame>,
    /// The current revision.
    revision: u64,
}

impl Clear for ClipContext {
    fn clear(&mut self) {
        self.reset();
    }
}

impl Default for ClipState {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipState {
    /// Create a new clip state.
    pub fn new() -> Self {
        Self {
            context: ClipContext::new(),
            context_pool: Pool::default(),
            path_elements: Vec::new(),
            raw_clips: Vec::new(),
            frames: Vec::new(),
            revision: 0,
        }
    }

    /// Return the current clip path.
    pub fn get(&self) -> Option<PathDataRef<'_>> {
        self.context.get()
    }

    /// Push a new root viewport.
    pub fn push_root_viewport(
        &mut self,
        source_shift: (u16, u16),
        strip_generator: &mut StripGenerator,
    ) {
        let parent_context = core::mem::replace(&mut self.context, self.context_pool.take());
        let source_shift =
            Affine::translate((f64::from(source_shift.0), f64::from(source_shift.1)))
                * self.active_shift();
        self.frames.push(ClipFrame {
            parent_context,
            source_shift,
            clip_revision: self.revision,
        });
        self.rebuild_context(strip_generator);
    }

    /// Pop the last root viewport.
    pub fn pop_root_viewport(&mut self, strip_generator: &mut StripGenerator) {
        let frame = self.frames.pop().expect("filter clip stack underflow");
        let filter_context = core::mem::replace(&mut self.context, frame.parent_context);
        self.context_pool.submit(filter_context);
        if self.revision == frame.clip_revision {
            // No new clip paths have been pushed or popped since then, so we don't have to rebuild it.
        } else {
            self.rebuild_context(strip_generator);
        }
    }

    /// Push a clip path.
    pub fn push_clip(
        &mut self,
        path: &BezPath,
        strip_generator: &mut StripGenerator,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        let path_start = self.path_elements.len();
        self.path_elements.extend(path.iter());
        let path = path_start..self.path_elements.len();
        let clip_transform = self.active_shift() * transform;

        self.context.push_clip(
            self.path_elements[path.clone()].iter().copied(),
            strip_generator,
            fill_rule,
            clip_transform,
            aliasing_threshold,
        );
        self.raw_clips.push(RawClip {
            path,
            fill_rule,
            transform,
            aliasing_threshold,
        });
        self.revision = self.revision.wrapping_add(1);
    }

    /// Pop the active clip path.
    pub fn pop_clip(&mut self) {
        let raw_clip = self.raw_clips.pop().expect("clip stack underflowed");
        self.path_elements.truncate(raw_clip.path.start);
        self.context.pop_clip();
        self.revision = self.revision.wrapping_add(1);
    }

    /// Reset the clip state.
    pub fn reset(&mut self) {
        self.context.reset();
        for frame in self.frames.drain(..) {
            self.context_pool.submit(frame.parent_context);
        }
        self.path_elements.clear();
        self.raw_clips.clear();
        self.revision = 0;
    }

    fn active_shift(&self) -> Affine {
        self.frames
            .last()
            .map_or(Affine::IDENTITY, |frame| frame.source_shift)
    }

    fn rebuild_context(&mut self, strip_generator: &mut StripGenerator) {
        self.context.reset();
        let active_shift = self.active_shift();
        for raw_clip in &self.raw_clips {
            self.context.push_clip(
                self.path_elements[raw_clip.path.clone()].iter().copied(),
                strip_generator,
                raw_clip.fill_rule,
                active_shift * raw_clip.transform,
                raw_clip.aliasing_threshold,
            );
        }
    }
}

/// Borrowed data of a stripped path.
#[derive(Clone, Copy, Debug)]
pub struct PathDataRef<'a> {
    /// The strips.
    pub strips: &'a [Strip],
    /// The alpha buffer.
    pub alphas: &'a [u8],
    /// A tile-aligned coarse bounding box of the clip path in pixel coordinates.
    ///
    /// These bounds have already been intersected with the viewport.
    pub bbox: RectU16,
}

/// Compute the sparse strips representation of a path that results
/// from intersecting the two input paths. This can be used to implement
/// clip paths.
pub fn intersect(
    level: Level,
    path_1: PathDataRef<'_>,
    path_2: PathDataRef<'_>,
    target: &mut StripStorage,
) {
    dispatch!(level, simd => intersect_impl(simd, path_1, path_2, target));
}

/// The implementation of the clipping algorithm using sparse strips. Conceptually, it is relatively
/// simple: We iterate over each strip and fill region of the two paths in lock step and determine
/// all overlaps between the two. For each overlap, we proceed depending on what kind of region
/// we have in the first path and the second one.
/// - In case we have two fill regions, the overlap region will also be filled.
/// - In case we have one strip and one fill region, the overlap region will copy the alpha mask of the strip region.
/// - Finally, if we have two strip regions, we combine the alpha masks of both.
/// - All regions that are not filled in either path are simply ignored.
///
/// This is all that this method does. It just looks more complicated as the logic for iterating
/// in lock step is a bit tricky.
#[inline(always)]
fn intersect_impl<S: Simd>(
    simd: S,
    path_1: PathDataRef<'_>,
    path_2: PathDataRef<'_>,
    target: &mut StripStorage,
) {
    // In case either path is empty, the clip path should be empty.
    if path_1.strips.is_empty() || path_2.strips.is_empty() {
        return;
    }

    // Ignore any y values that are outside the bounding box of either of the two paths, as
    // those are guaranteed to have neither fill nor strip regions.
    let path_1_start_y = path_1.strips[0].strip_y();
    let path_2_start_y = path_2.strips[0].strip_y();
    let mut cur_y = path_1_start_y.max(path_2_start_y);
    let end_y = path_1.strips[path_1.strips.len() - 1]
        .strip_y()
        .min(path_2.strips[path_2.strips.len() - 1].strip_y());

    let mut path_1_idx = 0;
    let mut path_2_idx = 0;

    // Use binary search to determine the first index of whichever
    // path has a smaller y to avoid a large linear scan in the
    // first iteration of the loop below in case the discrepancy
    // is large.
    if path_1_start_y < cur_y {
        path_1_idx = first_strip_at_or_after(path_1.strips, cur_y);
    } else if path_2_start_y < cur_y {
        path_2_idx = first_strip_at_or_after(path_2.strips, cur_y);
    }

    let mut strip_state = None;

    // Iterate over each strip row and handle them.
    while cur_y <= end_y {
        // For each row, we create two iterators that alternatingly yield the strips and fill
        // regions in that row, until the last strip has been reached.
        let mut p1_iter = RowIterator::new(path_1, &mut path_1_idx, cur_y);
        let mut p2_iter = RowIterator::new(path_2, &mut path_2_idx, cur_y);

        let mut p1_region = p1_iter.next();
        let mut p2_region = p2_iter.next();

        // If at least one region is none, it means that we reached the end of the row
        // for that path, meaning that we exceeded the bounding box of that path and no
        // additional strips should be generated for that row, even if the other path might
        // still have more strips left. They will all be clipped away. So only consider it
        // if both paths have a region left.
        while let (Some(region_1), Some(region_2)) = (p1_region, p2_region) {
            match region_1.overlap_relationship(&region_2) {
                // This means there is no overlap between the regions, so we need to advance
                // the iterator of the region that is further behind.
                OverlapRelationship::Advance(advance) => {
                    match advance {
                        Advance::Left => p1_region = p1_iter.next(),
                        Advance::Right => p2_region = p2_iter.next(),
                    };

                    continue;
                }
                // We have an overlap!
                OverlapRelationship::Overlap(overlap) => {
                    match (region_1, region_2) {
                        // Both regions are a fill. Flush the current strip and start a new
                        // one at the end of the overlap region setting `fill_gap` to true,
                        // so that the whole area before that will be filled with a sparse
                        // fill.
                        (Region::Fill(_), Region::Fill(_)) => {
                            flush_strip(&mut strip_state, &mut target.strips, cur_y);
                            start_strip(&mut strip_state, &target.alphas, overlap.end, true);
                        }
                        // One fill one strip, so we simply use the alpha mask from the strip region.
                        (Region::Strip(s), Region::Fill(_))
                        | (Region::Fill(_), Region::Strip(s)) => {
                            // If possible, don't create a new strip but just extend the current one.
                            if should_create_new_strip(&strip_state, &target.alphas, overlap.start)
                            {
                                flush_strip(&mut strip_state, &mut target.strips, cur_y);
                                start_strip(&mut strip_state, &target.alphas, overlap.start, false);
                            }

                            let s_alphas = &s.alphas[(overlap.start - s.start) as usize * 4..]
                                [..overlap.width() as usize * 4];
                            target.alphas.extend_from_slice(s_alphas);
                        }
                        // Two strips, we need to multiply the opacity masks from both paths.
                        (Region::Strip(s_region_1), Region::Strip(s_region_2)) => {
                            // Once again, only create a new strip if we can't extend the current one.
                            if should_create_new_strip(&strip_state, &target.alphas, overlap.start)
                            {
                                flush_strip(&mut strip_state, &mut target.strips, cur_y);
                                start_strip(&mut strip_state, &target.alphas, overlap.start, false);
                            }

                            let num_blocks = overlap.width() / Tile::HEIGHT;

                            // Get the right alpha values for the specific position.
                            let s1_alphas = s_region_1.alphas
                                [(overlap.start - s_region_1.start) as usize * 4..]
                                .chunks_exact(16)
                                .take(num_blocks as usize);
                            let s2_alphas = s_region_2.alphas
                                [(overlap.start - s_region_2.start) as usize * 4..]
                                .chunks_exact(16)
                                .take(num_blocks as usize);

                            for (s1_alpha, s2_alpha) in s1_alphas.zip(s2_alphas) {
                                let s1 = u8x16::from_slice(simd, s1_alpha);
                                let s2 = u8x16::from_slice(simd, s2_alpha);

                                // Combine them.
                                let res = simd.narrow_u16x16(normalized_mul_u8x16(s1, s2));
                                target.alphas.extend(res.as_slice());
                            }
                        }
                    }

                    // Advance the iterator of the path whose region's end is further behind.
                    match overlap.advance {
                        Advance::Left => p1_region = p1_iter.next(),
                        Advance::Right => p2_region = p2_iter.next(),
                    };
                }
            }
        }

        // Flush the strip before advancing to the next strip row.
        flush_strip(&mut strip_state, &mut target.strips, cur_y);
        cur_y += 1;
    }

    // Push the sentinel strip if the intersection is not empty.
    if !target.strips.is_empty() {
        target.strips.push(Strip::sentinel(
            end_y * Tile::HEIGHT,
            target.alphas.len() as u32,
        ));
    }
}

#[inline(always)]
fn first_strip_at_or_after(strips: &[Strip], strip_y: u16) -> usize {
    // Strips are guaranteed to be sorted in ascending y (and ascending x),
    // hence why we can do this.
    strips.partition_point(|strip| strip.strip_y() < strip_y)
}

/// An overlap between two regions.
struct Overlap {
    /// The start x coordinate.
    start: u16,
    /// The end x coordinate.
    end: u16,
    /// Whether the left or right region iterator should be advanced next.
    advance: Advance,
}

impl Overlap {
    fn width(&self) -> u16 {
        self.end - self.start
    }
}

enum Advance {
    Left,
    Right,
}

/// The relationship between two regions.
enum OverlapRelationship {
    /// There is no overlap between the regions, advance the region iterator on the given side.
    Advance(Advance),
    /// There is an overlap between the regions.
    Overlap(Overlap),
}

#[derive(Debug, Clone, Copy)]
struct FillRegion {
    start: u16,
    width: u16,
}

#[derive(Debug, Clone, Copy)]
struct StripRegion<'a> {
    start: u16,
    width: u16,
    alphas: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
enum Region<'a> {
    Fill(FillRegion),
    Strip(StripRegion<'a>),
}

impl Region<'_> {
    #[inline(always)]
    fn start(&self) -> u16 {
        match self {
            Region::Fill(fill) => fill.start,
            Region::Strip(strip) => strip.start,
        }
    }

    #[inline(always)]
    fn width(&self) -> u16 {
        match self {
            Region::Fill(fill) => fill.width,
            Region::Strip(strip) => strip.width,
        }
    }

    #[inline(always)]
    fn end(&self) -> u16 {
        self.start() + self.width()
    }

    fn overlap_relationship(&self, other: &Region<'_>) -> OverlapRelationship {
        if self.end() <= other.start() {
            OverlapRelationship::Advance(Advance::Left)
        } else if self.start() >= other.end() {
            OverlapRelationship::Advance(Advance::Right)
        } else {
            let start = self.start().max(other.start());
            let end = self.end().min(other.end());

            let shift = if self.end() <= other.end() {
                Advance::Left
            } else {
                Advance::Right
            };

            OverlapRelationship::Overlap(Overlap {
                advance: shift,
                start,
                end,
            })
        }
    }
}

/// An iterator of strip and fill regions of a single strip row.
struct RowIterator<'a> {
    /// The path in question.
    input: PathDataRef<'a>,
    /// The strip row we want to iterate over.
    strip_y: u16,
    /// The index of the current strip.
    cur_idx: &'a mut usize,
    /// Whether the iterator should yield a strip next or not.
    /// When iterating over a row, we alternate between emitting strips and filled regions (unless
    /// the region between two strips is not filled), so this flag acts as a toggle to store what
    /// should be yielded next.
    on_strip: bool,
}

impl<'a> RowIterator<'a> {
    fn new(input: PathDataRef<'a>, cur_idx: &'a mut usize, strip_y: u16) -> Self {
        // Forward the index until we have found the right strip.
        while input.strips[*cur_idx].strip_y() < strip_y {
            *cur_idx += 1;
        }

        Self {
            input,
            cur_idx,
            strip_y,
            on_strip: true,
        }
    }

    #[inline(always)]
    fn cur_strip(&self) -> &Strip {
        &self.input.strips[*self.cur_idx]
    }

    #[inline(always)]
    fn next_strip(&self) -> &Strip {
        &self.input.strips[*self.cur_idx + 1]
    }

    #[inline(always)]
    fn cur_strip_width(&self) -> u16 {
        let cur = self.cur_strip();
        let next = self.next_strip();
        ((next.alpha_idx() - cur.alpha_idx()) / Tile::HEIGHT as u32) as u16
    }

    #[inline(always)]
    fn cur_strip_alphas(&self) -> &'a [u8] {
        let cur = self.cur_strip();
        let next = self.next_strip();
        &self.input.alphas[cur.alpha_idx() as usize..next.alpha_idx() as usize]
    }

    fn cur_strip_fill_area(&self) -> Option<FillRegion> {
        let next = self.next_strip();

        // Note that if the next strip happens to be on the next line, it will always have
        // zero winding so we don't need to special case this.
        if next.fill_gap() {
            let cur = self.cur_strip();
            let x = cur.x + self.cur_strip_width();
            let width = next.x - x;

            (width > 0).then_some(FillRegion { start: x, width })
        } else {
            None
        }
    }
}

impl<'a> Iterator for RowIterator<'a> {
    type Item = Region<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we are currently not on a strip, we want to yield a filled region in case there is one.
            if !self.on_strip {
                // Flip boolean flag so we will yield a strip in the next iteration.
                self.on_strip = true;

                // if we have a filled area, yield it and return. Otherwise, do nothing and we will
                // instead yield the next strip below. In any case, we need to advance the current index
                // so that we point to the next strip now.
                if let Some(fill_area) = self.cur_strip_fill_area() {
                    *self.cur_idx += 1;

                    return Some(Region::Fill(fill_area));
                } else {
                    *self.cur_idx += 1;
                }
            }

            // If we reached this point, we will yield a strip this iteration, so toggle the flag
            // so that in the next iteration, we yield a filled region instead.
            self.on_strip = false;

            // If the current strip is sentinel or not within our target row, terminate.
            if self.cur_strip().is_sentinel() || self.cur_strip().strip_y() != self.strip_y {
                return None;
            }

            // Calculate the dimensions of the strip and yield it.
            let x = self.cur_strip().x;
            let width = self.cur_strip_width();

            // Zero-width strips only act as markers for cheaply delimiting the width
            // of filled regions, but are not actually relevant for clipping. This is assuming that
            // zero-width strips can only appear at the end of a row, see the comment in
            // `Strip::emit_culled_background`.
            if width == 0 {
                debug_assert!(
                    self.next_strip().is_sentinel() || self.next_strip().strip_y() != self.strip_y,
                    "zero-width strips must only appear at the end of a row"
                );

                continue;
            }

            let alphas = self.cur_strip_alphas();

            return Some(Region::Strip(StripRegion {
                start: x,
                width,
                alphas,
            }));
        }
    }
}

/// The data of the current strip we are building.
struct StripState {
    x: u16,
    alpha_idx: u32,
    fill_gap: bool,
}

fn flush_strip(strip_state: &mut Option<StripState>, strips: &mut Vec<Strip>, cur_y: u16) {
    if let Some(state) = core::mem::take(strip_state) {
        strips.push(Strip::new(
            state.x,
            cur_y * Tile::HEIGHT,
            state.alpha_idx,
            state.fill_gap,
        ));
    }
}

#[inline(always)]
fn start_strip(strip_data: &mut Option<StripState>, alphas: &[u8], x: u16, fill_gap: bool) {
    *strip_data = Some(StripState {
        x,
        alpha_idx: alphas.len() as u32,
        fill_gap,
    });
}

fn should_create_new_strip(
    strip_state: &Option<StripState>,
    alphas: &[u8],
    overlap_start: u16,
) -> bool {
    // Returns false in case we can append to the currently built strip.
    strip_state.as_ref().is_none_or(|state| {
        let width = ((alphas.len() as u32 - state.alpha_idx) / Tile::HEIGHT as u32) as u16;
        let strip_end = state.x + width;

        strip_end < overlap_start - 1
    })
}

#[cfg(test)]
mod tests {
    use crate::clip::{
        ClipContext, PathDataRef, Region, RowIterator, first_strip_at_or_after, intersect,
        path_as_integer_rect_set,
    };
    use crate::geometry::RectU16;
    use crate::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape};
    use crate::strip::Strip;
    use crate::strip_generator::{StripGenerator, StripStorage};
    use crate::tile::Tile;
    use fearless_simd::Level;
    use peniko::Fill;
    use std::vec;

    const VIEWPORT: RectU16 = RectU16::new(0, 0, 200, 100);

    fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        Rect::new(x0, y0, x1, y1).to_path(0.1)
    }

    fn detect(path: &BezPath, transform: Affine) -> Option<vec::Vec<RectU16>> {
        path_as_integer_rect_set(path.iter(), transform, VIEWPORT)
            .map(|set| set.as_slice().to_vec())
    }

    #[test]
    fn detect_integer_rect() {
        let detected = detect(&rect_path(10.0, 20.0, 50.0, 40.0), Affine::IDENTITY);
        assert_eq!(detected, Some(vec![RectU16::new(10, 20, 50, 40)]));
    }

    #[test]
    fn detect_integer_rect_reverse_winding() {
        let mut p = BezPath::new();
        p.move_to((10.0, 20.0));
        p.line_to((10.0, 40.0));
        p.line_to((50.0, 40.0));
        p.line_to((50.0, 20.0));
        p.close_path();
        assert_eq!(
            detect(&p, Affine::IDENTITY),
            Some(vec![RectU16::new(10, 20, 50, 40)])
        );
    }

    #[test]
    fn detect_open_rect_without_close() {
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.line_to((8.0, 0.0));
        p.line_to((8.0, 8.0));
        p.line_to((0.0, 8.0));
        assert_eq!(
            detect(&p, Affine::IDENTITY),
            Some(vec![RectU16::new(0, 0, 8, 8)])
        );
    }

    #[test]
    fn detect_rejects_fractional_rect() {
        assert_eq!(
            detect(&rect_path(10.5, 20.0, 50.0, 40.0), Affine::IDENTITY),
            None
        );
    }

    #[test]
    fn detect_snaps_near_integer_edges() {
        let detected = detect(
            &rect_path(10.0 + 1.0 / 1024.0, 20.0, 50.0 - 1.0 / 1024.0, 40.0),
            Affine::IDENTITY,
        );
        assert_eq!(detected, Some(vec![RectU16::new(10, 20, 50, 40)]));
        assert_eq!(
            detect(&rect_path(10.01, 20.0, 50.0, 40.0), Affine::IDENTITY),
            None
        );
    }

    #[test]
    fn detect_applies_transform() {
        let detected = detect(&rect_path(5.0, 10.0, 25.0, 20.0), Affine::scale(2.0));
        assert_eq!(detected, Some(vec![RectU16::new(10, 20, 50, 40)]));
        // 90° rotation maps an axis-aligned rect back to an axis-aligned rect.
        let rotated =
            Affine::rotate(core::f64::consts::FRAC_PI_2) * Affine::translate((0.0, -60.0));
        let detected = detect(&rect_path(10.0, 20.0, 30.0, 50.0), rotated);
        assert_eq!(detected, Some(vec![RectU16::new(10, 10, 40, 30)]));
        // A non-integer scale of integer coordinates is fractional in device space.
        assert_eq!(
            detect(&rect_path(5.0, 10.0, 25.0, 20.0), Affine::scale(1.25)),
            None
        );
    }

    #[test]
    fn detect_rejects_non_rects() {
        let mut triangle = BezPath::new();
        triangle.move_to((0.0, 0.0));
        triangle.line_to((10.0, 0.0));
        triangle.line_to((0.0, 10.0));
        triangle.close_path();
        assert_eq!(detect(&triangle, Affine::IDENTITY), None);

        // Corners span a rectangle but the edges are diagonal.
        let mut z_quad = BezPath::new();
        z_quad.move_to((0.0, 0.0));
        z_quad.line_to((10.0, 10.0));
        z_quad.line_to((0.0, 10.0));
        z_quad.line_to((10.0, 0.0));
        z_quad.close_path();
        assert_eq!(detect(&z_quad, Affine::IDENTITY), None);

        let circle = Circle::new(Point::new(20.0, 20.0), 10.0).to_path(0.1);
        assert_eq!(detect(&circle, Affine::IDENTITY), None);
    }

    #[test]
    fn detect_clamps_to_viewport() {
        // Out-of-viewport edges land on the integer viewport bounds, so they may be fractional.
        let detected = detect(&rect_path(-5.3, -2.7, 50.0, 300.9), Affine::IDENTITY);
        assert_eq!(detected, Some(vec![RectU16::new(0, 0, 50, 100)]));
        // Fully outside: a valid, empty set (everything is clipped away).
        let detected = detect(&rect_path(300.0, 0.0, 400.0, 50.0), Affine::IDENTITY);
        assert_eq!(detected, Some(vec![]));
    }

    #[test]
    fn detect_multi_rect_set() {
        let mut p = rect_path(0.0, 0.0, 40.0, 100.0);
        p.extend(rect_path(60.0, 0.0, 100.0, 100.0).iter());
        assert_eq!(
            detect(&p, Affine::IDENTITY),
            Some(vec![
                RectU16::new(0, 0, 40, 100),
                RectU16::new(60, 0, 100, 100)
            ])
        );
    }

    #[test]
    fn detect_rejects_overlapping_rects() {
        let mut p = rect_path(0.0, 0.0, 50.0, 50.0);
        p.extend(rect_path(40.0, 40.0, 90.0, 90.0).iter());
        assert_eq!(detect(&p, Affine::IDENTITY), None);
    }

    #[test]
    fn detect_rejects_too_many_rects() {
        let mut p = BezPath::new();
        for i in 0..17_usize {
            let x = (i * 10) as f64;
            p.extend(rect_path(x, 0.0, x + 5.0, 5.0).iter());
        }
        assert_eq!(detect(&p, Affine::IDENTITY), None);
    }

    #[test]
    fn clip_stack_tracks_int_rects() {
        let mut ctx = ClipContext::new();
        let mut generator = StripGenerator::new(200, 100, Level::baseline());
        let push = |ctx: &mut ClipContext, generator: &mut StripGenerator, path: &BezPath| {
            ctx.push_clip(
                path.iter(),
                generator,
                Fill::NonZero,
                Affine::IDENTITY,
                None,
            );
        };
        assert!(ctx.is_int_rect_clip());
        assert!(ctx.effective_int_rect_set().is_none());

        push(&mut ctx, &mut generator, &rect_path(10.0, 10.0, 90.0, 90.0));
        assert!(ctx.is_int_rect_clip());
        assert_eq!(
            ctx.effective_int_rect_set().unwrap().as_slice(),
            &[RectU16::new(10, 10, 90, 90)]
        );

        // Nested integer rect: effective set is the intersection.
        push(&mut ctx, &mut generator, &rect_path(50.0, 0.0, 120.0, 60.0));
        assert_eq!(
            ctx.effective_int_rect_set().unwrap().as_slice(),
            &[RectU16::new(50, 10, 90, 60)]
        );

        // A non-rect clip poisons the stack until it is popped.
        let mut triangle = BezPath::new();
        triangle.move_to((0.0, 0.0));
        triangle.line_to((100.0, 0.0));
        triangle.line_to((0.0, 100.0));
        triangle.close_path();
        push(&mut ctx, &mut generator, &triangle);
        assert!(!ctx.is_int_rect_clip());
        assert!(ctx.effective_int_rect_set().is_none());

        ctx.pop_clip();
        assert_eq!(
            ctx.effective_int_rect_set().unwrap().as_slice(),
            &[RectU16::new(50, 10, 90, 60)]
        );
        ctx.pop_clip();
        ctx.pop_clip();
        assert!(ctx.is_int_rect_clip());
        assert!(ctx.effective_int_rect_set().is_none());
    }

    #[test]
    fn int_rect_below_non_rect_stays_unrepresentable() {
        let mut ctx = ClipContext::new();
        let mut generator = StripGenerator::new(200, 100, Level::baseline());
        let mut triangle = BezPath::new();
        triangle.move_to((0.0, 0.0));
        triangle.line_to((100.0, 0.0));
        triangle.line_to((0.0, 100.0));
        triangle.close_path();
        ctx.push_clip(
            triangle.iter(),
            &mut generator,
            Fill::NonZero,
            Affine::IDENTITY,
            None,
        );
        ctx.push_clip(
            rect_path(10.0, 10.0, 50.0, 50.0).iter(),
            &mut generator,
            Fill::NonZero,
            Affine::IDENTITY,
            None,
        );
        assert!(!ctx.is_int_rect_clip());
        assert!(ctx.effective_int_rect_set().is_none());
    }

    #[test]
    fn intersect_partly_overlapping_strips() {
        let path_1 = StripBuilder::new().add_strip(0, 0, 32, false).finish();

        let path_2 = StripBuilder::new().add_strip(8, 0, 44, false).finish();

        let expected = StripBuilder::new().add_strip(8, 0, 32, false).finish();

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn intersect_multiple_overlapping_strips() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 1, 4, false)
            .add_strip(12, 1, 20, true)
            .add_strip(28, 1, 32, false)
            .add_strip(44, 1, 52, true)
            .finish();

        let path_2 = StripBuilder::new()
            .add_strip(4, 1, 8, false)
            .add_strip(16, 1, 20, true)
            .add_strip(24, 1, 28, false)
            .add_strip(32, 1, 36, false)
            .add_strip(44, 1, 48, true)
            .finish();

        let expected = StripBuilder::new()
            .add_strip(4, 1, 8, false)
            .add_strip(12, 1, 20, true)
            .add_strip(32, 1, 36, false)
            .add_strip(44, 1, 48, true)
            .finish();

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn multiple_rows() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(16, 0, 20, true)
            .add_strip(4, 1, 8, false)
            .add_strip(12, 1, 24, true)
            .add_strip(4, 2, 8, false)
            .add_strip(16, 2, 32, true)
            .finish();

        let path_2 = StripBuilder::new()
            .add_strip(0, 2, 4, false)
            .add_strip(16, 2, 24, true)
            .add_strip(8, 3, 12, false)
            .add_strip(16, 3, 28, true)
            .finish();

        let expected = StripBuilder::new()
            .add_strip(4, 2, 8, false)
            .add_strip(16, 2, 24, true)
            .finish();

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn alpha_buffer_correct_width() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(0, 1, 12, false)
            .finish();

        let path_2 = StripBuilder::new()
            .add_strip(4, 0, 8, false)
            .add_strip(0, 1, 4, false)
            .add_strip(12, 1, 16, true)
            .finish();

        let expected = StripBuilder::new().add_strip(0, 1, 12, false).finish();

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn first_strip_at_or_after_returns_first_matching_strip_y() {
        let path = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(0, 2, 4, false)
            .add_strip(8, 2, 12, false)
            .add_strip(16, 2, 20, false)
            .add_strip(0, 4, 4, false)
            .add_strip(8, 4, 12, false)
            .add_strip(0, 6, 4, false)
            .finish();

        assert_eq!(first_strip_at_or_after(&path.strips, 0), 0);
        assert_eq!(first_strip_at_or_after(&path.strips, 2), 1);
        assert_eq!(first_strip_at_or_after(&path.strips, 3), 4);
        assert_eq!(first_strip_at_or_after(&path.strips, 4), 4);
        assert_eq!(first_strip_at_or_after(&path.strips, 5), 6);
        assert_eq!(first_strip_at_or_after(&path.strips, 6), 6);
        assert_eq!(first_strip_at_or_after(&path.strips, 7), path.strips.len());
    }

    #[test]
    fn row_iterator_abort_next_line() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(0, 1, 4, false)
            .finish();

        let path_ref = PathDataRef {
            strips: &path_1.strips,
            alphas: &path_1.alphas,
            bbox: RectU16::new(0, 0, u16::MAX, u16::MAX),
        };

        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    #[test]
    fn row_iterator_row_end_fill_gap() {
        let path = StripBuilder::new()
            .add_strip(0, 0, Tile::WIDTH, false)
            .finish_with_fill_gap_row_end(16);
        let path_ref = path_ref(&path);

        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert_strip_region(iter.next(), 0, Tile::WIDTH);
        assert_fill_region(iter.next(), Tile::WIDTH, 16 - Tile::WIDTH);
        assert!(iter.next().is_none());
    }

    #[test]
    fn intersect_strip_with_row_end_fill_gap() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, Tile::WIDTH, false)
            .finish_with_fill_gap_row_end(16);
        let path_2 = StripBuilder::new().add_strip(8, 0, 12, false).finish();
        let expected = StripBuilder::new().add_strip(8, 0, 12, false).finish();

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn intersect_two_row_end_fill_gaps() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, 8, false)
            .finish_with_fill_gap_row_end(16);
        let path_2 = StripBuilder::new()
            .add_strip(4, 0, 12, false)
            .finish_with_fill_gap_row_end(20);
        let expected = StripBuilder::new()
            .add_strip(4, 0, 12, false)
            .finish_with_fill_gap_row_end(16);

        run_test(expected, path_1, path_2);
    }

    #[test]
    fn row_iterator_fill_gap_stops_at_row_boundary() {
        let path = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_row_end(0, 16, true)
            .add_strip(0, 1, 4, false)
            .finish();

        let path_ref = path_ref(&path);
        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert_strip_region(iter.next(), 0, 4);
        assert_fill_region(iter.next(), 4, 12);
        assert!(iter.next().is_none());

        let mut iter = RowIterator::new(path_ref, &mut idx, 1);

        assert_strip_region(iter.next(), 0, Tile::WIDTH);
        assert!(iter.next().is_none());
    }

    #[test]
    fn row_iterator_adjacent_unmerged_strips_no_fill() {
        let path = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(4, 0, 8, false)
            .finish();
        let path_ref = path_ref(&path);

        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert_strip_region(iter.next(), 0, 4);
        assert_strip_region(iter.next(), 4, 4);
        assert!(iter.next().is_none());
    }

    #[test]
    fn row_iterator_adjacent_unmerged_strips_with_fill_gap() {
        let path = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(4, 0, 8, true)
            .finish();
        let path_ref = path_ref(&path);

        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert_strip_region(iter.next(), 0, 4);
        assert_strip_region(iter.next(), 4, 4);
        assert!(iter.next().is_none());
    }

    #[test]
    fn intersect_adjacent_unmerged_strips() {
        let path = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(4, 0, 8, true)
            .finish();
        let cover = StripBuilder::new().add_strip(0, 0, 8, false).finish();
        let expected = StripBuilder::new().add_strip(0, 0, 8, false).finish();

        run_test(expected, path, cover);
    }

    fn run_test(expected: StripStorage, path_1: StripStorage, path_2: StripStorage) {
        let mut write_target = StripStorage::default();

        let path_1 = path_ref(&path_1);
        let path_2 = path_ref(&path_2);

        intersect(Level::new(), path_1, path_2, &mut write_target);

        assert_eq!(write_target, expected);
    }

    fn path_ref(path: &StripStorage) -> PathDataRef<'_> {
        PathDataRef {
            strips: &path.strips,
            alphas: &path.alphas,
            bbox: RectU16::new(0, 0, u16::MAX, u16::MAX),
        }
    }

    fn assert_strip_region(region: Option<Region<'_>>, start: u16, width: u16) {
        match region {
            Some(Region::Strip(strip)) => {
                assert_eq!(strip.start, start);
                assert_eq!(strip.width, width);
                assert_eq!(strip.alphas.len(), (width * Tile::HEIGHT) as usize);
            }
            other => panic!("expected strip region, got {other:?}"),
        }
    }

    fn assert_fill_region(region: Option<Region<'_>>, start: u16, width: u16) {
        match region {
            Some(Region::Fill(fill)) => {
                assert_eq!(fill.start, start);
                assert_eq!(fill.width, width);
            }
            other => panic!("expected fill region, got {other:?}"),
        }
    }

    struct StripBuilder {
        storage: StripStorage,
    }

    impl StripBuilder {
        fn new() -> Self {
            Self {
                storage: StripStorage::default(),
            }
        }

        fn add_strip(self, x: u16, strip_y: u16, end: u16, fill_gap: bool) -> Self {
            let width = end - x;
            self.add_strip_with(
                x,
                strip_y,
                end,
                fill_gap,
                &vec![0; (width * Tile::HEIGHT) as usize],
            )
        }

        fn add_strip_with(
            mut self,
            x: u16,
            strip_y: u16,
            end: u16,
            fill_gap: bool,
            alphas: &[u8],
        ) -> Self {
            let width = end - x;
            assert_eq!(alphas.len(), (width * Tile::HEIGHT) as usize);
            let idx = self.storage.alphas.len();
            self.storage
                .strips
                .push(Strip::new(x, strip_y * Tile::HEIGHT, idx as u32, fill_gap));
            self.storage.alphas.extend_from_slice(alphas);

            self
        }

        fn finish(mut self) -> StripStorage {
            let last_y = self.storage.strips.last().unwrap().y;
            let idx = self.storage.alphas.len();

            self.storage
                .strips
                .push(Strip::sentinel(last_y, idx as u32));

            self.storage
        }

        fn add_row_end(mut self, strip_y: u16, x: u16, fill_gap: bool) -> Self {
            let idx = self.storage.alphas.len();
            self.storage
                .strips
                .push(Strip::new(x, strip_y * Tile::HEIGHT, idx as u32, fill_gap));

            self
        }

        fn finish_with_fill_gap_row_end(self, x: u16) -> StripStorage {
            let strip_y = self.storage.strips.last().unwrap().strip_y();

            self.add_row_end(strip_y, x, true).finish()
        }
    }
}
