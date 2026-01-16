// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Managing clipping state.

use crate::kurbo::{Affine, BezPath};
use crate::strip::Strip;
use crate::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use crate::tile::Tile;
use crate::util::normalized_mul_u8x16;
use alloc::vec;
use alloc::vec::Vec;
use fearless_simd::{Level, Simd, SimdBase, dispatch, u8x16};
use peniko::Fill;

#[derive(Debug)]
struct ClipData {
    alpha_start: u32,
    strip_start: u32,
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
        }
    }
}

/// A context for managing clip stacks.
#[derive(Debug)]
pub struct ClipContext {
    storage: StripStorage,
    temp_storage: StripStorage,
    clip_stack: Vec<ClipData>,
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

    /// Push a new clip path to the stack.
    #[inline]
    pub fn push_clip(
        &mut self,
        clip_path: &BezPath,
        strip_generator: &mut StripGenerator,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.temp_storage.clear();

        let alpha_start = self.storage.alphas.len() as u32;
        let strip_start = self.storage.strips.len() as u32;

        let clip_data = ClipData {
            alpha_start,
            strip_start,
        };

        let existing_clip = self
            .clip_stack
            .last()
            .map(|c| c.to_path_data_ref(&self.storage));

        strip_generator.generate_filled_path(
            clip_path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.temp_storage,
            existing_clip,
        );

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

/// Borrowed data of a stripped path.
#[derive(Clone, Copy, Debug)]
pub struct PathDataRef<'a> {
    /// The strips.
    pub strips: &'a [Strip],
    /// The alpha buffer.
    pub alphas: &'a [u8],
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
    let mut cur_y = path_1.strips[0].strip_y().min(path_2.strips[0].strip_y());
    let end_y = path_1.strips[path_1.strips.len() - 1]
        .strip_y()
        .min(path_2.strips[path_2.strips.len() - 1].strip_y());

    let mut path_1_idx = 0;
    let mut path_2_idx = 0;
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

    // Push the sentinel strip.
    target.strips.push(Strip::new(
        u16::MAX,
        end_y * Tile::HEIGHT,
        target.alphas.len() as u32,
        false,
    ));
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

            Some(FillRegion { start: x, width })
        } else {
            None
        }
    }
}

impl<'a> Iterator for RowIterator<'a> {
    type Item = Region<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
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
        let alphas = self.cur_strip_alphas();

        Some(Region::Strip(StripRegion {
            start: x,
            width,
            alphas,
        }))
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
    use crate::clip::{PathDataRef, RowIterator, intersect};
    use crate::strip::Strip;
    use crate::strip_generator::StripStorage;
    use crate::tile::Tile;
    use fearless_simd::Level;
    use std::vec;

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
    fn row_iterator_abort_next_line() {
        let path_1 = StripBuilder::new()
            .add_strip(0, 0, 4, false)
            .add_strip(0, 1, 4, false)
            .finish();

        let path_ref = PathDataRef {
            strips: &path_1.strips,
            alphas: &path_1.alphas,
        };

        let mut idx = 0;
        let mut iter = RowIterator::new(path_ref, &mut idx, 0);

        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    fn run_test(expected: StripStorage, path_1: StripStorage, path_2: StripStorage) {
        let mut write_target = StripStorage::default();

        let path_1 = PathDataRef {
            strips: &path_1.strips,
            alphas: &path_1.alphas,
        };

        let path_2 = PathDataRef {
            strips: &path_2.strips,
            alphas: &path_2.alphas,
        };

        intersect(Level::new(), path_1, path_2, &mut write_target);

        assert_eq!(write_target, expected);
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
                .push(Strip::new(u16::MAX, last_y, idx as u32, false));

            self.storage
        }
    }
}
