// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Global operation buffers and ranged views used by scheduled passes.

use super::round::{BlendOp, FilterOp};
use crate::GpuStrip;
use alloc::vec::Vec;
use core::ops::{Index, Range};
use core::slice::SliceIndex;
use vello_common::util::Clear;

#[derive(Debug, Default, Clone)]
pub(super) struct Ranges {
    ranges: Vec<Range<usize>>,
    len: usize,
}

impl Ranges {
    fn push(&mut self, range: Range<usize>) {
        self.len += range.len();
        if let Some(last) = self.ranges.last_mut()
            && last.end == range.start
        {
            last.end = range.end;
        } else {
            self.ranges.push(range);
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }
}

impl Clear for Ranges {
    fn clear(&mut self) {
        self.ranges.clear();
        self.len = 0;
    }
}

#[derive(Debug, Default)]
pub(super) struct ScheduleBuffers {
    pub(super) strips: RangedBuffer<GpuStrip>,
    pub(super) filters: RangedBuffer<FilterOp>,
    pub(super) blends: RangedBuffer<BlendOp>,
}

impl ScheduleBuffers {
    pub(super) fn clear(&mut self) {
        self.strips.clear();
        self.filters.clear();
        self.blends.clear();
    }
}

#[derive(Debug)]
pub(super) struct RangedBuffer<T> {
    values: Vec<T>,
}

impl<T> Default for RangedBuffer<T> {
    fn default() -> Self {
        Self { values: Vec::new() }
    }
}

impl<T> RangedBuffer<T> {
    pub(super) fn push(&mut self, ranges: &mut Ranges, value: T) {
        self.values.push(value);
        let end = self.values.len();
        let index = end - 1;
        ranges.push(index..end);
    }

    // TODO: See whether can be removed.
    pub(super) fn append(&mut self, values: &mut Vec<T>) -> Range<usize> {
        let start = self.values.len();
        self.values.append(values);
        start..self.values.len()
    }

    pub(super) fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T> {
        RangedSlice::new(&self.values, ranges)
    }

    pub(super) fn empty(&self) -> RangedSlice<'_, T> {
        RangedSlice::empty(&self.values)
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

impl<T, I> Index<I> for RangedBuffer<T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index]
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RangedSlice<'a, T> {
    buffer: &'a [T],
    ranges: &'a [Range<usize>],
    len: usize,
}

impl<'a, T> RangedSlice<'a, T> {
    fn new(buffer: &'a [T], ranges: &'a Ranges) -> Self {
        Self {
            buffer,
            ranges: &ranges.ranges,
            len: ranges.len,
        }
    }

    fn empty(buffer: &'a [T]) -> Self {
        Self {
            buffer,
            ranges: &[],
            len: 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn slices(&self) -> impl Iterator<Item = &'a [T]> + '_ {
        self.ranges.iter().map(|range| &self.buffer[range.clone()])
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        self.slices().flatten()
    }
}
