// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Global operation buffers and ranged views used by scheduled passes.

use super::round::{BlendOp, FilterOp};
use crate::GpuStrip;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::util::Clear;

#[derive(Debug, Default, Clone)]
pub(crate) struct Ranges {
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

    pub(crate) fn len(&self) -> usize {
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
pub(crate) struct ScheduleBuffers {
    pub(crate) strips: Vec<GpuStrip>,
    pub(crate) filter_ops: Vec<FilterOp>,
    pub(crate) blends: Vec<BlendOp>,
}

impl ScheduleBuffers {
    pub(super) fn clear(&mut self) {
        self.strips.clear();
        self.filter_ops.clear();
        self.blends.clear();
    }
}

pub(super) trait VecExt<T> {
    fn push_ranged(&mut self, ranges: &mut Ranges, value: T);

    fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T>;
}

impl<T> VecExt<T> for Vec<T> {
    fn push_ranged(&mut self, ranges: &mut Ranges, value: T) {
        self.push(value);
        let end = self.len();
        let index = end - 1;
        ranges.push(index..end);
    }

    fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T> {
        RangedSlice::new(self, ranges)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RangedSlice<'a, T> {
    buffer: &'a [T],
    ranges: &'a [Range<usize>],
    len: usize,
}

impl<'a, T> RangedSlice<'a, T> {
    pub(crate) const fn empty() -> Self {
        Self {
            buffer: &[],
            ranges: &[],
            len: 0,
        }
    }

    fn new(buffer: &'a [T], ranges: &'a Ranges) -> Self {
        Self {
            buffer,
            ranges: &ranges.ranges,
            len: ranges.len,
        }
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
