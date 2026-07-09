// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Global operation buffers and ranged views used by scheduled passes.

use super::round::{BlendOp, FilterOp};
use crate::GpuStrip;
use crate::copy::GpuCopyInstance;
use crate::filter::FilterInstanceData;
use alloc::vec::Vec;
use core::ops::{Index, Range};
use core::slice::SliceIndex;
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
    pub(crate) strips: RangedBuffer<GpuStrip>,
    pub(crate) filter_ops: RangedBuffer<FilterOp>,
    pub(crate) filter_instances: RangedBuffer<FilterInstanceData>,
    pub(crate) filter_copies: RangedBuffer<GpuCopyInstance>,
    pub(crate) blends: RangedBuffer<BlendOp>,
}

impl ScheduleBuffers {
    pub(super) fn clear(&mut self) {
        self.strips.clear();
        self.filter_ops.clear();
        self.filter_instances.clear();
        self.filter_copies.clear();
        self.blends.clear();
    }
}

#[derive(Debug)]
pub(crate) struct RangedBuffer<T> {
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

    pub(super) fn extend_from_slice(&mut self, values: &[T]) -> Range<usize>
    where
        T: Copy,
    {
        let start = self.values.len();
        self.values.extend_from_slice(values);
        start..self.values.len()
    }

    pub(crate) fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T> {
        RangedSlice::new(&self.values, ranges)
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
