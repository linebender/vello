// Copyright © 2021 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those

//! An abstraction for writing to GPU buffers.

use bytemuck::Pod;

/// A GPU buffer to be filled.
pub struct BufWrite {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

impl BufWrite {
    pub(crate) fn new(ptr: *mut u8, len: usize, capacity: usize) -> BufWrite {
        BufWrite { ptr, len, capacity }
    }

    /// Append a plain data object to the buffer.
    ///
    /// Panics if capacity is inadequate.
    #[inline]
    pub fn push(&mut self, item: &impl Pod) {
        self.push_bytes(bytemuck::bytes_of(item));
    }

    /// Extend with a slice of plain data objects.
    ///
    /// Panics if capacity is inadequate.
    #[inline]
    pub fn extend_slice(&mut self, slice: &[impl Pod]) {
        self.push_bytes(bytemuck::cast_slice(slice));
    }

    /// Extend with a byte slice.
    ///
    /// Panics if capacity is inadequate.
    #[inline]
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        assert!(self.capacity - self.len >= len);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(self.len), len);
        }
        self.len += len;
    }

    /// Extend with zeros.
    ///
    /// Panics if capacity is inadequate.
    #[inline]
    pub fn fill_zero(&mut self, len: usize) {
        assert!(self.capacity - self.len >= len);
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.ptr.add(self.len), len);
            slice.fill(0);
        }
        self.len += len;
    }

    /// The total capacity of the buffer, in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Extend with an iterator over plain data objects.
    ///
    /// Currently, this doesn't panic, just truncates. That may change.
    // Note: when specialization lands, this can be another impl of
    // `Extend`.
    pub fn extend_ref_iter<'a, I, T: Pod + 'a>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        let item_size = std::mem::size_of::<T>();
        if item_size == 0 {
            return;
        }
        let mut iter = iter.into_iter();
        let n_remaining = (self.capacity - self.len) / item_size;
        unsafe {
            let mut dst = self.ptr.add(self.len);
            for _ in 0..n_remaining {
                if let Some(item) = iter.next() {
                    std::ptr::copy_nonoverlapping(
                        bytemuck::bytes_of(item).as_ptr(),
                        dst,
                        item_size,
                    );
                    self.len += item_size;
                    dst = dst.add(item_size);
                } else {
                    break;
                }
            }
        }
        // TODO: should we test the iter and panic on overflow?
    }
}

impl std::ops::Deref for BufWrite {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for BufWrite {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Pod> std::iter::Extend<T> for BufWrite {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        let item_size = std::mem::size_of::<T>();
        if item_size == 0 {
            return;
        }
        let mut iter = iter.into_iter();
        let n_remaining = (self.capacity - self.len) / item_size;
        unsafe {
            let mut dst = self.ptr.add(self.len);
            for _ in 0..n_remaining {
                if let Some(item) = iter.next() {
                    std::ptr::copy_nonoverlapping(
                        bytemuck::bytes_of(&item).as_ptr(),
                        dst,
                        item_size,
                    );
                    self.len += item_size;
                    dst = dst.add(item_size);
                } else {
                    break;
                }
            }
        }
        // TODO: should we test the iter and panic on overflow?
    }
}
