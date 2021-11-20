// Copyright © 2021 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those

//! A simple best-fit allocator.

use std::collections::{BTreeMap, BTreeSet};

/// An allocator that tracks free ranges and returns best fit.
pub struct BestFit {
    // map offset to size of free block
    free_by_ix: BTreeMap<u32, u32>,
    // size and offset
    free_by_size: BTreeSet<(u32, u32)>,
}

impl BestFit {
    pub fn new(size: u32) -> BestFit {
        let mut free_by_ix = BTreeMap::new();
        free_by_ix.insert(0, size);
        let mut free_by_size = BTreeSet::new();
        free_by_size.insert((size, 0));
        BestFit {
            free_by_ix,
            free_by_size,
        }
    }

    pub fn alloc(&mut self, size: u32) -> Option<u32> {
        let block = *self.free_by_size.range((size, 0)..).next()?;
        let ix = block.1;
        self.free_by_ix.remove(&ix);
        self.free_by_size.remove(&block);
        let fragment_size = block.0 - size;
        if fragment_size > 0 {
            let fragment_ix = ix + size;
            self.free_by_ix.insert(fragment_ix, fragment_size);
            self.free_by_size.insert((fragment_size, fragment_ix));
        }
        Some(ix)
    }

    pub fn free(&mut self, ix: u32, size: u32) {
        let next_ix = size + ix;
        if let Some((&prev_ix, &prev_size)) = self.free_by_ix.range(..ix).rev().next() {
            if prev_ix + prev_size == ix {
                self.free_by_size.remove(&(prev_size, prev_ix));
                if let Some(&next_size) = self.free_by_ix.get(&next_ix) {
                    // consolidate with prev and next
                    let new_size = prev_size + size + next_size;
                    *self.free_by_ix.get_mut(&prev_ix).unwrap() = new_size;
                    self.free_by_ix.remove(&next_ix);
                    self.free_by_size.remove(&(next_size, next_ix));
                    self.free_by_size.insert((new_size, prev_ix));
                } else {
                    // consolidate with prev
                    let new_size = prev_size + size;
                    *self.free_by_ix.get_mut(&prev_ix).unwrap() = new_size;
                    self.free_by_size.insert((new_size, prev_ix));
                }
                return;
            }
        }
        if let Some(&next_size) = self.free_by_ix.get(&next_ix) {
            // consolidate with next
            let new_size = size + next_size;
            self.free_by_ix.remove(&next_ix);
            self.free_by_ix.insert(ix, new_size);
            self.free_by_size.remove(&(next_size, next_ix));
            self.free_by_size.insert((new_size, ix));
        } else {
            // new isolated free block
            self.free_by_ix.insert(ix, size);
            self.free_by_size.insert((size, ix));
        }
    }
}
