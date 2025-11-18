// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A basic `no_std` + `alloc` compatible collection of "garbage collected" items where exclusive access is needed,
//! with a single allocation per added item (plus any `Vec` reallocation).

use alloc::sync::{Arc, Weak};
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct ResourceVec<T, Handle> {
    data: Vec<Option<T>>,
    associated_handles: Vec<Option<Weak<Handle>>>,
    metadata: Arc<ResourceVecMetadata>,
    // TODO: Consider limiting the size of the vector to reduce the memory usage of this?
    free_indices: Vec<usize>,
}

impl<T, Handle> ResourceVec<T, Handle> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            associated_handles: vec![],
            metadata: Arc::new(ResourceVecMetadata {
                // There aren't any free items to start with, so don't bother.
                first_free: AtomicUsize::new(usize::MAX),
            }),
            free_indices: vec![],
        }
    }
    pub fn insert(
        &mut self,
        mut value: T,
        create_handle: impl FnOnce(&mut T, ResourceVecMember) -> Handle,
    ) -> Arc<Handle> {
        // We choose an index from the end of the vector. This is
        // "more likely" to be at the end of the list. This should mean that "thrashing"
        // allocations shouldn't
        let index = self.free_indices.pop().unwrap_or_else(|| {
            debug_assert_eq!(self.data.len(), self.associated_handles.len());
            let idx = self.data.len();
            self.data.push(None);
            self.associated_handles.push(None);
            idx
        });
        let metadata = ResourceVecMember {
            meta: self.metadata.clone(),
            index,
        };

        let handle = create_handle(&mut value, metadata);
        let handle = Arc::new(handle);

        let data_slot = &mut self.data[index];
        let handle_slot = &mut self.associated_handles[index];

        let previously = data_slot.replace(value);
        *handle_slot = Some(Arc::downgrade(&handle));
        debug_assert!(
            previously.is_none(),
            "Index should only be in `free_indices` if it has been deallocated."
        );
        handle
    }

    pub fn cleanup(&mut self) {
        // Logic: This is the only time the value can ever increase, so doing `fetch_update` here
        // will never overwrite a "lower" write.
        // We never use this atomic result to justify reading/writing to other data, so `Relaxed` is fine.
        let first_relevant_index = self
            .metadata
            .first_free
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |_| Some(usize::MAX))
            .expect("Closure always returns `Some`");
        if first_relevant_index == usize::MAX {
            return;
        }
        // If `first_relevant_index` is zero, we skip 0, so we do always visit the `first_relevant_index`.
        for (idx, item) in self
            .associated_handles
            .iter_mut()
            .enumerate()
            .skip(first_relevant_index)
        {
            if let Some(handle) = item
                && handle.strong_count() == 0
            {
                debug_assert!(self.data[idx].is_some());
                // TODO: Callback instead? Because these also represent "atlas members".
                // That is, this trait turns things into.
                *item = None;
                self.data[idx] = None;
                self.free_indices.push(idx);
            }
        }
    }

    pub fn get_mut(&mut self, handle: &ResourceVecMember) -> &mut T {
        self.data[handle.index].as_mut().unwrap()
    }
    pub fn get(&self, handle: &ResourceVecMember) -> &T {
        self.data[handle.index].as_ref().unwrap()
    }
    pub fn index_mut(&mut self, index: usize) -> &mut T {
        self.data[index].as_mut().unwrap()
    }
    pub fn index_ref(&self, index: usize) -> &T {
        self.data[index].as_ref().unwrap()
    }
}

impl<T, Metadata> Default for ResourceVec<T, Metadata> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ResourceVecMember {
    meta: Arc<ResourceVecMetadata>,
    index: usize,
}

impl ResourceVecMember {
    pub fn raw_idx(&self) -> usize {
        self.index
    }
}

/// Stores the range within the vector where "free" elements can possibly be found.
///
/// This is only an optimisation over a linear scan (that is, to avoid a linear scan).
#[derive(Debug)]
struct ResourceVecMetadata {
    first_free: AtomicUsize,
}

impl Drop for ResourceVecMember {
    fn drop(&mut self) {
        self.meta
            .first_free
            .fetch_min(self.index, Ordering::Relaxed);
    }
}
