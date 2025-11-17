// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A basic `no_std` compatible collection of "garbage collected" items where exclusive access is needed,
//! with a single allocation per added item (plus any `Vec` reallocation).

use alloc::{
    sync::{Arc, Weak},
    vec,
    vec::Vec,
};
use core::sync::atomic::{AtomicU32, Ordering};

pub struct ResourceVec<T, Metadata> {
    data: Vec<(Option<T>, Weak<Metadata>)>,
    metadata: Arc<ResourceVecMetadata>,
    free_indices: Vec<u32>,
}

impl<T, Metadata> ResourceVec<T, Metadata> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            metadata: Arc::new(ResourceVecMetadata {
                first_free: AtomicU32::new(u32::MAX),
            }),
            free_indices: vec![],
        }
    }
    pub fn insert(
        &mut self,
        value: T,
        metadata: impl FnOnce(&mut T, ResourceVecMember) -> Metadata,
    ) {
        // We choose an index from the end of the vector. This is
        // "more likely" to be at the end of the list, improving performance without
        // increasing fragmentation.
        let index = self.free_indices.pop();
        let metada = self.metadata.clone();
        let metadata = metadata(&mut value);
    }
}

impl<T, Metadata> Default for ResourceVec<T, Metadata> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ResourceVecMember {
    meta: Arc<ResourceVecMetadata>,
    index: u32,
}

/// Stores the range within the vector where "free" elements can possibly be found.
///
/// This is only an optimisation over a linear scan (that is, to avoid a linear scan).
struct ResourceVecMetadata {
    first_free: AtomicU32,
}

impl Drop for ResourceVecMember {
    fn drop(&mut self) {
        self.meta
            .first_free
            .fetch_min(self.index, Ordering::Relaxed);
    }
}
