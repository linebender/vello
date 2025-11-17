// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An atomic free list, specialised to Vello API's use cases.
//!
//! TODO: Is this valid?

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

struct FreeListItem {
    root_index: Arc<AtomicU32>,
    next_free: AtomicU32,
    index: u32,
}

impl FreeListItem {
    fn free(&self) {
        self.root_index
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |previous_next| {
                self.next_free.store(previous_next, Ordering::Relaxed);
                Some(self.index)
            })
            .expect("Function always returns `Some`");
    }
}
