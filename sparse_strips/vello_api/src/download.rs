// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Functionality related to downloading rendering results from the renderer into CPU memory.

use core::sync::atomic::{self, Ordering};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DownloadId(u64);

impl DownloadId {
    pub fn next() -> Self {
        #[cfg(target_has_atomic = "64")]
        {
            // Overflow: u64 starting at 0 incremented by 1 at a time, so cannot overflow.
            static DOWNLOAD_IDS: atomic::AtomicU64 = atomic::AtomicU64::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed))
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            // Overflow: We expect running this code on 32-bit targets to be rare enough in practise
            // that we don't handle overflow.
            // Overflow could only really "matter" in practise if you are running two
            // renderers, where one is moving significantly faster than the other.
            static DOWNLOAD_IDS: atomic::AtomicU32 = atomic::AtomicU32::new(0);
            Self(DOWNLOAD_IDS.fetch_add(1, Ordering::Relaxed).into())
        }
    }
    pub fn to_raw(&self) -> u64 {
        self.0
    }
    // TODO: Do we want/need a `from_raw`?
    // Note that these are "globally scoped", i.e. any renderers using this API make download ids in the same namespace.
}
