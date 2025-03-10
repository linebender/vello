// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{num::NonZeroU64, sync::atomic::AtomicU64};

use peniko::kurbo::BezPath;

/// Unique identifier for paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU64);

/// A path with a unique identifier
#[derive(Debug, Clone)]
pub struct Path {
    /// Unique identifier for the path
    pub id: Id,
    /// The actual path geometry
    pub path: BezPath,
    // TODO: Vello encoding. kurbo BezPath can be used in interim
    // Question: probably want to special-case rect, line, ellipse at least
    // Probably also rounded-rect (incl varying corner radii)
}

/// Counter for generating unique IDs
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Id {
    /// Generate a new unique ID
    pub fn get() -> Self {
        let n = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(x) = n.checked_add(1) {
            Self(NonZeroU64::new(x).unwrap())
        } else {
            panic!("wow, overflow of u64, congratulations")
        }
    }
}

impl From<BezPath> for Path {
    /// Create a new Path from a `BezPath`, automatically generating a unique ID
    fn from(path: BezPath) -> Self {
        let id = Id::get();
        Self { id, path }
    }
}
