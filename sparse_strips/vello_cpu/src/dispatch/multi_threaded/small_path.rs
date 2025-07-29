// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::kurbo::{BezPath, PathEl, PathSeg, segments};
use smallvec::SmallVec;

const SMALL_PATH_THRESHOLD: usize = 10;

/// Profiling showed that when dealing with paths with few segments (for example rectangles),
/// there is a lot of overhead that comes from allocations/deallocations due to extensive cloning
/// of paths. Because of this, we use a `smallvec` for such paths to prevent excessive
/// allocations/deallocations.
///
/// This optimization is not just based on intuition but has actually been shown to have
/// a significant positive effect on certain benchmarks.
#[expect(
    clippy::large_enum_variant,
    reason = "we make this trade-off on purpose"
)]
#[derive(Debug, Clone)]
pub(crate) enum Path {
    Bez(BezPath),
    Small(SmallPath),
}

impl Path {
    pub(crate) fn new(path: &BezPath) -> Self {
        if path.elements().len() < SMALL_PATH_THRESHOLD {
            Self::Small(SmallPath::new(path))
        } else {
            Self::Bez(path.clone())
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SmallPath(SmallVec<[PathEl; SMALL_PATH_THRESHOLD]>);

impl SmallPath {
    fn new(path: &BezPath) -> Self {
        let mut small_path = SmallVec::new();
        small_path.extend_from_slice(path.elements());

        Self(small_path)
    }

    pub(crate) fn segments(&self) -> impl Iterator<Item = PathSeg> {
        segments(self.0.iter().copied())
    }

    pub(crate) fn elements(&self) -> impl Iterator<Item = PathEl> {
        self.0.iter().copied()
    }
}
