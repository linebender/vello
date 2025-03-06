// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::tile::{TILE_WIDTH, Tile};

/// A footprint represents in a compact fashion the range of pixels covered by a tile.
/// We represent this as a u32 so that we can work with bit-shifting for better performance.
pub(crate) struct Footprint(pub(crate) u32);

impl Footprint {
    /// Create a new, empty footprint.
    pub(crate) fn empty() -> Self {
        Self(0)
    }

    /// Create a new footprint from a single index, i.e. [i, i + 1).
    pub(crate) fn from_index(index: u8) -> Self {
        Self(1 << index)
    }

    /// Create a new footprint from a single index, i.e. [start, end).
    pub(crate) fn from_range(start: u8, end: u8) -> Self {
        Self((1 << end) - (1 << start))
    }

    /// The start point of the covered range (inclusive).
    pub(crate) fn x0(&self) -> u32 {
        self.0.trailing_zeros()
    }

    /// The end point of the covered range (exclusive).
    pub(crate) fn x1(&self) -> u32 {
        32 - self.0.leading_zeros()
    }

    /// Extend the range with a single index.
    pub(crate) fn extend(&mut self, index: u8) {
        self.0 |= (1 << index) as u32;
    }

    /// Merge another footprint with the current one.
    pub(crate) fn merge(&mut self, fp: &Self) {
        self.0 |= fp.0;
    }
}

impl Tile {
    // TODO: Profiling shows that this method takes up quite a lot of time in AVX SIMD, investigate
    // if it can be improved.
    pub(crate) fn footprint(&self) -> Footprint {
        let x0 = self.p0.x;
        let x1 = self.p1.x;
        let x_min = x0.min(x1).floor();
        let x_max = x0.max(x1).ceil();
        let start_i = x_min as u32;
        let end_i = (start_i + 1).max(x_max as u32).min(TILE_WIDTH);

        Footprint::from_range(start_i as u8, end_i as u8)
    }
}

#[cfg(test)]
mod tests {
    use crate::footprint::Footprint;

    #[test]
    fn footprint_empty() {
        let fp1 = Footprint::empty();
        // Not optimal behavior, but currently how it is.
        assert_eq!(fp1.x0(), 32);
        assert_eq!(fp1.x1(), 0);
    }

    #[test]
    fn footprint_from_index() {
        let fp1 = Footprint::from_index(0);
        assert_eq!(fp1.x0(), 0);
        assert_eq!(fp1.x1(), 1);

        let fp2 = Footprint::from_index(3);
        assert_eq!(fp2.x0(), 3);
        assert_eq!(fp2.x1(), 4);

        let fp3 = Footprint::from_index(6);
        assert_eq!(fp3.x0(), 6);
        assert_eq!(fp3.x1(), 7);
    }

    #[test]
    fn footprint_from_range() {
        let fp1 = Footprint::from_range(1, 3);
        assert_eq!(fp1.x0(), 1);
        assert_eq!(fp1.x1(), 3);

        // Same comment as for empty.
        let fp2 = Footprint::from_range(2, 2);
        assert_eq!(fp2.x0(), 32);
        assert_eq!(fp2.x1(), 0);

        let fp3 = Footprint::from_range(3, 7);
        assert_eq!(fp3.x0(), 3);
        assert_eq!(fp3.x1(), 7);
    }

    #[test]
    fn footprint_extend() {
        let mut fp = Footprint::empty();
        fp.extend(5);
        assert_eq!(fp.x0(), 5);
        assert_eq!(fp.x1(), 6);

        fp.extend(3);
        assert_eq!(fp.x0(), 3);
        assert_eq!(fp.x1(), 6);

        fp.extend(8);
        assert_eq!(fp.x0(), 3);
        assert_eq!(fp.x1(), 9);

        fp.extend(0);
        assert_eq!(fp.x0(), 0);
        assert_eq!(fp.x1(), 9);

        fp.extend(9);
        assert_eq!(fp.x0(), 0);
        assert_eq!(fp.x1(), 10);
    }

    #[test]
    fn footprint_merge() {
        let mut fp1 = Footprint::from_range(2, 4);
        let fp2 = Footprint::from_range(5, 6);
        fp1.merge(&fp2);

        assert_eq!(fp1.x0(), 2);
        assert_eq!(fp1.x1(), 6);

        let mut fp3 = Footprint::from_range(5, 9);
        let fp4 = Footprint::from_range(7, 10);
        fp3.merge(&fp4);

        assert_eq!(fp3.x0(), 5);
        assert_eq!(fp3.x1(), 10);
    }
}
