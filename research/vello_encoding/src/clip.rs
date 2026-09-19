// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};

/// Clip stack element.
///
/// This is the bicyclic semigroup, a monoid useful for representing
/// stack depth. There is considerably more detail in the draft paper
///  [Fast GPU bounding boxes on tree-structured scenes].
///
/// [Fast GPU bounding boxes on tree-structured scenes]: https://arxiv.org/abs/2205.11659
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClipBic {
    /// When interpreted as a stack operation, the number of pop operations.
    pub a: u32,
    /// When interpreted as a stack operation, the number of push operations.
    pub b: u32,
}

/// Clip element.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
#[expect(
    clippy::partial_pub_fields,
    reason = "Padding is meaningless to manipulate directly"
)]
pub struct ClipElement {
    pub parent_ix: u32,
    _padding: [u8; 12],
    pub bbox: [f32; 4],
}

/// Clip resolution.
///
/// This is an intermediate element used to match clips to associated paths
/// and is also used to connect begin and end clip pairs.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct Clip {
    /// Index of the draw object.
    pub ix: u32,
    /// This is a packed encoding of an enum with the sign bit as the tag. If positive,
    /// this entry is a `BeginClip` and contains the associated path index. If negative,
    /// it is an `EndClip` and contains the bitwise-not of the `EndClip` draw object index.    
    pub path_ix: i32,
}

/// Clip bounding box.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClipBbox {
    pub bbox: [f32; 4],
}

impl ClipBic {
    pub fn new(a: u32, b: u32) -> Self {
        Self { a, b }
    }

    /// The bicyclic semigroup operation.
    ///
    /// This operation is associative. When interpreted as a stack
    /// operation, it represents doing the pops of `self`, the pushes of
    /// `self`, the pops of `other`, and the pushes of `other`. The middle
    /// two can cancel each other out.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        let m = self.b.min(other.a);
        Self::new(self.a + other.a - m, self.b + other.b - m)
    }
}
