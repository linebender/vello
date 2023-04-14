// Copyright 2022 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};

/// Clip stack element.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClipBic {
    pub a: u32,
    pub b: u32,
}

/// Clip element.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
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
    // Index of the draw object.
    pub ix: u32,
    /// This is a packed encoding of an enum with the sign bit as the tag. If positive,
    /// this entry is a BeginClip and contains the associated path index. If negative,
    /// it is an EndClip and contains the bitwise-not of the EndClip draw object index.    
    pub path_ix: i32,
}

/// Clip bounding box.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct ClipBbox {
    pub bbox: [f32; 4],
}
