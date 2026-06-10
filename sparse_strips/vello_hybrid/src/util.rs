// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This file is a modified version of the vello/src/util.rs file.

//! Simple helpers for managing wgpu state and surfaces.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct IntOffset(pub [u32; 2]);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct IntSize(pub [u32; 2]);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct IntRect {
    pub offset: IntOffset,
    pub size: IntSize,
}

impl IntRect {
    pub(crate) fn new(offset: impl Into<IntOffset>, size: impl Into<IntSize>) -> Self {
        Self {
            offset: offset.into(),
            size: size.into(),
        }
    }
}

impl From<[u32; 2]> for IntOffset {
    fn from(v: [u32; 2]) -> Self {
        Self(v)
    }
}

impl From<[u32; 2]> for IntSize {
    fn from(v: [u32; 2]) -> Self {
        Self(v)
    }
}
