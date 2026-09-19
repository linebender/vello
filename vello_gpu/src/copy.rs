// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU instance data for copies between intermediate textures.

use bytemuck::{Pod, Zeroable};

/// Per-instance data for one copy pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuCopyInstance {
    /// Origin in the destination texture page, packed as `u16x2`.
    pub(crate) dest_texture_origin: u32,
    /// Origin in the source texture page, packed as `u16x2`.
    pub(crate) source_texture_origin: u32,
    /// Width and height of the copied region, packed as `u16x2`.
    pub(crate) copy_rect_size: u32,
    /// Width and height of the destination texture page, packed as `u16x2`.
    pub(crate) dest_texture_size: u32,
}
