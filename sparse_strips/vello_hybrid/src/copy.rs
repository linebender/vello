// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU instance data for copies between intermediate textures.

use bytemuck::{Pod, Zeroable};

/// Per-instance data for `copy.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuCopyInstance {
    /// Destination origin in the target atlas texture, packed as `u16x2`.
    pub(crate) target_texture_origin: u32,
    /// Source origin in the scratch texture, packed as `u16x2`.
    pub(crate) source_texture_origin: u32,
    /// Width and height of the copied region, packed as `u16x2`.
    pub(crate) copy_rect_size: u32,
    /// Width and height of the target texture, packed as `u16x2`.
    pub(crate) target_texture_size: u32,
}
