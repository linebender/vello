// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};

/// Per-instance data for `copy.wgsl`.
///
/// ```text
/// offset  size  field
/// 0       4     target_texture_origin: packed u16x2
/// 4       4     source_texture_origin: packed u16x2
/// 8       4     copy_rect_size: packed u16x2
/// 12      4     target_texture_size: packed u16x2
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuCopyInstance {
    /// Destination origin in the target atlas texture.
    pub(crate) target_texture_origin: u32,
    /// Source origin in the scratch texture.
    pub(crate) source_texture_origin: u32,
    /// Copy size.
    pub(crate) copy_rect_size: u32,
    /// Size of the render target texture that receives the copy.
    pub(crate) target_texture_size: u32,
}
