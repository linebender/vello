// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};
use vello_common::geometry::RectU16;

/// Per-instance data for `copy.wgsl`.
///
/// ```text
/// offset  size  field
/// 0       4     dest_origin: packed u16x2
/// 4       4     source_origin: packed u16x2
/// 8       4     size: packed u16x2
/// 12      4     target_size: packed u16x2
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuCopyInstance {
    /// Destination origin in the target atlas texture.
    pub(crate) dest_origin: u32,
    /// Source origin in the scratch texture.
    pub(crate) source_origin: u32,
    /// Copy size.
    pub(crate) size: u32,
    /// Size of the render target that receives the copy.
    pub(crate) target_size: u32,
}

impl GpuCopyInstance {
    pub(crate) fn clear_rect(&self) -> RectU16 {
        let [x0, y0] = unpack_u16_pair(self.dest_origin);
        let [width, height] = unpack_u16_pair(self.size);
        let x1 = x0.checked_add(width).unwrap();
        let y1 = y0.checked_add(height).unwrap();
        RectU16::new(x0, y0, x1, y1)
    }
}

pub(crate) fn pack_u16_pair(x: u16, y: u16) -> u32 {
    u32::from(x) | (u32::from(y) << 16)
}

pub(crate) fn pack_u32_pair(x: u32, y: u32) -> u32 {
    pack_u16_pair(u16::try_from(x).unwrap(), u16::try_from(y).unwrap())
}

fn unpack_u16_pair(packed: u32) -> [u16; 2] {
    [packed as u16, (packed >> 16) as u16]
}
