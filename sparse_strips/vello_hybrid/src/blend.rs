// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::copy::{GpuCopyInstance, pack_u16_pair};
use crate::schedule::BlendOp;
use bytemuck::{Pod, Zeroable};
use vello_common::peniko::{Compose, Mix};

/// Per-instance data for `blend.wgsl`.
///
/// ```text
/// offset  size  field
/// 0       4     parent_texture_origin: packed u16x2
/// 4       4     parent_texture_size: packed u16x2
/// 8       4     child_texture_origin: packed u16x2
/// 12      4     child_rect_origin: packed u16x2
/// 16      4     child_rect_size: packed u16x2
/// 20      4     blend_rect_origin: packed u16x2
/// 24      4     blend_rect_size: packed u16x2
/// 28      4     blend_config: u32
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuBlendInstance {
    /// Atlas-space origin in the parent layer texture.
    pub(crate) parent_texture_origin: u32,
    /// Size of the parent texture that receives the blend result.
    pub(crate) parent_texture_size: u32,
    /// Atlas-space origin in the child layer texture.
    pub(crate) child_texture_origin: u32,
    /// Scene-space origin of the sampled child layer.
    pub(crate) child_rect_origin: u32,
    /// Scene-space size of the sampled child layer.
    pub(crate) child_rect_size: u32,
    /// Scene-space origin affected by this blend operation.
    pub(crate) blend_rect_origin: u32,
    /// Scene-space size affected by this blend operation.
    pub(crate) blend_rect_size: u32,
    /// Blend mode, opacity, and parent/child texture indices.
    pub(crate) blend_config: u32,
}

impl GpuBlendInstance {
    pub(crate) fn copy_from_parent_in_scratch(self) -> GpuCopyInstance {
        GpuCopyInstance {
            target_texture_origin: self.parent_texture_origin,
            source_texture_origin: self.parent_texture_origin,
            copy_rect_size: self.blend_rect_size,
            target_texture_size: self.parent_texture_size,
        }
    }
}

pub(crate) fn gpu_blend_instance(
    blend: BlendOp,
    parent_texture_size: (u16, u16),
) -> GpuBlendInstance {
    let parent_x = blend.parent.x + (blend.bbox.x0 - blend.parent.scene_bbox.x0);
    let parent_y = blend.parent.y + (blend.bbox.y0 - blend.parent.scene_bbox.y0);

    GpuBlendInstance {
        parent_texture_origin: pack_u16_pair(parent_x, parent_y),
        parent_texture_size: pack_u16_pair(parent_texture_size.0, parent_texture_size.1),
        child_texture_origin: pack_u16_pair(blend.child.x, blend.child.y),
        child_rect_origin: pack_u16_pair(blend.child.scene_bbox.x0, blend.child.scene_bbox.y0),
        child_rect_size: pack_u16_pair(
            blend.child.scene_bbox.width(),
            blend.child.scene_bbox.height(),
        ),
        blend_rect_origin: pack_u16_pair(blend.bbox.x0, blend.bbox.y0),
        blend_rect_size: pack_u16_pair(blend.bbox.width(), blend.bbox.height()),
        blend_config: pack_blend_config(
            blend.blend_mode.mix,
            blend.blend_mode.compose,
            blend.opacity,
            blend.parent.texture_index,
            blend.child.texture_index,
        ),
    }
}

fn pack_blend_config(
    mix: Mix,
    compose: Compose,
    opacity: f32,
    parent_texture_index: usize,
    child_texture_index: usize,
) -> u32 {
    debug_assert!(parent_texture_index <= 1);
    debug_assert!(child_texture_index <= 1);
    pack_compose(compose)
        | (pack_mix(mix) << 8)
        | (u32::from(opacity_to_u8(opacity)) << 16)
        | ((parent_texture_index as u32) << 24)
        | ((child_texture_index as u32) << 25)
}

fn pack_mix(mix: Mix) -> u32 {
    match mix {
        Mix::Normal => 0,
        Mix::Multiply => 1,
        Mix::Screen => 2,
        Mix::Overlay => 3,
        Mix::Darken => 4,
        Mix::Lighten => 5,
        Mix::ColorDodge => 6,
        Mix::ColorBurn => 7,
        Mix::HardLight => 8,
        Mix::SoftLight => 9,
        Mix::Difference => 10,
        Mix::Exclusion => 11,
        Mix::Hue => 12,
        Mix::Saturation => 13,
        Mix::Color => 14,
        Mix::Luminosity => 15,
    }
}

fn pack_compose(compose: Compose) -> u32 {
    match compose {
        Compose::Clear => 0,
        Compose::Copy => 1,
        Compose::Dest => 2,
        Compose::SrcOver => 3,
        Compose::DestOver => 4,
        Compose::SrcIn => 5,
        Compose::DestIn => 6,
        Compose::SrcOut => 7,
        Compose::DestOut => 8,
        Compose::SrcAtop => 9,
        Compose::DestAtop => 10,
        Compose::Xor => 11,
        Compose::Plus => 12,
        Compose::PlusLighter => 13,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "opacity is clamped to the normalized u8 range before packing"
)]
fn opacity_to_u8(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}
