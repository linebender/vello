// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::copy::{GpuCopyInstance, pack_u16_pair, pack_u32_pair};
use crate::schedule::BlendOp;
use bytemuck::{Pod, Zeroable};
use vello_common::peniko::{Compose, Mix};

/// Per-instance data for `blend.wgsl`.
///
/// ```text
/// offset  size  field
/// 0       4     dest_origin: packed u16x2
/// 4       4     source_origin: packed u16x2
/// 8       4     size: packed u16x2
/// 12      4     texture_indices: packed u16x2
/// 16      4     blend_opacity: u32
/// 20      4     target_size: packed u16x2
/// 24      4     bbox_origin: packed u16x2
/// 28      4     source_scene_origin: packed u16x2
/// 32      4     source_size: packed u16x2
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuBlendInstance {
    /// Destination origin in the target atlas texture.
    pub(crate) dest_origin: u32,
    /// Source layer allocation origin in the sampled layer atlas texture.
    pub(crate) source_origin: u32,
    /// Destination/source draw size for this blend operation.
    pub(crate) size: u32,
    /// Destination texture index followed by source texture index.
    pub(crate) texture_indices: u32,
    /// Blend mode and opacity information.
    pub(crate) blend_opacity: u32,
    /// Size of the render target that receives the blend result.
    pub(crate) target_size: u32,
    /// Scene-space origin of the blended bbox.
    pub(crate) bbox_origin: u32,
    /// Scene-space origin of the sampled source layer.
    pub(crate) source_scene_origin: u32,
    /// Scene-space size of the sampled source layer.
    pub(crate) source_size: u32,
}

impl GpuBlendInstance {
    pub(crate) fn copy_from_dest_in_scratch(self) -> GpuCopyInstance {
        GpuCopyInstance {
            dest_origin: self.dest_origin,
            source_origin: self.dest_origin,
            size: self.size,
            target_size: self.target_size,
        }
    }
}

pub(crate) fn gpu_blend_instance(blend: BlendOp, target_size: (u32, u32)) -> GpuBlendInstance {
    let dest_x = blend.parent.x + u32::from(blend.bbox.x0 - blend.parent.scene_bbox.x0);
    let dest_y = blend.parent.y + u32::from(blend.bbox.y0 - blend.parent.scene_bbox.y0);

    GpuBlendInstance {
        dest_origin: pack_u32_pair(dest_x, dest_y),
        source_origin: pack_u32_pair(blend.source.x, blend.source.y),
        size: pack_u16_pair(blend.bbox.width(), blend.bbox.height()),
        texture_indices: pack_u16_pair(
            u16::try_from(blend.parent.texture_index)
                .expect("layer texture index fits into instance payload"),
            u16::try_from(blend.source.texture_index)
                .expect("layer texture index fits into instance payload"),
        ),
        blend_opacity: pack_blend_opacity(
            blend.blend_mode.mix,
            blend.blend_mode.compose,
            blend.opacity,
        ),
        target_size: pack_u32_pair(target_size.0, target_size.1),
        bbox_origin: pack_u16_pair(blend.bbox.x0, blend.bbox.y0),
        source_scene_origin: pack_u16_pair(blend.source.scene_bbox.x0, blend.source.scene_bbox.y0),
        source_size: pack_u16_pair(
            blend.source.scene_bbox.width(),
            blend.source.scene_bbox.height(),
        ),
    }
}

fn pack_blend_opacity(mix: Mix, compose: Compose, opacity: f32) -> u32 {
    pack_compose(compose) | (pack_mix(mix) << 8) | (u32::from(opacity_to_u8(opacity)) << 16)
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
