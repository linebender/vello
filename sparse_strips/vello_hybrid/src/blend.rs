// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::copy::GpuCopyInstance;
use crate::schedule::round::BlendOp;
use crate::util::{Int16Size, pack_opacity, pack_u16_pair};
use bytemuck::{Pod, Zeroable};
use vello_common::peniko::{Compose, Mix};

pub(crate) const BLEND_SCRATCH_INDEX: u8 = 0;

/// Per-instance data for `blend.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuBlendInstance {
    /// Atlas-space origin in the parent layer texture, packed as `u16x2`.
    pub(crate) parent_texture_origin: u32,
    /// Width and height of the blend render target, packed as `u16x2`.
    pub(crate) target_texture_size: u32,
    /// Atlas-space origin in the child layer texture, packed as `u16x2`.
    pub(crate) child_texture_origin: u32,
    /// Scene-space origin of the sampled child layer, packed as `u16x2`.
    pub(crate) child_rect_origin: u32,
    /// Scene-space width and height of the sampled child layer, packed as `u16x2`.
    pub(crate) child_rect_size: u32,
    /// Scene-space origin affected by this blend operation, packed as `u16x2`.
    pub(crate) blend_rect_origin: u32,
    /// Scene-space width and height affected by this blend operation, packed as `u16x2`.
    pub(crate) blend_rect_size: u32,
    /// Packed blend mode, opacity, and parent/child texture indices.
    pub(crate) blend_config: u32,
}

impl GpuBlendInstance {
    pub(crate) fn copy_from_parent_in_scratch(
        self,
        parent_texture_size: Int16Size,
    ) -> GpuCopyInstance {
        GpuCopyInstance {
            target_texture_origin: self.parent_texture_origin,
            source_texture_origin: self.parent_texture_origin,
            copy_rect_size: self.blend_rect_size,
            target_texture_size: pack_u16_pair(
                parent_texture_size.width(),
                parent_texture_size.height(),
            ),
        }
    }
}

pub(crate) fn gpu_blend_instance(
    blend: BlendOp,
    target_texture_size: Int16Size,
) -> GpuBlendInstance {
    let parent_x = blend.parent_region.texture.rect.x0
        + (blend.blend_bbox.x0 - blend.parent_region.scene_bbox.x0);
    let parent_y = blend.parent_region.texture.rect.y0
        + (blend.blend_bbox.y0 - blend.parent_region.scene_bbox.y0);
    GpuBlendInstance {
        parent_texture_origin: pack_u16_pair(parent_x, parent_y),
        target_texture_size: pack_u16_pair(
            target_texture_size.width(),
            target_texture_size.height(),
        ),
        child_texture_origin: blend
            .child_region
            .map(|child| pack_u16_pair(child.texture.rect.x0, child.texture.rect.y0))
            .unwrap_or(0),
        child_rect_origin: blend
            .child_region
            .map(|child| pack_u16_pair(child.scene_bbox.x0, child.scene_bbox.y0))
            .unwrap_or(0),
        child_rect_size: blend
            .child_region
            .map(|child| pack_u16_pair(child.scene_bbox.width(), child.scene_bbox.height()))
            .unwrap_or(0),
        blend_rect_origin: pack_u16_pair(blend.blend_bbox.x0, blend.blend_bbox.y0),
        blend_rect_size: pack_u16_pair(blend.blend_bbox.width(), blend.blend_bbox.height()),
        blend_config: pack_blend_config(
            blend.blend_mode.mix,
            blend.blend_mode.compose,
            blend.opacity,
            blend.parent_region.texture.texture_index,
            blend
                .child_region
                .map(|child| child.texture.texture_index)
                .unwrap_or(0),
        ),
    }
}

fn pack_blend_config(
    mix: Mix,
    compose: Compose,
    opacity: f32,
    parent_texture_index: u8,
    child_texture_index: u8,
) -> u32 {
    debug_assert!(parent_texture_index <= 1);
    debug_assert!(child_texture_index <= 1);

    pack_compose(compose)
        | (pack_mix(mix) << 8)
        | (u32::from(pack_opacity(opacity)) << 16)
        | (u32::from(parent_texture_index) << 24)
        | (u32::from(child_texture_index) << 25)
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
