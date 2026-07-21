// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU instance construction for non-default layer blending.

use crate::copy::GpuCopyInstance;
use crate::schedule::round::BlendOp;
use crate::target::TextureParity;
use crate::util::{pack_opacity, pack_u16_pair};
use bytemuck::{Pod, Zeroable};
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::peniko::{Compose, Mix};
use vello_common::tile::Tile;

/// A strip for performing a clipped blend operation.
#[derive(Debug, Copy, Clone)]
pub(crate) struct BlendStrip {
    /// Atlas-space segment origin, packed as `u16x2`.
    origin: u32,
    /// Width of the segment.
    width: u16,
    /// Alpha texture column index, or `None` for a plain fill segment.
    alpha_col_idx: Option<u32>,
}

impl BlendStrip {
    pub(crate) fn from_fill(rect: RectU16, alpha_col_idx: Option<u32>) -> Self {
        Self {
            origin: pack_u16_pair(rect.x0, rect.y0),
            width: rect.width(),
            alpha_col_idx,
        }
    }
}

/// Per-instance data for one blend pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuBlendInstance {
    /// Atlas-space geometry origin, packed as `u16x2`.
    pub(crate) geometry_origin: u32,
    /// Geometry width in the low 16 bits and height in the high 16 bits.
    pub(crate) geometry_size: u32,
    /// Alpha texture column index for strip geometry.
    pub(crate) geometry_alpha_col_idx: u32,
    /// Width and height of the parent layer texture, packed as `u16x2`.
    pub(crate) parent_texture_size: u32,
    /// Atlas-space origin of the layer in the child layer texture, packed as `u16x2`.
    pub(crate) child_texture_origin: u32,
    /// Origin of the child layer expressed in parent texture coordinates, packed as `u16x2`.
    pub(crate) child_parent_origin: u32,
    /// Scene-space width and height of the sampled child layer, packed as `u16x2`.
    pub(crate) child_rect_size: u32,
    /// Packed blend mode, opacity, parent/child texture indices, and alpha-presence flag.
    pub(crate) blend_config: u32,
}

impl GpuBlendInstance {
    pub(crate) fn copy_from_scratch(self) -> GpuCopyInstance {
        GpuCopyInstance {
            dest_texture_origin: self.geometry_origin,
            source_texture_origin: self.geometry_origin,
            copy_rect_size: self.geometry_size,
            dest_texture_size: self.parent_texture_size,
        }
    }
}

pub(crate) fn gpu_blend_instance(
    blend: &BlendOp,
    clip_strip: Option<BlendStrip>,
    parent_texture_size: SizeU16,
) -> GpuBlendInstance {
    let parent_rect = blend.parent_region.texture_rect(blend.blend_bbox);
    let child_parent_rect = blend
        .parent_region
        .texture_rect(blend.child_region.layer_bbox);
    let geometry = clip_strip.map_or_else(
        || GpuBlendGeometry {
            origin: pack_u16_pair(parent_rect.x0, parent_rect.y0),
            size: pack_u16_pair(parent_rect.width(), parent_rect.height()),
            alpha_col_idx: 0,
            has_alpha: false,
        },
        |strip| GpuBlendGeometry {
            origin: strip.origin,
            size: pack_u16_pair(strip.width, Tile::HEIGHT),
            alpha_col_idx: strip.alpha_col_idx.unwrap_or(0),
            has_alpha: strip.alpha_col_idx.is_some(),
        },
    );

    GpuBlendInstance {
        geometry_origin: geometry.origin,
        geometry_size: geometry.size,
        geometry_alpha_col_idx: geometry.alpha_col_idx,
        parent_texture_size: pack_u16_pair(
            parent_texture_size.width(),
            parent_texture_size.height(),
        ),
        child_texture_origin: pack_u16_pair(
            blend.child_region.texture.rect.x0,
            blend.child_region.texture.rect.y0,
        ),
        child_parent_origin: pack_u16_pair(child_parent_rect.x0, child_parent_rect.y0),
        child_rect_size: pack_u16_pair(
            blend.child_region.layer_bbox.width(),
            blend.child_region.layer_bbox.height(),
        ),
        blend_config: pack_blend_config(
            blend.blend_mode.mix,
            blend.blend_mode.compose,
            blend.opacity,
            blend.parent_region.texture.target.texture_parity,
            blend.child_region.texture.target.texture_parity,
            geometry.has_alpha,
        ),
    }
}

/// Packed geometry fields shared while constructing a [`GpuBlendInstance`].
///
/// Either represents a simple rectangle, or a strip with potential alpha, in case the blend layer
/// has clipping.
#[derive(Debug, Copy, Clone)]
struct GpuBlendGeometry {
    /// Atlas-space origin packed as `u16x2`.
    origin: u32,
    /// Width and optional rectangle height packed as `u16x2`.
    size: u32,
    /// Alpha texture column used by clipped strip geometry.
    ///
    /// Irrelevant if the "alpha-presence" flag is not activated.
    alpha_col_idx: u32,
    /// Whether the geometry has per-pixel alpha coverage.
    has_alpha: bool,
}

fn pack_blend_config(
    mix: Mix,
    compose: Compose,
    opacity: f32,
    parent_texture_parity: TextureParity,
    child_texture_parity: TextureParity,
    has_alpha: bool,
) -> u32 {
    pack_compose(compose)
        | (pack_mix(mix) << 8)
        | (u32::from(pack_opacity(opacity)) << 16)
        | (u32::from(parent_texture_parity) << 24)
        | (u32::from(child_texture_parity) << 25)
        | u32::from(has_alpha) << 26
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
