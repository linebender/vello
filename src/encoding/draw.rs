// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use bytemuck::{Pod, Zeroable};
use peniko::{BlendMode, Color};

use super::Monoid;

/// Draw tag representation.
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DrawTag(pub u32);

impl DrawTag {
    /// No operation.
    pub const NOP: Self = Self(0);

    /// Color fill.
    pub const COLOR: Self = Self(0x44);

    /// Linear gradient fill.
    pub const LINEAR_GRADIENT: Self = Self(0x114);

    /// Radial gradient fill.
    pub const RADIAL_GRADIENT: Self = Self(0x2dc);

    /// Image fill.
    pub const IMAGE: Self = Self(0x248);

    /// Begin layer/clip.
    pub const BEGIN_CLIP: Self = Self(0x9);

    /// End layer/clip.
    pub const END_CLIP: Self = Self(0x21);
}

impl DrawTag {
    /// Returns the size of the info buffer (in u32s) used by this tag.
    pub const fn info_size(self) -> u32 {
        (self.0 >> 6) & 0xf
    }
}

/// Draw data for a solid color.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawColor {
    /// Packed little endian RGBA premultiplied color with the alpha component
    /// in the low byte.
    pub rgba: u32,
}

impl DrawColor {
    /// Creates new solid color draw data.
    pub fn new(color: Color) -> Self {
        Self {
            rgba: color.to_premul_u32(),
        }
    }
}

/// Draw data for a linear gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawLinearGradient {
    /// Ramp index.
    pub index: u32,
    /// Start point.
    pub p0: [f32; 2],
    /// End point.
    pub p1: [f32; 2],
}

/// Draw data for a radial gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawRadialGradient {
    /// Ramp index.
    pub index: u32,
    /// Start point.
    pub p0: [f32; 2],
    /// End point.
    pub p1: [f32; 2],
    /// Start radius.
    pub r0: f32,
    /// End radius.
    pub r1: f32,
}

/// Draw data for an image.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawImage {
    /// Packed atlas coordinates.
    pub xy: u32,
    /// Packed image dimensions.
    pub width_height: u32,
}

/// Draw data for a clip or layer.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawBeginClip {
    /// Blend mode.
    pub blend_mode: u32,
    /// Group alpha.
    pub alpha: f32,
}

impl DrawBeginClip {
    /// Creates new clip draw data.
    pub fn new(blend_mode: BlendMode, alpha: f32) -> Self {
        Self {
            blend_mode: (blend_mode.mix as u32) << 8 | blend_mode.compose as u32,
            alpha,
        }
    }
}

/// Monoid for the draw tag stream.
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable, Default)]
#[repr(C)]
pub struct DrawMonoid {
    // The number of paths preceding this draw object.
    pub path_ix: u32,
    // The number of clip operations preceding this draw object.
    pub clip_ix: u32,
    // The offset of the encoded draw object in the scene (u32s).
    pub scene_offset: u32,
    // The offset of the associated info.
    pub info_offset: u32,
}

impl Monoid for DrawMonoid {
    type SourceValue = DrawTag;

    fn new(tag: DrawTag) -> Self {
        Self {
            path_ix: (tag != DrawTag::NOP) as u32,
            clip_ix: tag.0 & 1,
            scene_offset: (tag.0 >> 2) & 0x7,
            info_offset: (tag.0 >> 6) & 0xf,
        }
    }

    fn combine(&self, other: &Self) -> Self {
        Self {
            path_ix: self.path_ix + other.path_ix,
            clip_ix: self.clip_ix + other.clip_ix,
            scene_offset: self.scene_offset + other.scene_offset,
            info_offset: self.info_offset + other.info_offset,
        }
    }
}
