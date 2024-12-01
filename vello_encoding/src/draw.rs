// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};
use peniko::{
    color::{AlphaColor, ColorSpace, DynamicColor, OpaqueColor, PremulColor, Srgb},
    BlendMode,
};

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
    pub const RADIAL_GRADIENT: Self = Self(0x29c);

    /// Sweep gradient fill.
    pub const SWEEP_GRADIENT: Self = Self(0x254);

    /// Image fill.
    pub const IMAGE: Self = Self(0x248);

    /// Blurred rounded rectangle.
    pub const BLUR_RECT: Self = Self(0x2d4); // info: 11, scene: 5 (DrawBlurRoundedRect)

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

/// The first word of each draw info stream entry contains the flags.
///
/// This is not part of the draw object stream but gets used after the draw
/// objects get reduced on the GPU. `0` represents a non-zero fill.
/// `1` represents an even-odd fill.
pub const DRAW_INFO_FLAGS_FILL_RULE_BIT: u32 = 1;

/// Draw object bounding box.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct DrawBbox {
    pub bbox: [f32; 4],
}

/// Draw data for a solid color.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawColor {
    /// Packed little endian RGBA premultiplied color with the alpha component
    /// in the low byte.
    pub rgba: u32,
}

impl<CS: ColorSpace> From<AlphaColor<CS>> for DrawColor {
    fn from(color: AlphaColor<CS>) -> Self {
        Self {
            rgba: color.convert::<Srgb>().premultiply().to_rgba8().to_u32(),
        }
    }
}

impl From<DynamicColor> for DrawColor {
    fn from(color: DynamicColor) -> Self {
        Self {
            rgba: color
                .to_alpha_color::<Srgb>()
                .premultiply()
                .to_rgba8()
                .to_u32(),
        }
    }
}

impl<CS: ColorSpace> From<OpaqueColor<CS>> for DrawColor {
    fn from(color: OpaqueColor<CS>) -> Self {
        Self {
            rgba: color
                .convert::<Srgb>()
                .with_alpha(1.)
                .premultiply()
                .to_rgba8()
                .to_u32(),
        }
    }
}

impl<CS: ColorSpace> From<PremulColor<CS>> for DrawColor {
    fn from(color: PremulColor<CS>) -> Self {
        Self {
            rgba: color.convert::<Srgb>().to_rgba8().to_u32(),
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

/// Draw data for a sweep gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawSweepGradient {
    /// Ramp index.
    pub index: u32,
    /// Center point.
    pub p0: [f32; 2],
    /// Normalized start angle.
    pub t0: f32,
    /// Normalized end angle.
    pub t1: f32,
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

/// Draw data for a blurred rounded rectangle.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawBlurRoundedRect {
    /// Solid color brush.
    pub color: DrawColor,
    /// Rectangle width.
    pub width: f32,
    /// Rectangle height.
    pub height: f32,
    /// Rectangle corner radius.
    pub radius: f32,
    /// Standard deviation of gaussian filter.
    pub std_dev: f32,
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
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable, Default, Debug)]
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
