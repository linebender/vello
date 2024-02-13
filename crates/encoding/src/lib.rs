// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Raw scene encoding.

#![warn(clippy::doc_markdown, clippy::semicolon_if_nothing_returned)]

mod binning;
mod clip;
mod config;
mod draw;
mod encoding;
#[cfg(feature = "bump_estimate")]
mod estimate;
#[cfg(feature = "full")]
mod glyph;
#[cfg(feature = "full")]
mod glyph_cache;
#[cfg(feature = "full")]
mod image_cache;
mod mask;
pub mod math;
mod monoid;
mod path;
#[cfg(feature = "full")]
mod ramp_cache;
mod resolve;

pub use binning::BinHeader;
pub use clip::{Clip, ClipBbox, ClipBic, ClipElement};
pub use config::{
    BufferSize, BufferSizes, BumpAllocatorMemory, BumpAllocators, ConfigUniform, IndirectCount,
    RenderConfig, WorkgroupCounts, WorkgroupSize,
};
pub use draw::{
    DrawBbox, DrawBeginClip, DrawColor, DrawImage, DrawLinearGradient, DrawMonoid,
    DrawRadialGradient, DrawTag, DRAW_INFO_FLAGS_FILL_RULE_BIT,
};
pub use encoding::{Encoding, StreamOffsets};
pub use mask::{make_mask_lut, make_mask_lut_16};
pub use math::Transform;
pub use monoid::Monoid;
pub use path::{
    Cubic, LineSoup, Path, PathBbox, PathEncoder, PathMonoid, PathSegment, PathSegmentType,
    PathTag, SegmentCount, Style, Tile,
};
pub use resolve::{resolve_solid_paths_only, Layout};

#[cfg(feature = "full")]
pub use {
    encoding::Resources,
    glyph::{Glyph, GlyphRun},
    ramp_cache::Ramps,
    resolve::{Patch, Resolver},
};

#[cfg(feature = "bump_estimate")]
pub use estimate::BumpEstimator;
