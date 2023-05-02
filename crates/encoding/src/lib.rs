// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Raw scene encoding.

mod binning;
mod clip;
mod config;
mod draw;
mod encoding;
mod glyph;
mod glyph_cache;
mod image_cache;
mod math;
mod monoid;
mod path;
mod ramp_cache;
mod resolve;

pub use binning::BinHeader;
pub use clip::{Clip, ClipBbox, ClipBic, ClipElement};
pub use config::{
    BufferSize, BufferSizes, BumpAllocators, ConfigUniform, RenderConfig, WorkgroupCounts,
    WorkgroupSize,
};
pub use draw::{
    DrawBbox, DrawBeginClip, DrawColor, DrawImage, DrawLinearGradient, DrawMonoid,
    DrawRadialGradient, DrawTag,
};
pub use encoding::{Encoding, StreamOffsets};
pub use glyph::{Glyph, GlyphRun};
pub use math::Transform;
pub use monoid::Monoid;
pub use path::{
    Cubic, Path, PathBbox, PathEncoder, PathMonoid, PathSegment, PathSegmentType, PathTag, Tile,
};
pub use ramp_cache::Ramps;
pub use resolve::{resolve_solid_paths_only, Layout, Patch, Resolver};
