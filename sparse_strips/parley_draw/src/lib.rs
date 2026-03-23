// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parley Draw provides APIs for efficiently rendering glyphs and paint styles like underline.
//!
//! ## Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from `libm`.
//! - `vello_cpu` (enabled by default): Implements `GlyphRenderer` for Vello CPU's `RenderContext`.
//! - `png`: Enables PNG support for drawing bitmap glyphs.
//!
//! At least one of `std` and `libm` is required.

// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

extern crate alloc;
#[cfg(feature = "libm")]
use core_maths as _;
#[cfg(feature = "png")]
use png as _;
#[cfg(feature = "std")]
extern crate std;

use vello_common::{color, kurbo, peniko, pixmap::Pixmap};

pub mod atlas;
mod colr;
mod glyph;
mod math;

pub mod renderers;

pub use atlas::{
    AtlasCommand, AtlasCommandRecorder, AtlasConfig, AtlasPaint, AtlasSlot, GLYPH_PADDING,
    GlyphAtlas, GlyphCache, GlyphCacheConfig, GlyphCacheKey, ImageCache, PendingClearRect,
    RasterMetrics,
};
pub use colr::{ColrPainter, ColrRenderer};
pub use glyph::{
    CachedGlyphType, Glyph, GlyphBitmap, GlyphCaches, GlyphColr, GlyphOutline, GlyphRenderer,
    GlyphRunBuilder, GlyphRunRenderer, GlyphType, HintCache, HintKey, OutlineCache, PreparedGlyph,
};

#[cfg(feature = "vello_cpu")]
pub use renderers::vello_cpu::{CpuGlyphAtlas, CpuGlyphCaches};

#[cfg(feature = "vello_hybrid")]
pub use renderers::vello_hybrid::{GpuGlyphAtlas, GpuGlyphCaches};
