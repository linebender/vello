// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=vello_cpu

//! Vello CPU is a 2D graphics rendering engine written in Rust, for devices with no or underpowered GPUs.
//!
//! We also develop [Vello](https://crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
//! Vello CPU is being developed as part of work to address shortcomings in Vello.
//!
//! # Usage
//!
//! To use Vello CPU, you need to:
//!
//! - Create a [`RenderContext`][], a 2D drawing context for a fixed-size target area.
//! - For each object in your scene:
//!   - Set how the object will be painted, using [`set_paint`][RenderContext::set_paint].
//!   - Set the shape to be drawn for that object, using methods like [`fill_path`][RenderContext::fill_path],
//!     [`stroke_path`][RenderContext::stroke_path], or [`glyph_run`][RenderContext::glyph_run].
//! - Render it to an image using [`RenderContext::render_to_pixmap`][].
//!
//! ```rust
//! use vello_cpu::{RenderContext, Pixmap, RenderMode};
//! use vello_cpu::{color::{palette::css, PremulRgba8}, kurbo::Rect};
//! let width = 10;
//! let height = 5;
//! let mut context = RenderContext::new(width, height);
//! context.set_paint(css::MAGENTA);
//! context.fill_rect(&Rect::from_points((3., 1.), (7., 4.)));
//!
//! let mut target = Pixmap::new(width, height);
//! // While calling `flush` is only strictly necessary if you are rendering using
//! // multiple threads, it is recommended to always do this.
//! context.flush();
//! context.render_to_pixmap(&mut target);
//!
//! let expected_render = b"\
//!     0000000000\
//!     0001111000\
//!     0001111000\
//!     0001111000\
//!     0000000000";
//! let magenta = css::MAGENTA.premultiply().to_rgba8();
//! let transparent = PremulRgba8 {r: 0, g: 0, b: 0, a: 0};
//! let mut result = Vec::new();
//! for pixel in target.data() {
//!     if *pixel == magenta {
//!         result.push(b'1');
//!     } else if *pixel == transparent {
//!         result.push(b'0');
//!     } else {
//!          panic!("Got unexpected pixel value {pixel:?}");
//!     }
//! }
//! assert_eq!(&result, expected_render);
//! ```
//!
//! Feel free to take a look at some further
//! [examples](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
//! to better understand how to interact with Vello CPU's API,
//!
//! # Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from [libm][].
//! - `png`(enabled by default): Allow loading [`Pixmap`]s from PNG images.
//!   Also required for rendering glyphs with an embedded PNG. Implies `std`.
//! - `multithreading`: Enable multi-threaded rendering. Implies `std`.
//! - `text` (enabled by default): Enables glyph rendering ([`glyph_run`][RenderContext::glyph_run]).
//! - `u8_pipeline` (enabled by default): Enable the u8 pipeline, for speed focused rendering using u8 math.
//!   The `u8` pipeline will be used for [`OptimizeSpeed`][RenderMode::OptimizeSpeed], if both pipelines are enabled.
//!   If you're using Vello CPU for application rendering, you should prefer this pipeline.
//! - `f32_pipeline`: Enable the `f32` pipeline, which is slower but has more accurate
//!   results. This is espectially useful for rendering test snapshots.
//!   The `f32` pipeline will be used for [`OptimizeQuality`][RenderMode::OptimizeQuality], if both pipelines are enabled.
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//! At least one of `u8_pipeline` and `f32_pipeline` must be enabled.
//! You might choose to disable one of these pipelines if your application
//! won't use it, so as to reduce binary size.
//!
//! # Caveats
//!
//! Overall, Vello CPU is already very feature-rich and should be ready for
//! production use cases. The main caveat at the moment is that the API is
//! still likely to change and not stable yet. For example, we have
//! known plans to change the API around how image resources are used.
//!
//! Additionally, there are certain APIs that are still very much experimental,
//! including for example support for filters. This will be reflected in the
//! documentation of those APIs.
//!
//! Another caveat is that multi-threading with large thread counts
//! (more than 4) might give diminishing returns, especially when
//! making heavy use of layers and clip paths.
//!
//! # Performance
//!
//! Performance benchmarks can be found [here](https://laurenzv.github.io/vello_chart/),
//! As can be seen, Vello CPU achieves compelling performance on both,
//! aarch64 and x86 platforms. We also have SIMD optimizations for WASM SIMD,
//! meaning that you can expect good performance there as well.
//!
//! # Implementation
//!
//! If you want to gain a better understanding of Vello CPU and the
//! sparse strips paradigm, you can take a look at the [accompanying
//! master's thesis](https://ethz.ch/content/dam/ethz/special-interest/infk/inst-pls/plf-dam/documents/StudentProjects/MasterTheses/2025-Laurenz-Thesis.pdf)
//! that was written on the topic. Note that parts of the descriptions might
//! become outdated as the implementation changes, but it should give a good
//! overview nevertheless.
//!
//! <!-- We can't directly link to the libm crate built locally, because our feature is only a pass-through  -->
//! [libm]: https://crates.io/crates/libm
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
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast u16s to u8 in various places where we know for sure that it's < 256"
)]
#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

#[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
compile_error!("vello_cpu must have at least one of the u8 or f32 pipelines enabled");

mod render;

mod dispatch;
mod filter;
mod util;

pub mod api;
#[doc(hidden)]
pub mod fine;
#[doc(hidden)]
pub mod layer_manager;
#[doc(hidden)]
pub mod region;

pub use render::{RenderContext, RenderSettings};
pub use vello_common::fearless_simd::Level;
#[cfg(feature = "text")]
pub use vello_common::glyph::Glyph;
pub use vello_common::mask::Mask;
pub use vello_common::paint::{Image, ImageSource, Paint, PaintType};
pub use vello_common::pixmap::Pixmap;
pub use vello_common::{color, kurbo, peniko};

/// The selected rendering mode.
/// For using [`RenderMode::OptimizeQuality`] you also need to enable `f32_pipeline` feature.
#[derive(Copy, Clone, Debug, Default)]
pub enum RenderMode {
    /// Optimize speed (by performing calculations with u8/16).
    #[default]
    OptimizeSpeed,
    /// Optimize quality (by performing calculations with f32).
    OptimizeQuality,
}
