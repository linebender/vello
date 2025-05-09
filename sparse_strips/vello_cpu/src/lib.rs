// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello CPU is a 2D graphics rendering engine written in Rust, for devices with no or underpowered GPUs.
//!
//! It is currently available as an alpha.
//! See the [Caveats](#caveats) section for things you need to be aware of.
//!
//! We also develop [Vello](crates.io/crates/vello), which makes use of the GPU for 2d rendering and has higher performance than Vello CPU..
//! Vello CPU is being developed as part of work to address shortcomings in Vello.
//!
//! ## Usage
//!
//! To use Vello CPU, you need to:
//!
//! - Create a [`RenderContext`][], a 2d drawing context for a fixed-size target area.
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
//! context.render_to_pixmap(&mut target, RenderMode::default());
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
//! ## Caveats
//!
//! Vello CPU is an alpha for several reasons, including the following.
//!
//! ### API stability
//!
//! This API has been developed for an initial version, and has no stability guarantees.
//! Whilst we are in the `0.0.x` release series, any release is likely to breaking.
//! We have known plans to change the API around how image resources are used.
//!
//! ### Documentation
//!
//! We have not yet put any work into documentation.
//!
//! ### Optimisation
//!
//! We have several important optimisation planned, including use of multithreading and SIMD.
//! These are not yet implemented, so performance is not representative of the potential performance.
//!
//! ## Implementation
//!
//! TODO: Point to documentation of sparse strips pattern.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast u16s to u8 in various places where we know for sure that it's < 256"
)]

extern crate alloc;

mod render;

#[doc(hidden)]
/// This is an internal module, do not access directly.
pub mod fine;
mod util;

pub use render::RenderContext;
pub use vello_common::glyph::Glyph;
pub use vello_common::mask::Mask;
pub use vello_common::paint::{Image, Paint, PaintType};
pub use vello_common::pixmap::Pixmap;
pub use vello_common::{color, kurbo, peniko};

/// The selected rendering mode.
#[derive(Copy, Clone, Debug, Default)]
pub enum RenderMode {
    /// Optimize speed (by performing calculations with u8/16).
    #[default]
    OptimizeSpeed,
    /// Optimize quality (by performing calculations with f32).
    OptimizeQuality,
}
