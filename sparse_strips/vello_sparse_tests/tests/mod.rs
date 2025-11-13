// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains the test harness for `vello_cpu`.
//! - The `util` module contains shared utility functions that are needed by different
//!   test methods.
//! - We do not use the default Rust test harness, but instead use this `mod.rs` file as the
//!   entry point to run all other tests. The reason we chose this design is that it makes it
//!   easier to define shared utility functions needed by different tests.
//! - If you want to add new tests, try to follow these guidelines:
//!   - If your test can be classified to a clear "topic" (e.g. clipping, blending, etc.), put
//!     it into the corresponding module, or create a new one in case it doesn't exist yet.
//!   - If it cannot be classified cleanly, for now you can just put it into `basic.rs` which
//!     currently holds a bunch of different kinds of tests.
//!   - Tests for bugs should go into `issues.rs`.
//!   - For test naming, try to put the "topic" of the test at the start of the name instead of
//!     the end. For example, if your test case is about blend modes, `blend_mode_hard_light` is
//!     better than `hard_light_blend_mode`. This makes it easier to inspect the reference
//!     snapshots by topic.

#![allow(missing_docs, reason = "we don't need docs for testing")]
#![allow(clippy::cast_possible_truncation, reason = "not critical for testing")]
#![allow(
    clippy::large_stack_arrays,
    reason = "after some experimentation, this lint seems to be\
triggered from `vello_dev_macros` by some mechanism in quote after adding more SIMD variants to the
tests, so there isn't much we can do to prevent it."
)]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod basic;
mod blurred_rounded_rect;
mod clip;
mod compose;
mod filter;
mod glyph;
mod gradient;
mod image;
mod issues;
mod layer;
mod mask;
mod mix;
mod opacity;
mod recording;
mod renderer;
mod scenes;
#[macro_use]
mod util;
#[cfg(target_arch = "wasm32")]
mod wasm_binary_invariants;
