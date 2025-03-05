// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::todo, reason = "still a prototype")]
#![allow(clippy::cast_possible_truncation, reason = "we need to do this a lot")]

pub mod api;
mod fine;
mod flatten;
mod gpu;
mod pixmap;
mod render;
mod simd;
mod strip;
mod tiling;
mod wide_tile;

pub use gpu::{GpuRenderBufs, GpuRenderCtx, GpuSession};
pub use pixmap::Pixmap;
pub use render::{CsRenderCtx, CsResourceCtx};
pub use tiling::FlatLine;

// TODO: this export should be removed, buffer upload will be internal
pub use gpu::Strip;
