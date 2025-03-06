// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::todo, reason = "still a prototype")]
#![allow(clippy::cast_possible_truncation, reason = "we need to do this a lot")]

pub mod common;
mod fine;
mod flatten;
mod gpu;
mod render;
mod strip;
mod tiling;
mod wide_tile;

pub use gpu::{GpuRenderBufs, GpuRenderCtx, GpuSession};
pub use render::RenderContext;
