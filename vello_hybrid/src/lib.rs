// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::todo, reason = "still a prototype")]
#![allow(clippy::cast_possible_truncation, reason = "we need to do this a lot")]

mod fine;
mod flatten;
mod pixmap;
mod render;
mod simd;
mod strip;
mod tiling;
mod wide_tile;

pub use pixmap::Pixmap;
pub use render::{CsRenderCtx, CsResourceCtx};
pub use tiling::FlatLine;
