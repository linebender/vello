// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::todo, reason = "still a prototype")]
#![allow(clippy::cast_possible_truncation, reason = "we need to do this a lot")]

pub mod common;
mod gpu;
mod gpu2;
mod render;

pub use gpu::{Config, RenderData, Renderer};
pub use render::RenderContext;
