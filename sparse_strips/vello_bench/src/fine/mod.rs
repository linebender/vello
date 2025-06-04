// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod blend;
pub(crate) mod fill;
mod gradient;
mod image;
mod rounded_blurred_rect;
mod strip;
// CI will attempt to build this crate with all features enabled, but the problem
// is that the `update_regions` function has a slightly different signature with multithreading
// enabled, which makes it incompatible with the `Bencher` closure. Because of this, we add
// this feature to `vello_bench` as well and disable the benchmark in case it's enabled.
#[cfg(not(feature = "multithreading"))]
mod pack;

pub use blend::*;
pub use fill::*;
pub use gradient::*;
pub use image::*;
#[cfg(not(feature = "multithreading"))]
pub use pack::*;
pub use rounded_blurred_rect::*;
pub use strip::*;
use vello_common::peniko::{BlendMode, Compose, Mix};

pub(crate) fn default_blend() -> BlendMode {
    BlendMode::new(Mix::Normal, Compose::SrcOver)
}
