// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod blend;
pub(crate) mod fill;
mod gradient;
mod image;
mod pack;
mod rounded_blurred_rect;
mod strip;

pub use blend::*;
pub use fill::*;
pub use gradient::*;
pub use image::*;
pub use pack::*;
pub use rounded_blurred_rect::*;
pub use strip::*;
use vello_common::peniko::{BlendMode, Compose, Mix};

use crate::harness::Registry;

pub(crate) const BENCH_WIDTH: u16 = 256;

pub(crate) fn register(registry: &mut Registry) {
    fill(registry);
    strip(registry);
    pack(registry);
    gradient(registry);
    rounded_blurred_rect(registry);
    blend(registry);
    image(registry);
}

pub(crate) fn default_blend() -> BlendMode {
    BlendMode::new(Mix::Normal, Compose::SrcOver)
}
