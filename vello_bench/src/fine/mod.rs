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
use std::sync::Arc;
pub use strip::*;
use vello_common::paint::{ImageId, ImageResolver};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_common::pixmap::Pixmap;

pub(crate) const BENCH_WIDTH: u16 = 256;

pub(crate) struct NoOpImageResolver;

impl ImageResolver for NoOpImageResolver {
    fn resolve(&self, _id: ImageId) -> Option<Arc<Pixmap>> {
        None
    }
}

pub(crate) fn default_blend() -> BlendMode {
    BlendMode::new(Mix::Normal, Compose::SrcOver)
}
