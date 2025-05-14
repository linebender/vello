// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod blend;
pub(crate) mod fill;
mod gradient;
mod image;
mod rounded_blurred_rect;
mod strip;

use criterion::Bencher;
use vello_cpu::fine::{Fine, FineType, SCRATCH_BUF_SIZE};
use vello_dev_macros::vello_bench;

pub use blend::*;
pub use fill::*;
pub use gradient::*;
pub use image::*;
pub use rounded_blurred_rect::*;
pub use strip::*;
use vello_common::peniko::{BlendMode, Compose, Mix};

#[vello_bench]
pub fn pack<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];

    b.iter(|| {
        fine.pack(&mut buf);
        std::hint::black_box(&buf);
    });
}

pub(crate) fn default_blend() -> BlendMode {
    BlendMode::new(Mix::Normal, Compose::SrcOver)
}
