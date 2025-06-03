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
use vello_common::coarse::WideTile;
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_common::tile::Tile;
use vello_cpu::region::Regions;

#[vello_bench]
pub fn pack<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf);

    regions.update_regions(|r| {
        b.iter(|| {
            fine.pack(r);
            std::hint::black_box(&r);
        });
    });
}

pub(crate) fn default_blend() -> BlendMode {
    BlendMode::new(Mix::Normal, Compose::SrcOver)
}
