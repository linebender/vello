// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::Bencher;
use vello_common::coarse::WideTile;
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineType, SCRATCH_BUF_SIZE};
use vello_cpu::region::Regions;
use vello_dev_macros::vello_bench;

#[vello_bench]
pub fn pack<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf);

    b.iter(|| {
        regions.update_regions(|r| {
            fine.pack(r);
            std::hint::black_box(&r);
        });
    });
}
