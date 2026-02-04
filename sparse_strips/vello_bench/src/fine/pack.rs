// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{Bencher, Criterion};
use vello_common::coarse::WideTile;
use vello_common::fearless_simd::Simd;
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineKernel, SCRATCH_BUF_SIZE};
use vello_cpu::region::Regions;
use vello_dev_macros::vello_bench;

pub fn pack(c: &mut Criterion) {
    pack_block(c);
    pack_regular(c);
    unpack_block(c);
    unpack_regular(c);
}

#[vello_bench]
pub fn pack_block<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf);

    b.iter(|| {
        regions.update_regions(|region| {
            fine.pack(region);
        });

        std::hint::black_box(&regions);
    });
}

#[vello_bench]
pub fn pack_regular<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH - 1, Tile::HEIGHT, &mut buf);

    b.iter(|| {
        regions.update_regions(|region| {
            fine.pack(region);
        });

        std::hint::black_box(&regions);
    });
}

#[vello_bench]
pub fn unpack_block<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf);

    b.iter(|| {
        regions.update_regions(|region| {
            fine.unpack(region);
        });

        std::hint::black_box(&regions);
    });
}

#[vello_bench]
pub fn unpack_regular<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];
    let mut regions = Regions::new(WideTile::WIDTH - 1, Tile::HEIGHT, &mut buf);

    b.iter(|| {
        regions.update_regions(|region| {
            fine.unpack(region);
        });

        std::hint::black_box(&regions);
    });
}
