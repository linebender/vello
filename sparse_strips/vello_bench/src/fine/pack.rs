// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{Bencher, Criterion};
use vello_common::coarse::WideTile;
use vello_common::fearless_simd::Simd;
use vello_common::geometry::RectU16;
use vello_common::pixmap::PixmapMut;
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineKernel};
use vello_cpu::region::Region;
use vello_dev_macros::vello_bench;

// 256 was the original width of a wide tile, hence we are keeping it for consistency.
const TILE_BUFFER_SIZE: usize = 256 * Tile::HEIGHT as usize * 4;

pub fn pack(c: &mut Criterion) {
    pack_block(c);
    unpack_block(c);
}

#[vello_bench]
pub fn pack_block<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; TILE_BUFFER_SIZE];

    b.iter(|| {
        let mut pixmap = PixmapMut::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf).unwrap();
        let mut region = Region::new(
            &mut pixmap,
            RectU16::new(0, 0, WideTile::WIDTH, Tile::HEIGHT),
        );
        fine.pack(0, &mut region);

        std::hint::black_box(&buf);
    });
}

#[vello_bench]
pub fn unpack_block<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
    let mut buf = vec![0; TILE_BUFFER_SIZE];

    b.iter(|| {
        let mut pixmap = PixmapMut::new(WideTile::WIDTH, Tile::HEIGHT, &mut buf).unwrap();
        let mut region = Region::new(
            &mut pixmap,
            RectU16::new(0, 0, WideTile::WIDTH, Tile::HEIGHT),
        );
        fine.unpack(0, &mut region);

        std::hint::black_box(&fine);
    });
}
