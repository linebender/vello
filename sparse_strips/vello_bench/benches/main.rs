// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(dead_code, reason = "Might be unused on platforms not supporting SIMD")]

use criterion::{criterion_group, criterion_main};
use vello_bench::{allocator, coarse, fine, flatten, integration, strip, tile};

criterion_group!(allocator_bench, allocator::allocator);
criterion_group!(coarse_bench, coarse::coarse);
criterion_group!(coarse_with_layer, coarse::coarse_with_layer);
criterion_group!(coarse_layer_4k, coarse::coarse_with_layer_large_viewport);
criterion_group!(fine_solid, fine::fill);
criterion_group!(fine_strip, fine::strip);
criterion_group!(fine_pack, fine::pack);
criterion_group!(fine_gradient, fine::gradient);
criterion_group!(fine_rounded_blurred_rect, fine::rounded_blurred_rect);
criterion_group!(fine_blend, fine::blend);
criterion_group!(fine_image, fine::image);
criterion_group!(tile, tile::tile);
criterion_group!(flatten, flatten::flatten);
criterion_group!(strokes, flatten::strokes);
criterion_group!(render_strips, strip::render_strips);
criterion_group!(render_rect, strip::render_rect);
criterion_group!(integration_bench, integration::images);
criterion_main!(
    allocator_bench,
    coarse_bench,
    coarse_with_layer,
    coarse_layer_4k,
    tile,
    render_strips,
    render_rect,
    flatten,
    strokes,
    fine_solid,
    fine_strip,
    fine_pack,
    fine_gradient,
    fine_rounded_blurred_rect,
    fine_blend,
    fine_image,
    integration_bench
);
