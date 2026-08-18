// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(dead_code, reason = "Might be unused on platforms not supporting SIMD")]

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vello_bench::{allocator, fine, flatten, glyph, integration, sort, strip, tile};
use vello_common::fearless_simd::{Level, Simd, SimdBase, SimdInt, SimdMask, dispatch, mask8x16};
use vello_common::util::Div255Ext;

const PREMULTIPLY_DIMENSIONS: [(usize, usize); 3] = [(800, 500), (1920, 1080), (3840, 2160)];

fn premultiply(c: &mut Criterion) {
    for (width, height) in PREMULTIPLY_DIMENSIONS {
        let rgba = premultiply_source_data(width * height);
        let mut group = c.benchmark_group(format!("premultiply_rgba8/{width}x{height}"));
        group.throughput(Throughput::Bytes(rgba.len() as u64));
        group.bench_function("old_scalar", |b| {
            b.iter_batched_ref(
                || rgba.clone(),
                |rgba| {
                    black_box(old_premultiply(rgba));
                    black_box(&*rgba);
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_function("new_simd_interleaved_64_with_transparency", |b| {
            b.iter_batched_ref(
                || rgba.clone(),
                |rgba| {
                    black_box(new_premultiply_interleaved_64(rgba));
                    black_box(&*rgba);
                },
                BatchSize::LargeInput,
            );
        });
        group.finish();
    }
}

fn premultiply_source_data(pixel_count: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let value = index.to_le_bytes()[0];
        let alpha = if index.is_multiple_of(8) { 128 } else { 255 };
        rgba.extend_from_slice(&[value, 255 - value, value / 2, alpha]);
    }
    rgba
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "premultiplication always produces a value in the u8 range"
)]
fn old_premultiply(data: &mut [u8]) -> bool {
    let mut may_have_transparency = false;
    for pixel in data.chunks_exact_mut(4) {
        let alpha = pixel[3];
        may_have_transparency |= alpha != 255;
        let alpha = u16::from(alpha);
        let premultiply = |component| (u16::from(component) * alpha / 255) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }
    may_have_transparency
}

fn new_premultiply_interleaved_64(data: &mut [u8]) -> bool {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    dispatch!(level, simd => new_premultiply_interleaved_64_impl(simd, data))
}

fn new_premultiply_interleaved_64_impl<S: Simd>(simd: S, data: &mut [u8]) -> bool {
    let (body, tail) = data.as_chunks_mut::<64>();
    let mut transparency = mask8x16::splat(simd, 0);
    for chunk in body {
        let rgba = simd.load_interleaved_128_u8x64(chunk);
        let (rg, ba) = simd.split_u8x64(rgba);
        let (r, g) = simd.split_u8x32(rg);
        let (b, a) = simd.split_u8x32(ba);

        transparency |= !a.simd_eq(255);
        let premultiply = |component| {
            let product = simd.widen_u8x16(component) * simd.widen_u8x16(a);
            simd.narrow_u16x16(product.div_255())
        };
        let premultiplied = simd.combine_u8x32(
            simd.combine_u8x16(premultiply(r), premultiply(g)),
            simd.combine_u8x16(premultiply(b), a),
        );
        simd.store_interleaved_128_u8x64(premultiplied, chunk);
    }

    let mut may_have_transparency = transparency.any_true();
    for pixel in tail.chunks_exact_mut(4) {
        let alpha = u16::from(pixel[3]);
        may_have_transparency |= alpha != 255;
        let premultiply = |component| ((u16::from(component) * alpha + 255) >> 8) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }
    may_have_transparency
}

criterion_group!(allocator_bench, allocator::allocator);
criterion_group!(premultiply_bench, premultiply);
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
criterion_group!(render_strips_cull, strip::render_strips_cull);
criterion_group!(render_rect, strip::render_rect);
criterion_group!(glyph, glyph::glyph);
criterion_group!(sort_tiles, sort::sort);
criterion_group!(integration_bench, integration::images);
criterion_main!(
    premultiply_bench,
    allocator_bench,
    tile,
    render_strips,
    render_strips_cull,
    render_rect,
    flatten,
    strokes,
    glyph,
    fine_solid,
    fine_strip,
    fine_pack,
    fine_gradient,
    fine_rounded_blurred_rect,
    fine_blend,
    fine_image,
    sort_tiles,
    integration_bench
);
