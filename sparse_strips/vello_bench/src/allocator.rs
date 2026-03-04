// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{Criterion, black_box};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use vello_common::multi_atlas::Atlas;
use vello_common::multi_atlas::AtlasId;

use crate::SEED;

const ATLAS_SIZE: u32 = 4096;

fn make_atlas() -> Atlas {
    Atlas::new(AtlasId::new(0), ATLAS_SIZE, ATLAS_SIZE)
}

pub fn allocator(c: &mut Criterion) {
    allocate_varied(c);
    allocate_until_full(c);
    alloc_dealloc_churn(c);
}

/// Allocate 1000 rectangles with random sizes between 8x8 and 128x128.
fn allocate_varied(c: &mut Criterion) {
    let mut rng = SmallRng::from_seed(SEED);
    let sizes: Vec<(u32, u32)> = (0..1000)
        .map(|_| (rng.random_range(8..=128), rng.random_range(8..=128)))
        .collect();

    let mut g = c.benchmark_group("allocator");
    g.bench_function("alloc_1000_varied_8_128", |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            for &(w, h) in &sizes {
                black_box(atlas.allocate(w, h));
            }
        });
    });
    g.finish();
}

/// Pack as many 32x32 tiles as possible until the atlas is full.
fn allocate_until_full(c: &mut Criterion) {
    let mut g = c.benchmark_group("allocator");
    g.bench_function("alloc_until_full_32x32", |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            let mut count = 0_u32;
            while atlas.allocate(32, 32).is_some() {
                count += 1;
            }
            black_box(count);
        });
    });
    g.finish();
}

/// Steady-state churn: allocate 500 rects, then repeatedly deallocate one and allocate a new one
/// (500 cycles). Measures reuse / merge performance under typical glyph-cache turnover.
fn alloc_dealloc_churn(c: &mut Criterion) {
    let mut rng = SmallRng::from_seed(SEED);
    let sizes: Vec<(u32, u32)> = (0..1000)
        .map(|_| (rng.random_range(16..=64), rng.random_range(16..=64)))
        .collect();

    let mut g = c.benchmark_group("allocator");
    g.bench_function("churn_500_steady_state", |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            let mut live: Vec<(vello_common::multi_atlas::AllocId, u32, u32)> = Vec::new();

            for &(w, h) in sizes.iter().take(500) {
                if let Some(a) = atlas.allocate(w, h) {
                    live.push((a.id, w, h));
                }
            }

            let mut rng = SmallRng::from_seed(SEED);
            for &(w, h) in sizes.iter().skip(500) {
                if !live.is_empty() {
                    let idx = rng.random_range(0..live.len());
                    let (id, dw, dh) = live.swap_remove(idx);
                    atlas.deallocate(id, dw, dh);
                }
                if let Some(a) = atlas.allocate(w, h) {
                    live.push((a.id, w, h));
                }
            }
            black_box(&live);
        });
    });
    g.finish();
}
