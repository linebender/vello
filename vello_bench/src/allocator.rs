// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use vello_common::multi_atlas::Atlas;
use vello_common::multi_atlas::AtlasId;

use crate::SEED;
use crate::harness::Registry;

const ATLAS_SIZE: u16 = 4096;

fn make_atlas() -> Atlas {
    Atlas::new(AtlasId::new(0), ATLAS_SIZE, ATLAS_SIZE)
}

pub fn register(registry: &mut Registry) {
    registry.extended(|registry| {
        allocate_varied(registry);
        allocate_until_full(registry);
        alloc_dealloc_churn(registry);
    });
}

fn allocate_varied(registry: &mut Registry) {
    let mut rng = SmallRng::from_seed(SEED);
    let sizes: Vec<(u16, u16)> = (0..1000)
        .map(|_| (rng.random_range(8..=128), rng.random_range(8..=128)))
        .collect();

    registry.add("allocator/alloc_1000_varied_8_128", move |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            for &(w, h) in &sizes {
                std::hint::black_box(atlas.allocate(w, h));
            }
        });
    });
}

fn allocate_until_full(registry: &mut Registry) {
    registry.add("allocator/alloc_until_full_32x32", |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            let mut count = 0_u32;
            while atlas.allocate(32, 32).is_some() {
                count += 1;
            }
            std::hint::black_box(count);
        });
    });
}

fn alloc_dealloc_churn(registry: &mut Registry) {
    let mut rng = SmallRng::from_seed(SEED);
    let sizes: Vec<(u16, u16)> = (0..1000)
        .map(|_| (rng.random_range(16..=64), rng.random_range(16..=64)))
        .collect();

    registry.add("allocator/churn_500_steady_state", move |b| {
        b.iter(|| {
            let mut atlas = make_atlas();
            let mut live: Vec<(vello_common::multi_atlas::AllocId, u16, u16)> = Vec::new();

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
            std::hint::black_box(&live);
        });
    });
}
