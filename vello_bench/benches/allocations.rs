// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use vello_bench::allocations::{CountingAllocator, run, tiger};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    run(&[
        tiger::HOT_SCENE_BUILD,
        tiger::HOT_RASTERIZE,
        tiger::HOT_FULL_FRAME,
    ]);
}
