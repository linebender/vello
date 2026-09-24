// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(dead_code, reason = "Might be unused on platforms not supporting SIMD")]

mod abi;
#[cfg(not(target_arch = "wasm32"))]
pub mod allocations;
#[cfg(not(target_arch = "wasm32"))]
pub mod allocator;
#[cfg(not(target_arch = "wasm32"))]
pub mod data;
pub mod fine;
#[cfg(not(target_arch = "wasm32"))]
pub mod flatten;
#[cfg(not(target_arch = "wasm32"))]
pub mod glyph;
pub mod harness;
#[cfg(not(target_arch = "wasm32"))]
pub mod integration;
#[cfg(not(target_arch = "wasm32"))]
pub mod pixmap;
#[cfg(not(target_arch = "wasm32"))]
pub mod sort;
#[cfg(not(target_arch = "wasm32"))]
pub mod strip;
#[cfg(not(target_arch = "wasm32"))]
pub mod tile;

/// Construct the benchmark registry.
pub fn registry() -> harness::Registry {
    let mut registry = harness::Registry::new();
    fine::register(&mut registry);
    #[cfg(not(target_arch = "wasm32"))]
    {
        allocator::register(&mut registry);
        pixmap::register(&mut registry);
        tile::register(&mut registry);
        strip::register(&mut registry);
        flatten::register(&mut registry);
        glyph::register(&mut registry);
        sort::register(&mut registry);
        integration::register(&mut registry);
    }
    registry.finish();
    registry
}

pub(crate) const SEED: [u8; 32] = [0; 32];
