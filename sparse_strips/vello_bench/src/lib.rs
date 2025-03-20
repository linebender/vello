// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use std::cell::LazyCell;
use std::path::PathBuf;

pub mod fill;
pub mod read;
pub mod strip;

pub const FINE_ITERS: usize = 50;
pub const SEED: [u8; 32] = [0; 32];
pub const DATA_PATH: LazyCell<PathBuf> =
    LazyCell::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
