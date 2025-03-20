// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use std::cell::LazyCell;
use std::path::PathBuf;

pub mod data;
pub mod fine;
pub mod read;
pub mod tile;

pub(crate) const FINE_ITERS: usize = 50;
pub(crate) const SEED: [u8; 32] = [0; 32];
pub(crate) static DATA_PATH: LazyCell<PathBuf> =
    LazyCell::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
