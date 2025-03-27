// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use std::path::PathBuf;
use std::sync::LazyLock;

pub mod cpu_fine;
pub mod data;
pub mod strip;
pub mod tile;

pub(crate) const SEED: [u8; 32] = [0; 32];
pub static DATA_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
