// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use criterion::{criterion_group, criterion_main};
use vello_bench::{fine, tile};

criterion_group!(ff, fine::fill);
criterion_group!(fs, fine::strip);
criterion_group!(tt, tile::tile);
criterion_main!(tt, ff, fs);
