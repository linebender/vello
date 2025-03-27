// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use criterion::{criterion_group, criterion_main};
use vello_bench::{cpu_fine, strip, tile};

criterion_group!(ff, cpu_fine::fill);
criterion_group!(fs, cpu_fine::strip);
criterion_group!(tt, tile::tile);
criterion_group!(srs, strip::render_strips);
criterion_main!(tt, srs, ff, fs);
