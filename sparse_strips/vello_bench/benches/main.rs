// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use criterion::{criterion_group, criterion_main};
use vello_bench::fine;

criterion_group!(fine_solid, fine::fill);
criterion_group!(fine_strip, fine::strip);
criterion_group!(fine_pack, fine::pack);
criterion_group!(fine_gradient, fine::gradient);
// criterion_group!(tt, tile::tile);
// criterion_group!(srs, strip::render_strips);
criterion_main!(fine_solid, fine_strip, fine_pack, fine_gradient);
