// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

use criterion::{criterion_group, criterion_main};
use vello_bench::{fill, strip};

criterion_group!(f, fill::fill);
criterion_group!(s, strip::strip);
criterion_main!(f, s);
