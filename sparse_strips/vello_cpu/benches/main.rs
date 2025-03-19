// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]

mod fill;

use criterion::{criterion_group, criterion_main};

criterion_group!(f, fill::fill);
criterion_main!(f);
