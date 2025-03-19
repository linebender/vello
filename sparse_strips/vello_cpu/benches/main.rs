mod fill;

use criterion::{criterion_group, criterion_main};

criterion_group!(f, fill::fill);
criterion_main!(f);
