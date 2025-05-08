pub(crate) mod fill;
mod gradient;
mod strip;

use criterion::Bencher;
pub use fill::*;
pub use gradient::*;
pub use strip::*;
use vello_cpu::fine::{Fine, FineType, SCRATCH_BUF_SIZE};
use vello_dev_macros::vello_bench;

#[vello_bench]
pub fn pack<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
    let mut buf = vec![0; SCRATCH_BUF_SIZE];

    b.iter(|| {
        fine.pack(&mut buf);
        std::hint::black_box(&buf);
    });
}
