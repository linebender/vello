use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;

const FILL_ITERS: usize = 2000;

pub fn fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("fill");

    macro_rules! fill_single {
        ($name:ident, $opaque:expr) => {
            g.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut out = vec![];
                    let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                    let mut color = ColorIter::new($opaque);

                    for _ in 0..FILL_ITERS {
                        fine.fill(0, WideTile::WIDTH as usize, &color.next().unwrap().into());
                    }
                })
            });
        };
    }

    fill_single!(fill_transparent, false);
    fill_single!(fill_opaque, true);
}

const SEED: [u8; 32] = [0; 32];

struct ColorIter {
    opaque: bool,
    rng: StdRng,
}

impl ColorIter {
    fn new(opaque: bool) -> Self {
        Self {
            opaque,
            rng: StdRng::from_seed(SEED),
        }
    }
}

impl Iterator for ColorIter {
    type Item = AlphaColor<Srgb>;

    fn next(&mut self) -> Option<Self::Item> {
        let r = self.rng.random_range(0..=255);
        let g = self.rng.random_range(0..=255);
        let b = self.rng.random_range(0..=255);
        let a = if self.opaque {
            255
        } else {
            self.rng.random_range(0..254)
        };

        Some(AlphaColor::from_rgba8(r, g, b, a))
    }
}
