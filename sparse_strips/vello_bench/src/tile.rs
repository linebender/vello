use crate::data::{COAT_OF_ARMS, GHOSTSCRIPT_TIGER, PARIS_30K};
use crate::read::PathContainer;
use criterion::Criterion;
use vello_common::tile::Tiles;

pub fn tile(c: &mut Criterion) {
    let mut g = c.benchmark_group("tile");
    g.sample_size(50);

    macro_rules! tile_single {
        ($item:expr) => {
            let container = PathContainer::from_data_file(&$item);
            let lines = container.lines();

            g.bench_function($item.name, |b| {
                b.iter(|| {
                    let mut tiler = Tiles::new();
                    tiler.make_tiles(&lines, $item.width, $item.height);
                })
            });
        };
    }

    tile_single!(GHOSTSCRIPT_TIGER);
    tile_single!(PARIS_30K);
    tile_single!(COAT_OF_ARMS);
}
