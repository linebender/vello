use criterion::Criterion;

pub fn tile(c: &mut Criterion) {
    let mut g = c.benchmark_group("tile");

    macro_rules! tile_single {
        ($name:ident) => {
            let container = PathContainer::from_data_file(stringify!($name));
            let lines = container.lines();

            g.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut tiler = Tiles::new();
                    tiler.make_tiles(&lines, 4, 4);
                })
            });
        };
    }

    #[cfg(feature = "gs")]
    tile_single!(gs);

    #[cfg(feature = "paris_30k")]
    tile_single!(paris_30k);
}
