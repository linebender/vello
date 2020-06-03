use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod tile {
        struct Path {
            bbox: [u16; 4],
            tiles: Ref<Tile>,
        }
        struct Tile {
            tile: Ref<TileSeg>,
            backdrop: i32,
        }
        // Segments within a tile are represented as a linked list.
        struct TileSeg {
            start: [f32; 2],
            end: [f32; 2],
            next: Ref<TileSeg>,
        }
    }
}
