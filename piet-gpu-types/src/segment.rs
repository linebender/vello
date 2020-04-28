use piet_gpu_derive::piet_gpu;

// Structures representing segments for stroke/fill items.

piet_gpu! {
    #[gpu_write]
    mod segment {
        struct TileHeader {
            n: u32,
            items: Ref<ItemHeader>,
        }

        // Note: this is only suitable for strokes, fills require backdrop.
        struct ItemHeader {
            n: u32,
            segments: Ref<Segment>,
        }

        // TODO: strongly consider using f16. If so, these would be
        // relative to the tile. We're doing f32 for now to minimize
        // divergence from piet-metal originals.
        struct Segment {
            start: [f32; 2],
            end: [f32; 2],
        }
    }
}
