use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod tilegroup {
        struct Instance {
            // Note: a better type would be `Ref<PietItem>` but to do that we
            // would need cross-module references. Punt for now.
            item_ref: u32,
            // A better type would be Point.
            offset: [f32; 2],
        }
        enum TileGroup {
            Instance(Instance),
            End,
        }
    }
}
