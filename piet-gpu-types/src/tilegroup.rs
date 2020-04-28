use piet_gpu_derive::piet_gpu;

// Structures representing tilegroup instances (output of kernel 1).
// There are three outputs: the main instances, the stroke instances,
// and the fill instances. All three are conceptually a list of
// instances, but the encoding is slightly different. The first is
// encoded with Instance, Jump, and End. The other two are encoded
// as a linked list of Chunk.

// The motivation for the difference is that the first requires fewer
// registers to track state, but the second contains information that
// is useful up front for doing dynamic allocation in kernel 2, as
// well as increasing read parallelism; the "jump" approach really is
// geared to sequential reading.

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
        struct Jump {
            new_ref: Ref<TileGroup>,
        }
        struct Chunk {
            chunk_n: u32,
            next: Ref<Chunk>,
        }
        enum TileGroup {
            Instance(Instance),
            Jump(Jump),
            End,
        }
    }
}
