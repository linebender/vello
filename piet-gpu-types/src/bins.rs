use piet_gpu_derive::piet_gpu;

// The output of the binning stage, organized as a linked list of chunks.

piet_gpu! {
    #[gpu_write]
    mod bins {
        struct BinInstance {
            element_ix: u32,
        }

        struct BinChunk {
            // First chunk can have n = 0, subsequent ones not.
            n: u32,
            next: Ref<BinChunk>,
            // Instances follow
        }
    }
}
