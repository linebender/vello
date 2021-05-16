use piet_gpu_derive::piet_gpu;

// The output of the binning stage, organized as a linked list of chunks.

piet_gpu! {
    #[gpu_write]
    mod bins {
        struct BinInstance {
            element_ix: u32,
        }
    }
}
