use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod pathseg {
        struct PathCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            path_ix: u32,
            // trans_ix is the transform index. It is 1-based, 0 means no transformation.
            trans_ix: u32,
            // Halfwidth in both x and y for binning. For strokes only.
            stroke: [f32; 2],
        }
        enum PathSeg {
            Nop,
            Cubic(TagFlags, PathCubic),
        }
    }
}
