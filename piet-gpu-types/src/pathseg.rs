use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod pathseg {
        struct PathFillCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            path_ix: u32,
            // trans_ix is 1-based, 0 means no transformation.
            trans_ix: u32,
            // A note: the layout of this struct is shared with
            // PathStrokeCubic. In that case, we actually write
            // [0.0, 0.0] as the stroke field, to minimize divergence.
        }
        struct PathStrokeCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            path_ix: u32,
            trans_ix: u32,
            // halfwidth in both x and y for binning
            stroke: [f32; 2],
        }
        enum PathSeg {
            Nop,
            FillCubic(PathFillCubic),
            StrokeCubic(PathStrokeCubic),
        }
    }
}
