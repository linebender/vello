use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod pathseg {
        struct PathFillCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            // succ_ix is the index of the successor whose p0 is the endpoint (p3) of
            // this segment. The last segment in a path has the first segment as its successor.
            succ_ix: u32,
            path_ix: u32,
            // A note: the layout of this struct is shared with
            // PathStrokeCubic. In that case, we actually write
            // [0.0, 0.0] as the stroke field, to minimize divergence.
        }
        struct PathStrokeCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            succ_ix: u32,
            path_ix: u32,
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
