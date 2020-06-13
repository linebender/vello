use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod pathseg {
        struct PathFillLine {
            p0: [f32; 2],
            p1: [f32; 2],
            path_ix: u32,
            // A note: the layout of this struct is shared with
            // PathStrokeLine. In that case, we actually write
            // [0.0, 0.0] as the stroke field, to minimize divergence.
        }
        struct PathStrokeLine {
            p0: [f32; 2],
            p1: [f32; 2],
            path_ix: u32,
            // halfwidth in both x and y for binning
            stroke: [f32; 2],
        }
        struct PathFillCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            path_ix: u32,
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
            // halfwidth in both x and y for binning
            stroke: [f32; 2],
        }
        /*
        struct PathQuad {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            stroke: [f32; 2],
        }
        struct PathCubic {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            stroke: [f32; 2],
        }
        */
        enum PathSeg {
            Nop,
            FillLine(PathFillLine),
            StrokeLine(PathStrokeLine),
            FillCubic(PathFillCubic),
            StrokeCubic(PathStrokeCubic),
            /*
            Quad(AnnoQuadSeg),
            Cubic(AnnoCubicSeg),
            */
        }
    }
}
