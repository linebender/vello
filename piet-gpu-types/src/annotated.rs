use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod annotated {
        // Note: path segments have moved to pathseg, delete these.
        struct AnnoFillLineSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            path_ix: u32,
            // A note: the layout of this struct is shared with
            // AnnoStrokeLineSeg. In that case, we actually write
            // [0.0, 0.0] as the stroke field, to minimize divergence.
        }
        struct AnnoStrokeLineSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            path_ix: u32,
            // halfwidth in both x and y for binning
            stroke: [f32; 2],
        }
        struct AnnoQuadSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            stroke: [f32; 2],
        }
        struct AnnoCubicSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
            stroke: [f32; 2],
        }
        struct AnnoFill {
            rgba_color: u32,
            bbox: [f32; 4],
        }
        struct AnnoStroke {
            rgba_color: u32,
            bbox: [f32; 4],
            // For the nonuniform scale case, this needs to be a 2x2 matrix.
            // That's expected to be uncommon, so we could special-case it.
            linewidth: f32,
        }
        enum Annotated {
            Nop,
            FillLine(AnnoFillLineSeg),
            StrokeLine(AnnoStrokeLineSeg),
            Quad(AnnoQuadSeg),
            Cubic(AnnoCubicSeg),
            Stroke(AnnoStroke),
            Fill(AnnoFill),
        }
    }
}
