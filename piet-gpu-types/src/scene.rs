use piet_gpu_derive::piet_gpu;

pub use self::scene::{
    BeginClip, CubicSeg, Element, EndClip, Fill, LineSeg, QuadSeg, SetLineWidth, Stroke, Transform,
};

piet_gpu! {
    #[rust_encode]
    mod scene {
        struct LineSeg {
            p0: [f32; 2],
            p1: [f32; 2],
        }
        struct QuadSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
        }
        struct CubicSeg {
            p0: [f32; 2],
            p1: [f32; 2],
            p2: [f32; 2],
            p3: [f32; 2],
        }
        struct Fill {
            rgba_color: u32,
        }
        struct FillMask {
            mask: f32,
        }
        struct Stroke {
            rgba_color: u32,
        }
        struct SetLineWidth {
            width: f32,
        }
        struct Transform {
            mat: [f32; 4],
            translate: [f32; 2],
        }
        struct BeginClip {
            bbox: [f32; 4],
            // TODO: add alpha?
        }
        struct EndClip {
            // The delta between the BeginClip and EndClip element indices.
            // It is stored as a delta to facilitate binary string concatenation.
            delta: u32,
        }
        enum Element {
            Nop,
            // Another approach to encoding would be to use a single
            // variant but have a bool for fill/stroke. This could be
            // packed into the tag, so the on-the-wire representation
            // would be very similar to what's here.
            StrokeLine(LineSeg),
            FillLine(LineSeg),

            StrokeQuad(QuadSeg),
            FillQuad(QuadSeg),
            StrokeCubic(CubicSeg),
            FillCubic(CubicSeg),
            Stroke(Stroke),
            Fill(Fill),
            SetLineWidth(SetLineWidth),
            Transform(Transform),
            FillMask(FillMask),
            FillMaskInv(FillMask),
            BeginClip(BeginClip),
            EndClip(EndClip),
        }
    }
}
