use piet_gpu_derive::piet_gpu;

pub use self::scene::{
    Clip, CubicSeg, Element, FillColor, LineSeg, QuadSeg, SetLineWidth, Transform,
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
        struct FillColor {
            rgba_color: u32,
        }
        struct FillImage {
            index: u32,
            offset: [i16; 2],
        }
        struct SetLineWidth {
            width: f32,
        }
        struct Transform {
            mat: [f32; 4],
            translate: [f32; 2],
        }
        struct Clip {
            bbox: [f32; 4],
            // TODO: add alpha?
        }
        enum Element {
            Nop,

            Line(TagFlags, LineSeg),
            Quad(TagFlags, QuadSeg),
            Cubic(TagFlags, CubicSeg),

            FillColor(TagFlags, FillColor),
            SetLineWidth(SetLineWidth),
            Transform(Transform),
            BeginClip(Clip),
            EndClip(Clip),
            FillImage(FillImage),
        }
    }
}
