use piet_gpu_derive::piet_gpu;

pub use self::scene::{
    Bbox, PietCircle, PietFill, PietItem, PietStrokeLine, PietStrokePolyLine, Point, SimpleGroup,
};

pub use self::scene::{CubicSeg, Element, Fill, LineSeg, QuadSeg, SetLineWidth, Stroke, Transform};

piet_gpu! {
    #[rust_encode]
    mod scene {
        struct Bbox {
            bbox: [i16; 4],
        }
        struct Point {
            xy: [f32; 2],
        }
        struct SimpleGroup {
            n_items: u32,
            // Note: both of the following items are actually arrays
            items: Ref<PietItem>,
            bboxes: Ref<Bbox>,
            offset: Point,
        }
        struct PietCircle {
            rgba_color: u32,
            center: Point,
            radius: f32,
        }
        struct PietStrokeLine {
            flags: u32,
            rgba_color: u32,
            width: f32,
            start: Point,
            end: Point,
        }
        struct PietFill {
            flags: u32,
            rgba_color: u32,
            n_points: u32,
            points: Ref<Point>,
        }
        struct PietStrokePolyLine {
            rgba_color: u32,
            width: f32,
            n_points: u32,
            points: Ref<Point>,
        }
        enum PietItem {
            Group(SimpleGroup),
            Circle(PietCircle),
            Line(PietStrokeLine),
            Fill(PietFill),
            Poly(PietStrokePolyLine),
        }

        // New approach follows (above to be deleted)
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
        }
    }
}
