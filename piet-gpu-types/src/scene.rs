use piet_gpu_derive::piet_gpu;

pub use self::scene::{
    Bbox, PietFill, PietItem, PietStrokeLine, PietStrokePolyLine, Point, SimpleGroup,
};

piet_gpu! {
    #[rust_encode]
    mod scene {
        struct Bbox {
            // TODO: this should be i16
            bbox: [u16; 4],
        }
        struct Point {
            xy: [f32; 2],
        }
        struct SimpleGroup {
            n_items: u32,
            // Note: both of the following items are actually arrays
            items: Ref<PietItem>,
            bboxes: Ref<Bbox>,
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
            Circle(),
            Line(PietStrokeLine),
            Fill(PietFill),
            Poly(PietStrokePolyLine),
        }
    }
}
