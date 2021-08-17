use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod annotated {
        struct AnnoImage {
            bbox: [f32; 4],
            linewidth: f32,
            index: u32,
            offset: [i16; 2],
        }
        struct AnnoColor {
            bbox: [f32; 4],
            // For stroked fills.
            // For the nonuniform scale case, this needs to be a 2x2 matrix.
            // That's expected to be uncommon, so we could special-case it.
            linewidth: f32,
            rgba_color: u32,
        }
        struct AnnoLinGradient {
            bbox: [f32; 4],
            // For stroked fills.
            linewidth: f32,
            index: u32,
            line_x: f32,
            line_y: f32,
            line_c: f32,
        }
        struct AnnoBeginClip {
            bbox: [f32; 4],
            linewidth: f32,
        }
        struct AnnoEndClip {
            bbox: [f32; 4],
        }
        enum Annotated {
            Nop,
            Color(TagFlags, AnnoColor),
            LinGradient(TagFlags, AnnoLinGradient),
            Image(TagFlags, AnnoImage),
            BeginClip(TagFlags, AnnoBeginClip),
            EndClip(AnnoEndClip),
        }
    }
}
