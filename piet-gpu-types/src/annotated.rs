use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod annotated {
        struct AnnoFillImage {
            bbox: [f32; 4],
            index: u32,
            offset: [i16; 2],
        }
        struct AnnoColor {
            bbox: [f32; 4],
            rgba_color: u32,
            // For stroked fills.
            // For the nonuniform scale case, this needs to be a 2x2 matrix.
            // That's expected to be uncommon, so we could special-case it.
            linewidth: f32,
        }
        struct AnnoClip {
            bbox: [f32; 4],
        }
        enum Annotated {
            Nop,
            Color(TagFlags, AnnoColor),
            FillImage(AnnoFillImage),
            BeginClip(AnnoClip),
            EndClip(AnnoClip),
        }
    }
}
