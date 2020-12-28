use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod annotated {
        struct AnnoFill {
            // The bbox is always first, as we take advantage of common
            // layout when binning.
            bbox: [f32; 4],
            rgba_color: u32,
        }
        struct AnnoFillImage {
            bbox: [f32; 4],
            index: u32,
            offset: [i16; 2],
        }
        struct AnnoStroke {
            bbox: [f32; 4],
            rgba_color: u32,
            // For the nonuniform scale case, this needs to be a 2x2 matrix.
            // That's expected to be uncommon, so we could special-case it.
            linewidth: f32,
        }
        struct AnnoClip {
            bbox: [f32; 4],
        }
        enum Annotated {
            Nop,
            Stroke(AnnoStroke),
            Fill(AnnoFill),
            FillImage(AnnoFillImage),
            BeginClip(AnnoClip),
            EndClip(AnnoClip),
        }
    }
}
