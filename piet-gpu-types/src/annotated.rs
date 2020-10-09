use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod annotated {
        struct AnnoFill {
            rgba_color: u32,
            bbox: [f32; 4],
        }
        struct AnnoFillMask {
            mask: f32,
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
            Stroke(AnnoStroke),
            Fill(AnnoFill),
            FillMask(AnnoFillMask),
            FillMaskInv(AnnoFillMask),
        }
    }
}
