use piet_gpu_derive::piet_gpu;

// Structures representing segments for fill items.

// There is some cut'n'paste here from stroke segments, which can be
// traced to the fact that buffers in GLSL are basically global.
// Maybe there's a way to address that, but in the meantime living
// with the duplication is easiest.

piet_gpu! {
    #[gpu_write]
    mod fill_seg {
        struct FillTileHeader {
            n: u32,
            items: Ref<FillItemHeader>,
        }

        struct FillItemHeader {
            backdrop: i32,
            segments: Ref<FillSegChunk>,
        }

        // TODO: strongly consider using f16. If so, these would be
        // relative to the tile. We're doing f32 for now to minimize
        // divergence from piet-metal originals.
        struct FillSegment {
            start: [f32; 2],
            end: [f32; 2],
        }

        struct FillSegChunk {
            n: u32,
            next: Ref<FillSegChunk>,
            // Segments follow (could represent this as a variable sized array).
        }
    }
}
