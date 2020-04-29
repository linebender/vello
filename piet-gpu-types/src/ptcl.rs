use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod ptcl {
        struct CmdCircle {
            center: [f32; 2],
            radius: f32,
            rgba_color: u32,
        }
        struct CmdLine {
            start: [f32; 2],
            end: [f32; 2],
        }
        struct CmdStroke {
            // Should be Ref<SegChunk> if we had cross-module references.
            seg_ref: u32,
            half_width: f32,
            rgba_color: u32,
        }
        struct CmdFill {
            start: [f32; 2],
            end: [f32; 2],
        }
        struct CmdFillEdge {
            // The sign is only one bit.
            sign: i32,
            y: f32,
        }
        struct CmdDrawFill {
            backdrop: i32,
            rgba_color: u32,
        }
        struct CmdSolid {
            rgba_color: u32,
        }
        struct CmdJump {
            new_ref: u32,
        }
        enum Cmd {
            End,
            Circle(CmdCircle),
            Line(CmdLine),
            Fill(CmdFill),
            Stroke(CmdStroke),
            FillEdge(CmdFillEdge),
            DrawFill(CmdDrawFill),
            Solid(CmdSolid),
            Jump(CmdJump),
            Bail,
        }
    }
}
