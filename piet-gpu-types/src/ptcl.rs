use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod ptcl {
        struct CmdStroke {
            // This is really a Ref<Tile>, but we don't have cross-module
            // references.
            tile_ref: u32,
            half_width: f32,
            rgba_color: u32,
        }
        struct CmdFill {
            // As above, really Ref<Tile>
            tile_ref: u32,
            backdrop: i32,
            rgba_color: u32,
        }
        struct CmdFillImage {
            // As above, really Ref<Tile>
            tile_ref: u32,
            backdrop: i32,
            index: u32,
            offset: [i16; 2],
        }
        struct CmdBeginClip {
            tile_ref: u32,
            backdrop: i32,
        }
        // This is mostly here for expedience and can always be optimized
        // out for pure clips, but will be useful for blend groups.
        struct CmdBeginSolidClip {
            alpha: f32,
        }
        struct CmdEndClip {
            // This will be 1.0 for clips, but we can imagine blend groups.
            alpha: f32,
        }
        struct CmdSolid {
            rgba_color: u32,
        }
        struct CmdSolidImage {
            index: u32,
            offset: [i16; 2],
        }
        struct CmdJump {
            new_ref: u32,
        }
        enum Cmd {
            End,
            Fill(CmdFill),
            FillImage(CmdFillImage),
            BeginClip(CmdBeginClip),
            BeginSolidClip(CmdBeginSolidClip),
            EndClip(CmdEndClip),
            Stroke(CmdStroke),
            Solid(CmdSolid),
            SolidImage(CmdSolidImage),
            Jump(CmdJump),
        }
    }
}
