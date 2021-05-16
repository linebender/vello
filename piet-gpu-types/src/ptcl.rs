use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod ptcl {
        struct CmdStroke {
            // This is really a Ref<Tile>, but we don't have cross-module
            // references.
            tile_ref: u32,
            half_width: f32,
        }
        struct CmdFill {
            // As above, really Ref<Tile>
            tile_ref: u32,
            backdrop: i32,
        }
        struct CmdColor {
            rgba_color: u32,
        }
        struct CmdImage {
            index: u32,
            offset: [i16; 2],
        }
        struct CmdAlpha {
            alpha: f32,
        }
        struct CmdJump {
            new_ref: u32,
        }
        enum Cmd {
            End,
            Fill(CmdFill),
            Stroke(CmdStroke),
            Solid,
            Alpha(CmdAlpha),
            Color(CmdColor),
            Image(CmdImage),
            BeginClip,
            EndClip,
            Jump(CmdJump),
        }
    }
}
