use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod state {
        struct State {
            mat: [f32; 4],
            translate: [f32; 2],
            bbox: [f32; 4],
            // tail is the index in pathsegs of the first segment of a path.
            tail: u32,
            linewidth: f32,
            flags: u32,
            path_count: u32,
            pathseg_count: u32,
        }
    }
}
