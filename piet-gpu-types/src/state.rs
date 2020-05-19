use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[gpu_write]
    mod state {
        struct State {
            mat: [f32; 4],
            translate: [f32; 2],
            bbox: [f32; 4],
            linewidth: f32,
            right_edge: f32,
            flags: u32,
        }
    }
}
