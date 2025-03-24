//! Tests whether the WebGL example compiles and runs without panicking.

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen_test::*;
    wasm_bindgen_test_configure!(run_in_browser);
    use webgl::{draw_triangle, render_scene};

    #[wasm_bindgen_test]
    async fn test_renders_triangle() {
        console_error_panic_hook::set_once();
        let mut scene = vello_hybrid::Scene::new(100, 100);
        draw_triangle(&mut scene);
        render_scene(scene, 100, 100).await;
    }
}
