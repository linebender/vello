/// Use [cargo-run-wasm](https://github.com/rukai/cargo-run-wasm) to build an example for web
///
/// Usage:
/// ```
/// cargo run_wasm --package [example_name]
/// ```
/// Generally:
/// ```
/// cargo run_wasm --package with_winit
/// ```

fn main() {
    // HACK: We rely heavily on compute shaders; which means we need WebGPU to be supported
    // However, that requires unstable APIs to be enabled, which are not exposed through a feature
    let current_value = std::env::var("RUSTFLAGS").unwrap_or("".to_owned());
    std::env::set_var(
        "RUSTFLAGS",
        format!("{current_value} --cfg=web_sys_unstable_apis",),
    );
    cargo_run_wasm::run_wasm_with_css("body { margin: 0px; }");
}
