fn main() {
    let current_value = std::env::var("RUSTFLAGS").unwrap_or("".to_owned());
    std::env::set_var(
        "RUSTFLAGS",
        format!("{current_value} --cfg=web_sys_unstable_apis",),
    );
    cargo_run_wasm::run_wasm_with_css("body { margin: 0px; }");
}
