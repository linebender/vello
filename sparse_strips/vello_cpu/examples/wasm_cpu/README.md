## Vello CPU in the Browser Demo

Run with `cargo run_wasm -p wasm_cpu --release` for scalar build.


To run the demo with SIMD enabled use:

`RUSTFLAGS=-Ctarget-feature=+simd128 cargo run_wasm -p wasm_cpu --release`

