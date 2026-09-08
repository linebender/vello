# Vello renderer WebAssembly builds

The build script produces browser-ready non-SIMD and SIMD128 JavaScript and
WebAssembly modules in `target/sparse-strips-wasm`. Pass the module and variant
to build:

```bash
./vello_tests/web/build_wasm.sh vello_cpu non-simd
./vello_tests/web/build_wasm.sh vello_gpu_webgl simd
```

The supported modules are `vello_cpu`, `vello_gpu_webgl`, and
`vello_gpu_wgpu`. Both variants use `opt-level=3`, fat LTO, and one codegen
unit.

The script requires the `wasm32-unknown-unknown` Rust target and the
`wasm-bindgen` CLI version used by this workspace:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114 --locked --force
```

To build the three SIMD128 modules, report raw and gzip-compressed sizes, and
enforce the committed raw size limits, run:

```bash
./vello_tests/web/check_wasm_sizes.sh
```

Set `WASM_BINDGEN` to use a `wasm-bindgen` executable that is not on `PATH`.
