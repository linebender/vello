# Sparse strips WebAssembly builds

The build script produces browser-ready non-SIMD and SIMD128 JavaScript and
WebAssembly modules in `target/sparse-strips-wasm`. Pass the module and variant
to build:

```bash
./sparse_strips/web/build_wasm.sh vello_cpu non-simd
./sparse_strips/web/build_wasm.sh vello_hybrid_webgl simd
```

The supported modules are `vello_cpu`, `vello_hybrid_webgl`, and
`vello_hybrid_wgpu`. Both variants use `opt-level=3`, fat LTO, and one codegen
unit.

The script requires the `wasm32-unknown-unknown` Rust target and the
`wasm-bindgen` CLI version used by this workspace:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114 --locked --force
```

To build all six module variants, report raw and gzip-compressed sizes, and
enforce the committed raw size limits, run:

```bash
./sparse_strips/web/check_wasm_sizes.sh
```

Set `WASM_BINDGEN` to use a `wasm-bindgen` executable that is not on `PATH`.
