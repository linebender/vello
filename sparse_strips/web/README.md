# Sparse strips WebAssembly builds

These scripts build browser-ready JavaScript and WebAssembly modules in
`target/sparse-strips-wasm`:

```bash
./sparse_strips/web/build_vello_cpu.sh
./sparse_strips/web/build_vello_hybrid_webgl.sh
./sparse_strips/web/build_vello_hybrid_wgpu.sh
```

The scripts require the `wasm32-unknown-unknown` Rust target and the
`wasm-bindgen` CLI version used by this workspace:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114 --locked --force
```

To build all three modules and enforce their committed size limits, run:

```bash
./sparse_strips/web/check_wasm_sizes.sh
```

Set `WASM_BINDGEN` to use a `wasm-bindgen` executable that is not on `PATH`.
