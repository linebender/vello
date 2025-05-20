## WebGL Demo

Uses Vello Hybrid with a `wgpu` powered WebGL2 backend in the browser.

## Development

Run with `cargo run_wasm -p wgpu_webgl --release`.

## Testing

In order to test this crate, you need to have [`wasm-pack`] installed. Install it using
the steps found in https://rustwasm.github.io/wasm-pack/installer/.

Thereafter, for interactive test sessions, run:

```
wasm-pack test --chrome
# Navigate to printed URL
```

[`wasm-pack`]: https://rustwasm.github.io/wasm-pack/