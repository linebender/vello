// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Wasm

/// Use [cargo-run-wasm](https://github.com/rukai/cargo-run-wasm) to build an example for web
///
/// Usage:
/// ```
/// cargo run_wasm --package [example_name]
/// ```
/// Generally:
/// ```
/// cargo run_wasm -p with_winit
/// ```
fn main() {
    cargo_run_wasm::run_wasm_cli_with_css("body { margin: 0px; }");
}
