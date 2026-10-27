// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generates the JavaScript snippet that starts compiling Vello's shaders before the WebAssembly
//! module loads.
//!
//! `wasm-bindgen` only accepts snippet files inside the crate, so the snippet is written to the
//! git-ignored `generated` directory.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/bootstrap.js");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bootstrap = std::fs::read_to_string(manifest_dir.join("src/bootstrap.js")).unwrap();
    let snippet = format!(
        "{}\n{bootstrap}",
        vello_gpu_shaders::webgl_bootstrap::JS_MODULE
    );

    let output_dir = manifest_dir.join("generated");
    let output_path = output_dir.join("bootstrap.js");
    // Rewriting unchanged contents would make Cargo rebuild the crate on every build.
    if std::fs::read_to_string(&output_path).ok().as_deref() != Some(snippet.as_str()) {
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(&output_path, snippet).unwrap();
    }
}
