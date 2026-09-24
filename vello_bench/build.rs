// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Embed user-provided SVG benchmark scenes in native and Wasm builds.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let data_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("data");
    println!("cargo:rerun-if-changed={}", data_dir.display());

    let mut paths = fs::read_dir(data_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "svg"))
        .collect::<Vec<_>>();
    paths.sort();

    let mut generated = String::from("const EXTRA_SVGS: &[(&str, &[u8])] = &[\n");
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_stem().unwrap().to_string_lossy();
        generated.push_str(&format!(
            "    ({name:?}, include_bytes!({:?})),\n",
            path.to_string_lossy()
        ));
    }
    generated.push_str("];\n");
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("bench_data.rs"),
        generated,
    )
    .unwrap();
}
