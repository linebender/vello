// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generate local WebGL GLSL outputs and the embedded shader module.

#[cfg(feature = "glsl")]
fn main() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_dir = manifest_dir.join("generated_glsl");
    std::fs::create_dir_all(&output_dir).unwrap();
    std::fs::copy(
        std::path::Path::new(env!("OUT_DIR")).join("compiled_shaders.rs"),
        output_dir.join("compiled_shaders.rs"),
    )
    .unwrap();

    for &(name, vertex_source, fragment_source) in vello_gpu_shaders::glsl::ALL {
        std::fs::write(output_dir.join(format!("{name}.vert.glsl")), vertex_source).unwrap();
        std::fs::write(
            output_dir.join(format!("{name}.frag.glsl")),
            fragment_source,
        )
        .unwrap();
    }
}

#[cfg(not(feature = "glsl"))]
fn main() {
    panic!("enable the `glsl` feature to generate local WebGL GLSL outputs");
}
