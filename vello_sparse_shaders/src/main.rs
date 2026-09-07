// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generate local WebGL GLSL outputs and the embedded shader module.

#[cfg(feature = "glsl")]
mod compile;
#[cfg(feature = "glsl")]
mod lint;
#[cfg(feature = "glsl")]
mod minify;
#[cfg(feature = "glsl")]
mod types;

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

    for &(name, wgsl_source) in vello_sparse_shaders::wgsl::ALL {
        let shader = compile::compile_wgsl_shader(
            wgsl_source,
            name,
            "vs_main",
            "fs_main",
            &std::collections::BTreeMap::default(),
        );
        std::fs::write(
            output_dir.join(format!("{name}.vert.glsl")),
            shader.vertex.source,
        )
        .unwrap();
        std::fs::write(
            output_dir.join(format!("{name}.frag.glsl")),
            shader.fragment.source,
        )
        .unwrap();
    }
}

#[cfg(not(feature = "glsl"))]
fn main() {
    panic!("enable the `glsl` feature to generate local WebGL GLSL outputs");
}
