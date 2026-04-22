// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Refresh checked-in WebGL GLSL snapshots for `vello_sparse_shaders`.

#[cfg(feature = "glsl")]
mod compile;
#[cfg(feature = "glsl")]
mod shader_info;
#[cfg(feature = "glsl")]
mod types;

#[cfg(feature = "glsl")]
fn main() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let snapshot_dir = manifest_dir.join("generated_glsl");
    std::fs::create_dir_all(&snapshot_dir).unwrap();

    let shader_infos = shader_info::load_shader_infos(&manifest_dir.join("shaders")).unwrap();

    for shader_info in shader_infos {
        let shader = compile::compile_wgsl_shader(&shader_info.wgsl_source, "vs_main", "fs_main");
        std::fs::write(
            vertex_snapshot_path(&snapshot_dir, &shader_info.name),
            shader.vertex.source,
        )
        .unwrap();
        std::fs::write(
            fragment_snapshot_path(&snapshot_dir, &shader_info.name),
            shader.fragment.source,
        )
        .unwrap();
    }
}

#[cfg(not(feature = "glsl"))]
fn main() {
    panic!("enable the `glsl` feature to regenerate WebGL shader snapshots");
}

#[cfg(feature = "glsl")]
fn vertex_snapshot_path(snapshot_dir: &std::path::Path, shader_name: &str) -> std::path::PathBuf {
    snapshot_dir.join(format!("{shader_name}.vert.glsl"))
}

#[cfg(feature = "glsl")]
fn fragment_snapshot_path(snapshot_dir: &std::path::Path, shader_name: &str) -> std::path::PathBuf {
    snapshot_dir.join(format!("{shader_name}.frag.glsl"))
}
