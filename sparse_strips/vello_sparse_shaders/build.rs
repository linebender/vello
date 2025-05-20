// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/compile.rs"]
mod compile;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/types.rs"]
mod types;

#[cfg(feature = "glsl")]
use compile::compile_wgsl_shader;

// TODO: Format the generated code via `rustfmt`.
// TODO: Use `quote` instead of string concatenation to generate code.
fn main() {
    // Rerun build if the shaders directory changes
    println!("cargo:rerun-if-changed=shaders");
    let out_dir = env::var_os("OUT_DIR").unwrap();
    // Build outputs a `compiled_shaders.rs` module containing the GLSL source and reflection
    // metadata.
    let dest_path = Path::new(&out_dir).join("compiled_shaders.rs");

    // Get paths to WGSL files
    let shader_dir = PathBuf::from("shaders");
    let shader_paths = fs::read_dir(&shader_dir)
        .expect("Unable to read shaders directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "wgsl" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut shader_infos = Vec::new();
    for path in shader_paths {
        let file_stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read shader {}", path.display()));
        shader_infos.push((file_stem, source));
    }
    shader_infos.sort_by(|a, b| a.0.cmp(&b.0));

    let mut buf = String::new();
    generate_compiled_shaders_module(&mut buf, &shader_infos);
    fs::write(dest_path, &buf).unwrap();
}

fn generate_compiled_shaders_module(buf: &mut String, shader_infos: &[(String, String)]) {
    writeln!(
        buf,
        "// Generated code by `vello_sparse_shaders` - DO NOT EDIT"
    )
    .unwrap();

    writeln!(buf, "/// Re-exporting wgsl shader source code.").unwrap();

    writeln!(buf, "pub mod wgsl {{").unwrap();
    for (shader_name, shader_source) in shader_infos {
        generate_wgsl_shader_module(buf, shader_name, shader_source).unwrap();
    }
    writeln!(buf, "}}").unwrap();

    // Implementation for creating a CompiledGlsl struct per shader assuming the standard entry
    // names of `vs_main` and `fs_main`.
    #[cfg(feature = "glsl")]
    {
        writeln!(
            buf,
            "/// Build time GLSL shaders derived from wgsl shaders."
        )
        .unwrap();

        for (shader_name, shader_source) in shader_infos {
            let compiled = compile_wgsl_shader(shader_source, "vs_main", "fs_main");

            let generated_code = compiled.to_generated_code(shader_name);
            writeln!(buf, "{generated_code}").unwrap();
        }
    }
}

fn generate_wgsl_shader_module<T: Write>(
    buf: &mut T,
    shader_name: &str,
    shader_source: &str,
) -> std::fmt::Result {
    let const_name = shader_name.to_uppercase();
    writeln!(buf, "    /// Source for `{shader_name}.wgsl`")?;
    writeln!(
        buf,
        "    pub const {const_name}: &str = r###\"{shader_source}\"###;"
    )?;

    Ok(())
}
