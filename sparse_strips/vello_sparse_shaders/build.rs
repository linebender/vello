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
#[path = "src/shader_info.rs"]
mod shader_info;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/types.rs"]
mod types;

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
    let shader_infos = shader_info::load_shader_infos(&shader_dir).expect("Unable to read shaders");
    fs::write(dest_path, generate_compiled_shaders_module(&shader_infos)).unwrap();
}

fn generate_compiled_shaders_module(shader_infos: &[shader_info::ShaderInfo]) -> String {
    let mut buf = String::new();
    writeln!(
        buf,
        "// Generated code by `vello_sparse_shaders` - DO NOT EDIT"
    )
    .unwrap();

    writeln!(buf, "/// Re-exporting wgsl shader source code.").unwrap();

    writeln!(buf, "pub mod wgsl {{").unwrap();
    for shader_info in shader_infos {
        generate_wgsl_shader_module(&mut buf, shader_info).unwrap();
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

        for shader_info in shader_infos {
            let shader =
                compile::compile_wgsl_shader(&shader_info.wgsl_source, "vs_main", "fs_main");
            let generated_code = shader.to_generated_code(&shader_info.name);
            writeln!(buf, "{generated_code}").unwrap();
        }
    }

    buf
}

fn generate_wgsl_shader_module(
    buf: &mut String,
    shader_info: &shader_info::ShaderInfo,
) -> std::fmt::Result {
    let const_name = shader_info.name.to_uppercase();
    writeln!(buf, "    /// Source for `{}.wgsl`", shader_info.name)?;
    writeln!(
        buf,
        "    pub const {const_name}: &str = r###\"{}\"###;",
        shader_info.wgsl_source
    )?;

    Ok(())
}
