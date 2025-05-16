// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(warnings)]
#[path = "src/compile.rs"]
mod compile;
#[allow(warnings)]
#[path = "src/types.rs"]
mod types;

use compile::compile_wgsl_shader;

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
        "// Generated code by `vello_hybrid_shaders` - DO NOT EDIT"
    )
    .unwrap();
    writeln!(
        buf,
        "use crate::types::{{CompiledGlsl, ReflectionMap, Stage}};"
    )
    .unwrap();
    writeln!(
        buf,
        "/// Build time GLSL shaders derived from wgsl shaders."
    )
    .unwrap();
    writeln!(buf, "pub mod shaders {{").unwrap();
    writeln!(
        buf,
        r#"    #![allow(unused_mut, reason = "Increases code gen complexity")]"#
    )
    .unwrap();
    writeln!(
        buf,
        "    use super::{{CompiledGlsl, ReflectionMap, Stage}};"
    )
    .unwrap();

    // Public interface for accessing the CompiledGlsl struct per shader.
    for (shader_name, _) in shader_infos {
        writeln!(buf, "    /// Compiled glsl for `{shader_name}.wgsl`").unwrap();
        writeln!(
            buf,
            "    pub fn {shader_name}() -> CompiledGlsl<&'static str> {{"
        )
        .unwrap();
        writeln!(buf, "        {shader_name}_impl()").unwrap();
        writeln!(buf, "    }}").unwrap();
    }

    // Implementation for creating a CompiledGlsl struct per shader assuming the standard entry
    // names of `vs_main` and `fs_main`.
    for (shader_name, shader_source) in shader_infos {
        let compiled = compile_wgsl_shader(shader_source, "vs_main", "fs_main");

        let generated_code = compiled.to_generated_code(&format!("{shader_name}_impl"));
        writeln!(buf, "{generated_code}").unwrap();
    }

    writeln!(buf, "}}\n").unwrap();
}
