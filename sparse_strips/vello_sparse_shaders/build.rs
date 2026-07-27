// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use wesl::Wesl;

#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/compile.rs"]
mod compile;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/lint/mod.rs"]
mod lint;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/types.rs"]
mod types;

struct ShaderInfo {
    name: String,
    wgsl_source: String,
}

// TODO: Format the generated code via `rustfmt`.
// TODO: Use `quote` instead of string concatenation to generate code.
fn main() {
    // Rerun build if the shaders directory changes
    println!("cargo:rerun-if-changed=shaders");
    let out_dir = env::var_os("OUT_DIR").unwrap();
    // Build outputs a `compiled_shaders.rs` module containing the GLSL source and reflection
    // metadata.
    let dest_path = Path::new(&out_dir).join("compiled_shaders.rs");

    // Link each WESL root module to WGSL.
    let shader_dir = PathBuf::from("shaders");
    let shader_infos = load_shader_infos(&shader_dir);
    fs::write(dest_path, generate_compiled_shaders_module(&shader_infos)).unwrap();
}

fn load_shader_infos(shader_dir: &Path) -> Vec<ShaderInfo> {
    let shader_names = load_shader_names(shader_dir);
    let mut compiler = Wesl::new(shader_dir);

    // Keep the initial migration behavior-preserving.
    compiler.use_stripping(false);

    shader_names
        .into_iter()
        .map(|name| compile_shader(&compiler, name))
        .collect()
}

fn load_shader_names(shader_dir: &Path) -> Vec<String> {
    let mut shader_names = fs::read_dir(shader_dir)
        .expect("Unable to discover WESL shaders")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()?.to_str()? == "wesl" {
                Some(path.file_stem()?.to_str()?.to_owned())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    shader_names.sort();
    shader_names
}

fn compile_shader<R: wesl::Resolver>(compiler: &Wesl<R>, name: String) -> ShaderInfo {
    let module_path = format!("package::{name}")
        .parse()
        .expect("generated WESL module path should be valid");
    let wgsl_source = compiler
        .compile(&module_path)
        .unwrap_or_else(|error| panic!("Unable to compile `{name}.wesl`: {error}"))
        .to_string();

    ShaderInfo { name, wgsl_source }
}

fn generate_compiled_shaders_module(shader_infos: &[ShaderInfo]) -> String {
    let mut buf = String::new();
    writeln!(
        buf,
        "// Generated code by `vello_sparse_shaders` - DO NOT EDIT"
    )
    .unwrap();

    writeln!(buf, "/// WGSL shader sources linked from WESL modules.").unwrap();

    writeln!(buf, "pub mod wgsl {{").unwrap();
    for shader_info in shader_infos {
        generate_wgsl_shader_module(&mut buf, shader_info).unwrap();
    }
    writeln!(
        buf,
        "    /// All linked WGSL shader sources, keyed by WESL root module name."
    )
    .unwrap();
    writeln!(buf, "    pub const ALL: &[(&str, &str)] = &[").unwrap();
    for shader_info in shader_infos {
        let const_name = shader_info.name.to_uppercase();
        writeln!(buf, "        (\"{}\", {const_name}),", shader_info.name).unwrap();
    }
    writeln!(buf, "    ];").unwrap();
    writeln!(buf, "}}").unwrap();

    // Implementation for creating a CompiledGlsl struct per shader assuming the standard entry
    // names of `vs_main` and `fs_main`.
    #[cfg(feature = "glsl")]
    {
        writeln!(
            buf,
            "/// Build-time GLSL shaders derived from linked WESL modules."
        )
        .unwrap();

        for shader_info in shader_infos {
            let shader = compile::compile_wgsl_shader(
                &shader_info.wgsl_source,
                &shader_info.name,
                "vs_main",
                "fs_main",
            );
            let generated_code = shader.to_generated_code(&shader_info.name);
            writeln!(buf, "{generated_code}").unwrap();
        }
    }

    buf
}

fn generate_wgsl_shader_module(buf: &mut String, shader_info: &ShaderInfo) -> std::fmt::Result {
    let const_name = shader_info.name.to_uppercase();
    writeln!(
        buf,
        "    /// Linked WGSL source for `{}.wesl`.",
        shader_info.name
    )?;
    writeln!(
        buf,
        "    pub const {const_name}: &str = r###\"{}\"###;",
        shader_info.wgsl_source
    )?;

    Ok(())
}
