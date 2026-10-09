// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build-time shader pipeline.
//!
//! Every top-level file in `shaders` is treated as a root module. WESL links each root with its
//! imports. By default, the linked WGSL is parsed into Naga IR, named declarations are
//! deterministically shortened while renderer-facing entry-point names remain unchanged, and
//! redundant comments and whitespace are removed from the serialized WGSL. The `unminified`
//! feature preserves the linked source for local inspection.
//!
//! The generated module embeds WGSL and, with the `glsl` feature, vertex and fragment GLSL plus
//! reflection metadata. Global names are recorded before default-mode renaming so that this
//! metadata continues to expose the resource names authored in WESL.
//!
//! GLSL is produced from a separate link in which `WGSL_ONLY_FEATURES` are disabled, so shader
//! paths that WebGL cannot support never reach it, regardless of which crate features are enabled.
//!
//! ```text
//!                                      |-> [default] renamed/minified WGSL -|
//! shaders/*.wesl -> linked WGSL -------|                                    |-> WGSL constants
//!                                      |-> [unminified] linked WGSL --------|
//!                                                                           |
//! shaders/*.wesl -> linked WGSL without WGSL-only features -> (same steps) -|-> [glsl] GLSL
//!                                                                           |   + reflection
//!                                                                           |
//!                                  OUT_DIR/compiled_shaders.rs <------------|
//! ```

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use wesl::{StandardResolver, Wesl};

/// WESL features and whether the corresponding crate feature is enabled.
const SHADER_FEATURES: &[(&str, bool)] = &[
    (
        "blurred_rounded_rect",
        cfg!(feature = "blurred_rounded_rect"),
    ),
    ("image_bicubic", cfg!(feature = "image_bicubic")),
    ("gradient_sweep", cfg!(feature = "gradient_sweep")),
    (
        "external_texture_ycbcr",
        cfg!(feature = "external_texture_ycbcr"),
    ),
];

/// Features with no WebGL implementation, always disabled when generating GLSL.
///
/// `external_texture_ycbcr` needs a second plane sampler that the WebGL backend never binds; an
/// unbound sampler defaults to texture unit 0 and conflicts with the integer alphas sampler.
#[cfg(feature = "glsl")]
const WGSL_ONLY_FEATURES: &[&str] = &["external_texture_ycbcr"];

#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/compile.rs"]
mod compile;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/lint/mod.rs"]
mod lint;
#[path = "src/minify.rs"]
mod minify;
#[allow(warnings)]
#[cfg(feature = "glsl")]
#[path = "src/types.rs"]
mod types;

struct ShaderInfo {
    name: String,
    wgsl: LinkedWgsl,
    /// Linked without `WGSL_ONLY_FEATURES`; the input for GLSL generation.
    #[cfg(feature = "glsl")]
    webgl_wgsl: LinkedWgsl,
}

struct LinkedWgsl {
    source: String,
    #[cfg(feature = "glsl")]
    original_global_names: std::collections::BTreeMap<String, String>,
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
    let wgsl_compiler = shader_compiler(shader_dir, &[]);
    #[cfg(feature = "glsl")]
    let webgl_compiler = shader_compiler(shader_dir, WGSL_ONLY_FEATURES);

    shader_names
        .into_iter()
        .map(|name| ShaderInfo {
            wgsl: link_shader(&wgsl_compiler, &name),
            #[cfg(feature = "glsl")]
            webgl_wgsl: link_shader(&webgl_compiler, &name),
            name,
        })
        .collect()
}

fn shader_compiler(shader_dir: &Path, disabled_features: &[&str]) -> Wesl<StandardResolver> {
    let mut compiler = Wesl::new(shader_dir);
    compiler.use_stripping(true);
    for &(feature, enabled) in SHADER_FEATURES {
        compiler.set_feature(feature, enabled && !disabled_features.contains(&feature));
    }
    compiler
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

fn link_shader<R: wesl::Resolver>(compiler: &Wesl<R>, name: &str) -> LinkedWgsl {
    let module_path = format!("package::{name}")
        .parse()
        .expect("generated WESL module path should be valid");
    let linked_wgsl = compiler
        .compile(&module_path)
        .unwrap_or_else(|error| panic!("Unable to compile `{name}.wesl`: {error}"))
        .to_string();

    #[cfg(feature = "unminified")]
    let source = linked_wgsl;
    #[cfg(all(feature = "unminified", feature = "glsl"))]
    let original_global_names = std::collections::BTreeMap::new();

    #[cfg(not(feature = "unminified"))]
    let minified = minify::minify_wgsl(&linked_wgsl);
    #[cfg(not(feature = "unminified"))]
    let source = minified.source;
    #[cfg(all(not(feature = "unminified"), feature = "glsl"))]
    let original_global_names = minified.original_global_names;

    LinkedWgsl {
        source,
        #[cfg(feature = "glsl")]
        original_global_names,
    }
}

fn generate_compiled_shaders_module(shader_infos: &[ShaderInfo]) -> String {
    let mut buf = String::new();
    writeln!(
        buf,
        "// Generated code by `vello_gpu_shaders` - DO NOT EDIT"
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
                &shader_info.webgl_wgsl.source,
                &shader_info.name,
                "vs_main",
                "fs_main",
                &shader_info.webgl_wgsl.original_global_names,
            );
            let generated_code = shader.to_generated_code(&shader_info.name);
            writeln!(buf, "{generated_code}").unwrap();
        }

        writeln!(buf, "/// Generated GLSL shader sources.").unwrap();
        writeln!(buf, "pub mod glsl {{").unwrap();
        writeln!(
            buf,
            "    /// All GLSL shader sources as `(name, vertex, fragment)`, keyed by WESL root module name."
        )
        .unwrap();
        writeln!(buf, "    pub const ALL: &[(&str, &str, &str)] = &[").unwrap();
        for shader_info in shader_infos {
            let name = &shader_info.name;
            writeln!(
                buf,
                "        (\"{name}\", super::{name}::VERTEX_SOURCE, super::{name}::FRAGMENT_SOURCE),"
            )
            .unwrap();
        }
        writeln!(buf, "    ];").unwrap();
        writeln!(buf, "}}").unwrap();
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
        shader_info.wgsl.source
    )?;

    Ok(())
}
