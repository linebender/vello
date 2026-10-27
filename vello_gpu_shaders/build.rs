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
//! With the `wgpu_external_texture` feature, roots in `TEXTURE_EXTERNAL_SHADERS` are linked a
//! second time with the `texture_external` WESL feature into an extra `<NAME>_TEXTURE_EXTERNAL`
//! WGSL constant. GLSL is always generated from the default link.
//!
//! ```text
//!                                      |-> [default] renamed/minified WGSL -|
//! shaders/*.wesl -> linked WGSL -------|                                    |-> WGSL constants
//!                                      |-> [unminified] linked WGSL --------|
//!                                                                           |-> [glsl] GLSL
//!                                                                           |   + reflection
//!                                                                           |
//!                                  OUT_DIR/compiled_shaders.rs <------------|
//! ```

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use wesl::Wesl;

/// Root modules that need a `texture_external` variant.
#[cfg(feature = "wgpu_external_texture")]
const TEXTURE_EXTERNAL_SHADERS: &[&str] = &["render"];

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
    wgsl_source: String,
    #[cfg(feature = "glsl")]
    original_global_names: std::collections::BTreeMap<String, String>,
    /// WGSL linked with the `texture_external` WESL feature, if this root has such a variant.
    #[cfg(feature = "wgpu_external_texture")]
    texture_external_wgsl_source: Option<String>,
}

// TODO: Format the generated code via `rustfmt`.
// TODO: Use `quote` instead of string concatenation to generate code.
fn main() {
    // Rerun build if the shaders directory changes
    println!("cargo:rerun-if-changed=shaders");
    #[cfg(feature = "glsl")]
    println!("cargo:rerun-if-changed=src/webgl_bootstrap.js");
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

    compiler.use_stripping(true);
    compiler.set_feature(
        "blurred_rounded_rect",
        cfg!(feature = "blurred_rounded_rect"),
    );
    compiler.set_feature("full_image_support", cfg!(feature = "full_image_support"));
    compiler.set_feature("gradient_sweep", cfg!(feature = "gradient_sweep"));
    compiler.set_feature("texture_external", false);

    #[cfg_attr(
        not(feature = "wgpu_external_texture"),
        expect(unused_mut, reason = "only extended with `texture_external` variants")
    )]
    let mut shader_infos: Vec<_> = shader_names
        .into_iter()
        .map(|name| compile_shader(&compiler, name))
        .collect();

    #[cfg(feature = "wgpu_external_texture")]
    {
        compiler.set_feature("texture_external", true);
        for shader_info in &mut shader_infos {
            if TEXTURE_EXTERNAL_SHADERS.contains(&shader_info.name.as_str()) {
                let variant = compile_shader(&compiler, shader_info.name.clone());
                shader_info.texture_external_wgsl_source = Some(variant.wgsl_source);
            }
        }
    }

    shader_infos
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
    let linked_wgsl = compiler
        .compile(&module_path)
        .unwrap_or_else(|error| panic!("Unable to compile `{name}.wesl`: {error}"))
        .to_string();

    #[cfg(feature = "unminified")]
    let wgsl_source = linked_wgsl;
    #[cfg(all(feature = "unminified", feature = "glsl"))]
    let original_global_names = std::collections::BTreeMap::new();

    #[cfg(not(feature = "unminified"))]
    let minified = minify::minify_wgsl(&linked_wgsl);
    #[cfg(not(feature = "unminified"))]
    let wgsl_source = minified.source;
    #[cfg(all(not(feature = "unminified"), feature = "glsl"))]
    let original_global_names = minified.original_global_names;

    ShaderInfo {
        name,
        wgsl_source,
        #[cfg(feature = "glsl")]
        original_global_names,
        #[cfg(feature = "wgpu_external_texture")]
        texture_external_wgsl_source: None,
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

        let mut webgl_programs = Vec::new();
        for shader_info in shader_infos {
            let shader = compile::compile_wgsl_shader(
                &shader_info.wgsl_source,
                &shader_info.name,
                "vs_main",
                "fs_main",
                &shader_info.original_global_names,
            );
            let generated_code = shader.to_generated_code(&shader_info.name);
            writeln!(buf, "{generated_code}").unwrap();
            if WEBGL_PROGRAMS.contains(&shader_info.name.as_str()) {
                webgl_programs.push((
                    shader_info.name.clone(),
                    shader.vertex.source,
                    shader.fragment.source,
                ));
            }
        }
        webgl_programs.sort_by_key(|(name, ..)| {
            WEBGL_PROGRAMS
                .iter()
                .position(|program| program == name)
                .unwrap()
        });
        assert_eq!(
            webgl_programs.len(),
            WEBGL_PROGRAMS.len(),
            "every WebGL program must have a WESL root module"
        );
        generate_webgl_bootstrap_module(&mut buf, &webgl_programs).unwrap();
    }

    buf
}

/// WESL root modules compiled into programs by the WebGL renderer, in the renderer's order.
#[cfg(feature = "glsl")]
const WEBGL_PROGRAMS: [&str; 4] = ["render", "filter", "blend", "copy"];

/// Generates the JavaScript that compiles the WebGL programs before the WebAssembly module loads.
#[cfg(feature = "glsl")]
fn generate_webgl_bootstrap_module(
    buf: &mut String,
    programs: &[(String, String, String)],
) -> std::fmt::Result {
    let mut body = String::new();
    writeln!(body, "const PROGRAMS = {{")?;
    for (name, vertex_source, fragment_source) in programs {
        writeln!(
            body,
            "  {name}: [{}, {}],",
            js_string_literal(vertex_source),
            js_string_literal(fragment_source)
        )?;
    }
    writeln!(body, "}};")?;
    body.push_str(include_str!("src/webgl_bootstrap.js"));
    let module_hash = fnv1a_hash(&body);

    let mut js = String::new();
    writeln!(js, "// Generated by `vello_gpu_shaders` - DO NOT EDIT")?;
    writeln!(js, "const MODULE_HASH = \"{module_hash}\";")?;
    js.push_str(&body);

    writeln!(
        buf,
        "/// JavaScript for compiling the WebGL programs before the WebAssembly module loads."
    )?;
    writeln!(buf, "pub mod webgl_bootstrap {{")?;
    writeln!(
        buf,
        "    /// Identifies [`JS_MODULE`], covering both its GLSL sources and its JavaScript."
    )?;
    writeln!(buf, "    pub const MODULE_HASH: &str = \"{module_hash}\";")?;
    writeln!(
        buf,
        "    /// ES module exporting `compileVelloShaders(canvas, contextAttributes = {{}})`."
    )?;
    writeln!(buf, "    ///")?;
    writeln!(
        buf,
        "    /// It must be generated with the same shader features as the WebAssembly module."
    )?;
    writeln!(buf, "    pub const JS_MODULE: &str = r###\"{js}\"###;")?;
    writeln!(buf, "}}")?;

    Ok(())
}

/// 64-bit FNV-1a, formatted as 16 hexadecimal digits.
#[cfg(feature = "glsl")]
fn fnv1a_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(feature = "glsl")]
fn js_string_literal(source: &str) -> String {
    let mut literal = String::with_capacity(source.len() + 2);
    literal.push('"');
    for c in source.chars() {
        match c {
            '"' => literal.push_str("\\\""),
            '\\' => literal.push_str("\\\\"),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            '\t' => literal.push_str("\\t"),
            c if c < ' ' || c == '\u{2028}' || c == '\u{2029}' => {
                write!(literal, "\\u{:04x}", u32::from(c)).unwrap();
            }
            c => literal.push(c),
        }
    }
    literal.push('"');
    literal
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

    #[cfg(feature = "wgpu_external_texture")]
    if let Some(source) = &shader_info.texture_external_wgsl_source {
        writeln!(
            buf,
            "    /// Linked WGSL source for `{}.wesl` with a `texture_external` binding.",
            shader_info.name
        )?;
        writeln!(
            buf,
            "    pub const {const_name}_TEXTURE_EXTERNAL: &str = r###\"{source}\"###;"
        )?;
    }

    Ok(())
}
