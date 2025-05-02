// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build glsl shaders
#[cfg(feature = "webgl")]
use naga::{
    ShaderStage,
    back::glsl,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
#[cfg(feature = "webgl")]
use std::{env, fs, path::Path};

fn main() {
    // Only re-run build script if shader changes.
    println!("cargo:rerun-if-changed=src/shaders/sparse_strip_renderer.wgsl");

    // Only transpile shaders on wasm32 target when "webgl" feature is active. Build.rs runs
    // unconditionally so a runtime check is required for what target is being used.
    #[cfg(feature = "webgl")]
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("wasm32")
    {
        println!("cargo:warning=Compiling GLSL shaders for WebGL target");
        transpile_shaders();
    }
}

#[cfg(feature = "webgl")]
fn transpile_shaders() {
    use naga::back::glsl::{PipelineOptions, Version};

    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_out_dir = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&shader_out_dir).unwrap();

    let source = fs::read_to_string("shaders/sparse_strip_renderer.wgsl").unwrap();
    let module = wgsl::parse_str(&source).unwrap();

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    let info = validator.validate(&module).unwrap();
    // Write the vertex glsl shader.
    {
        let mut options = glsl::Options::default();
        options.version = Version::Embedded {
            version: 300,
            is_webgl: true,
        };

        let pipeline_options = PipelineOptions {
            entry_point: "vs_main".into(),
            multiview: None,
            shader_stage: ShaderStage::Vertex,
        };

        let mut glsl_vs = String::new();
        let mut w = glsl::Writer::new(
            &mut glsl_vs,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Unchecked,
                buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                image_load: naga::proc::BoundsCheckPolicy::Unchecked,
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
        )
        .unwrap();
        w.write().unwrap();

        let out_path = shader_out_dir.join("sparse_strip_renderer.vert");
        fs::write(out_path, transform_vert_shader(glsl_vs)).unwrap();
    }
    // Write the fragment glsl shader.
    {
        let mut options = glsl::Options::default();
        options.version = Version::Embedded {
            version: 300,
            is_webgl: true,
        };

        let pipeline_options = PipelineOptions {
            entry_point: "fs_main".into(),
            multiview: None,
            shader_stage: ShaderStage::Fragment,
        };
        let mut output = String::new();
        let mut w = glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Unchecked,
                buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                image_load: naga::proc::BoundsCheckPolicy::Unchecked,
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
        )
        .unwrap();
        w.write().unwrap();
        let out_path = shader_out_dir.join("sparse_strip_renderer.frag");
        fs::write(out_path, transform_frag_shader(output)).unwrap();
    }
    println!("cargo:warning=Successfully compiled WGSL to GLSL ES 300");
}

#[cfg(feature = "webgl")]
fn replace_or_panic(source: &str, from: &str, to: &str) -> String {
    if !source.contains(from) {
        panic!("Expected pattern not found: {}", from);
    }
    source.replace(from, to)
}

#[cfg(feature = "webgl")]
fn transform_vert_shader(mut source: String) -> String {
    let replacements = [
        // Replace uniform block
        (
            "uniform Config_block_0Vertex { Config _group_0_binding_1_vs; };",
            "uniform ConfigBlock { Config config; };",
        ),
        // Replace input variable names
        (
            "layout(location = 0) in uint _p2vs_location0;",
            "layout(location = 0) in uint xy;",
        ),
        (
            "layout(location = 1) in uint _p2vs_location1;",
            "layout(location = 1) in uint widths;",
        ),
        (
            "layout(location = 2) in uint _p2vs_location2;",
            "layout(location = 2) in uint col;",
        ),
        (
            "layout(location = 3) in uint _p2vs_location3;",
            "layout(location = 3) in uint rgba;",
        ),
        // Replace output variable names
        (
            "smooth out vec2 _vs2fs_location0;",
            "smooth out vec2 tex_coord;",
        ),
        (
            "flat out uint _vs2fs_location1;",
            "flat out uint dense_end;",
        ),
        ("flat out uint _vs2fs_location2;", "flat out uint color;"),
        // Update instance creation in main function
        (
            "StripInstance instance = StripInstance(_p2vs_location0, _p2vs_location1, _p2vs_location2, _p2vs_location3);",
            "StripInstance instance = StripInstance(xy, widths, col, rgba);",
        ),
        // Replace config access paths
        ("_group_0_binding_1_vs.strip_height", "config.strip_height"),
        ("_group_0_binding_1_vs.width", "config.width"),
        ("_group_0_binding_1_vs.height", "config.height"),
        // Replace output assignments at the end of main
        (
            "_vs2fs_location0 = _e71.tex_coord;",
            "tex_coord = _e71.tex_coord;",
        ),
        (
            "_vs2fs_location1 = _e71.dense_end;",
            "dense_end = _e71.dense_end;",
        ),
        ("_vs2fs_location2 = _e71.color;", "color = _e71.color;"),
        // Remove the gl_Position.yz modification line
        (
            "gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);",
            "",
        ),
    ];

    for (original, replacement) in replacements {
        source = replace_or_panic(&source, original, replacement);
    }
    source
}

#[cfg(feature = "webgl")]
fn transform_frag_shader(mut source: String) -> String {
    dbg!(&source);
    let replacements = [
        (
            "Config_block_0Fragment { Config _group_0_binding_1_fs; }",
            "ConfigBlock { Config config; }",
        ),
        // Uniform samplers
        (
            "uniform highp usampler2D _group_0_binding_0_fs;",
            "uniform highp usampler2D alphas_texture;",
        ),
        // Vertex input/output
        (
            "smooth in vec2 _vs2fs_location0;",
            "smooth in vec2 tex_coord;",
        ),
        ("flat in uint _vs2fs_location1;", "flat in uint dense_end;"),
        ("flat in uint _vs2fs_location2;", "flat in uint color;"),
        (
            "layout(location = 0) out vec4 _fs2p_location0;",
            "out vec4 fragColor;",
        ),
        // Variable usage in code
        (
            "VertexOutput(_vs2fs_location0, _vs2fs_location1, _vs2fs_location2, gl_FragCoord)",
            "VertexOutput(tex_coord, dense_end, color, gl_FragCoord)",
        ),
        // Config field access
        (
            "_group_0_binding_1_fs.alphas_tex_width_bits",
            "config.alphas_tex_width_bits",
        ),
        // Texture access
        ("_group_0_binding_0_fs", "alphas_texture"),
        // Output variable
        ("_fs2p_location0 =", "fragColor ="),
    ];
    for (original, replacement) in replacements {
        source = replace_or_panic(&source, original, replacement);
    }
    source
}
