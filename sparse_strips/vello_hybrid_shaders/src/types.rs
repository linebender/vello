// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types that are shared between the main crate, and the compiled shader file.

#![allow(
    dead_code,
    reason = "False positives as this module is used at build time."
)]

use naga::{Arena, GlobalVariable, back::glsl::ReflectionInfo};
use std::collections::BTreeMap;

/// `ReflectionMap` is a simplified and easily construtable derivative of
/// [`naga::back::glsl::ReflectionInfo`], mapping the wgsl variable names to the generated glsl
/// names.
#[derive(Debug)]
pub(crate) struct ReflectionMap<S: AsRef<str> = String> {
    /// Mapping of wgsl texture identifier to the generated glsl identifier.
    // TODO: It may make sense to pass through the sampler type. E.g. `sampler2D` or `usampler2D`.
    pub texture_mapping: BTreeMap<S, S>,
    /// Mapping of wgsl uniform identifier to the generated glsl identifier.
    pub uniforms: BTreeMap<S, S>,
}

impl ReflectionMap {
    /// Create a new `ReflectionMap` given the [`naga`] compile info.
    pub(crate) fn new(info: ReflectionInfo, global_vars: &Arena<GlobalVariable>) -> Self {
        debug_assert_eq!(info.varying.len(), 0, "unimplemented");
        debug_assert_eq!(info.push_constant_items.len(), 0, "unimplemented");
        let mut texture_mapping = BTreeMap::default();
        let mut uniforms = BTreeMap::default();

        for (glsl_name, texture_handles) in info.texture_mapping {
            if let Ok(wgsl_var) = global_vars.try_get(texture_handles.texture) {
                if let Some(wgsl_name) = &wgsl_var.name {
                    texture_mapping.insert(wgsl_name.clone(), glsl_name);
                }
            }
        }

        for (handle, glsl_name) in info.uniforms {
            if let Ok(wgsl_var) = global_vars.try_get(handle) {
                if let Some(wgsl_name) = &wgsl_var.name {
                    uniforms.insert(wgsl_name.clone(), glsl_name);
                }
            }
        }

        Self {
            texture_mapping,
            uniforms,
        }
    }

    /// Output the code to construct this instance of a `ReflectionMap`.
    pub(crate) fn to_generated_code(&self, name: &str) -> String {
        let mut generated_code = String::new();
        generated_code.push_str(
            format!("pub struct {} {{\n}}\n", name).as_str(),
        );
        generated_code.push_str(
            format!("impl {} {{\n", name).as_str(),
        );
        for (wgsl_name, glsl_name) in &self.uniforms {
            generated_code.push_str(&format!(
                "\t pub const {}_UNIFORM: &'static str = \"{}\";\n",
                wgsl_name.to_uppercase(), glsl_name
            ));
        }
        for (wgsl_name, glsl_name) in &self.texture_mapping {
            generated_code.push_str(&format!(
                "\t pub const {}_TEXTURE: &'static str = \"{}\";\n",
                wgsl_name.to_uppercase(), glsl_name
            ));
        }
        generated_code.push_str("}\n");

        generated_code
    }
}

#[derive(Debug)]
/// A glsl vertex or fragment shader stage, with reflection info.
pub struct Stage<S: AsRef<str> = String> {
    /// glsl source code.
    pub source: S,
    /// Reflection info from wgsl identifiers to the glsl identifiers.
    pub reflection_map: ReflectionMap<S>,
}

#[derive(Debug)]
/// Compiled glsl with reflection info for mapping between the wgsl source of truth and generated
/// glsl.
pub struct CompiledGlsl<S: AsRef<str> = String> {
    /// Vertex stage.
    pub vertex: Stage<S>,
    /// Fragment stage.
    pub fragment: Stage<S>,
}

impl CompiledGlsl {
    /// Output the code to construct this instance.
    pub(crate) fn to_generated_code(&self, name: &str) -> String {
        let mut generated_code = String::new();
        // Generate vertex reflection map function.
        let vertex_reflection_name = "Vertex";
        generated_code.push_str(
            &self
                .vertex
                .reflection_map
                .to_generated_code(&vertex_reflection_name),
        );
        generated_code.push('\n');

        // Generate fragment reflection map function.
        let fragment_reflection_name = "Fragment";
        generated_code.push_str(
            &self
                .fragment
                .reflection_map
                .to_generated_code(&fragment_reflection_name),
        );
        generated_code.push('\n');

        generated_code.push_str("      pub const VERTEX_SOURCE: &'static str = r###\"");
        generated_code.push_str(&self.vertex.source);
        generated_code.push_str("\n\"###;\n");

        generated_code.push_str("      pub const FRAGMENT_SOURCE: &'static str = r###\"");
        generated_code.push_str(&self.fragment.source);
        generated_code.push_str("\n\"###;\n");

        generated_code
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use naga::{
        ShaderStage,
        back::glsl::{self, PipelineOptions, Version},
        front::wgsl,
        valid::{Capabilities, ValidationFlags, Validator},
    };

    use super::ReflectionMap;

    const EXAMPLE_WGSL_SHADER: &str = r#"
struct Config {
    width: u32,
}

struct StripInstance {
    @location(0) xy: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(1)
var<uniform> config: Config;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    instance: StripInstance,
) -> VertexOutput {
    var out: VertexOutput;
    let width = config.width;
    return out;
}

@group(0) @binding(0)
var alphas_texture: texture_2d<u32>;

@group(0) @binding(2)
var clip_input_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex1_size = textureDimensions(alphas_texture);
    let tex2_size = textureDimensions(clip_input_texture);
    let width = config.width;
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
"#;

    #[test]
    fn construct_reflection_map_from_vertex() {
        let module = wgsl::parse_str(EXAMPLE_WGSL_SHADER).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::default())
            .validate(&module)
            .unwrap();

        // Write the vertex glsl shader.
        let options = naga::back::glsl::Options {
            version: Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };

        let pipeline_options = PipelineOptions {
            entry_point: "vs_main".into(),
            shader_stage: ShaderStage::Vertex,
            multiview: None,
        };

        let mut glsl_vs = String::new();

        let mut w = glsl::Writer::new(
            &mut glsl_vs,
            &module,
            &info,
            &options,
            &pipeline_options,
            Default::default(),
        )
        .unwrap();
        let reflection_info = w.write().unwrap();

        let result = ReflectionMap::new(reflection_info, &module.global_variables);

        // Assertions
        assert_eq!(result.uniforms.len(), 1);
        assert_eq!(
            result.uniforms.get("config"),
            Some(&"Config_block_0Vertex".into())
        );
        assert_eq!(result.texture_mapping.len(), 0);
    }

    #[test]
    fn construct_reflection_map_from_fragment() {
        let module = wgsl::parse_str(EXAMPLE_WGSL_SHADER).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::default())
            .validate(&module)
            .unwrap();
        let options = naga::back::glsl::Options {
            version: Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline_options = PipelineOptions {
            entry_point: "fs_main".into(),
            shader_stage: ShaderStage::Fragment,
            multiview: None,
        };
        let mut glsl_vs = String::new();
        let mut w = glsl::Writer::new(
            &mut glsl_vs,
            &module,
            &info,
            &options,
            &pipeline_options,
            Default::default(),
        )
        .unwrap();
        let reflection_info = w.write().unwrap();

        let result = ReflectionMap::new(reflection_info, &module.global_variables);
        // Assertions
        assert_eq!(result.uniforms.len(), 1);
        assert_eq!(
            result.uniforms.get("config"),
            Some(&"Config_block_0Fragment".into())
        );
        assert_eq!(result.texture_mapping.len(), 2);
        assert_eq!(
            result.texture_mapping.get("alphas_texture"),
            Some(&"_group_0_binding_0_fs".into())
        );
        assert_eq!(
            result.texture_mapping.get("clip_input_texture"),
            Some(&"_group_0_binding_2_fs".into())
        );
    }

    #[test]
    fn test_generating_reflection_map() {
        let mut texture_mapping = BTreeMap::new();
        texture_mapping.insert(
            "alphas_texture".to_string(),
            "_group_0_binding_0_fs".to_string(),
        );
        texture_mapping.insert(
            "clip_input_texture".to_string(),
            "_group_0_binding_2_fs".to_string(),
        );

        let mut uniforms = BTreeMap::new();
        uniforms.insert("config".to_string(), "Config_block_0Fragment".to_string());

        let reflection_map = ReflectionMap {
            texture_mapping,
            uniforms,
        };

        let generated_code = reflection_map.to_generated_code("gen_reflection");

        let expected_code = r#"fn gen_reflection() -> ReflectionMap<&'static str> {
  let mut uniforms = alloc::collections::btree_map::BTreeMap::new();
  uniforms.insert("config", "Config_block_0Fragment");
  let mut texture_mapping = alloc::collections::btree_map::BTreeMap::new();
  texture_mapping.insert("alphas_texture", "_group_0_binding_0_fs");
  texture_mapping.insert("clip_input_texture", "_group_0_binding_2_fs");
  ReflectionMap {
    uniforms,
    texture_mapping,
  }
}
"#;

        // Assert the generated code matches the expected output
        assert_eq!(generated_code, expected_code);
    }
}

pub(crate) fn snake_to_pascal(input: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    
    for c in input.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    
    result
}
