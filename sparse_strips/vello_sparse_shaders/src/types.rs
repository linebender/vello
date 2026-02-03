// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types that are used to build the generated code file.

#![allow(
    dead_code,
    reason = "False positives as this module is used at build time."
)]

use naga::{Arena, GlobalVariable, back::glsl::ReflectionInfo};
use std::collections::BTreeMap;

/// `ReflectionMap` is a simplified derivative of [`naga::back::glsl::ReflectionInfo`], mapping the
/// wgsl variable names to the generated glsl names.
#[derive(Debug)]
pub(crate) struct ReflectionMap {
    /// Mapping of wgsl texture identifier to the generated glsl identifier.
    // TODO: It may make sense to pass through the sampler type. E.g. `sampler2D` or `usampler2D`.
    texture_mapping: BTreeMap<String, String>,
    /// Mapping of wgsl uniform identifier to the generated glsl identifier.
    uniforms: BTreeMap<String, String>,
}

impl ReflectionMap {
    /// Create a new `ReflectionMap` given the [`naga`] compile info.
    pub(crate) fn new(info: ReflectionInfo, global_vars: &Arena<GlobalVariable>) -> Self {
        debug_assert_eq!(info.varying.len(), 0, "unimplemented");
        debug_assert_eq!(info.push_constant_items.len(), 0, "unimplemented");
        let mut texture_mapping = BTreeMap::default();
        let mut uniforms = BTreeMap::default();

        for (glsl_name, texture_handles) in info.texture_mapping {
            if let Ok(wgsl_var) = global_vars.try_get(texture_handles.texture)
                && let Some(wgsl_name) = &wgsl_var.name
            {
                texture_mapping.insert(wgsl_name.clone(), glsl_name);
            }
        }

        for (handle, glsl_name) in info.uniforms {
            if let Ok(wgsl_var) = global_vars.try_get(handle)
                && let Some(wgsl_name) = &wgsl_var.name
            {
                uniforms.insert(wgsl_name.clone(), glsl_name);
            }
        }

        Self {
            texture_mapping,
            uniforms,
        }
    }
}

#[derive(Debug)]
/// A glsl vertex or fragment shader stage, with reflection info.
pub(crate) struct Stage {
    /// glsl source code.
    pub(crate) source: String,
    /// Reflection info from wgsl identifiers to the glsl identifiers.
    pub(crate) reflection_map: ReflectionMap,
}

#[derive(Debug)]
/// Compiled glsl with reflection info for mapping between the wgsl source of truth and generated
/// glsl.
pub(crate) struct CompiledGlsl {
    /// Vertex stage.
    pub(crate) vertex: Stage,
    /// Fragment stage.
    pub(crate) fragment: Stage,
}

impl CompiledGlsl {
    /// Generate Rust modules that contain vertex and fragment source code. Uniforms and texture
    /// identifier mappings are also generated from the wgsl identifier to the compiled naga
    /// identifier.
    pub(crate) fn to_generated_code(&self, shader_name: &str) -> String {
        let mut code = format!("/// Compiled glsl for `{shader_name}.wgsl`\n");
        code.push_str(&format!("pub mod {shader_name} {{\n"));
        code.push_str(r#"    #![allow(missing_docs, reason="No metadata to generate precise documentation for generated code.")]"#);
        code.push_str("\n\n");

        code.push_str("    pub const VERTEX_SOURCE: &str = r###\"");
        code.push_str(&self.vertex.source);
        code.push_str("\"###;\n\n");

        // Add Vertex stage identifier mapping if there are any.
        if self.vertex.reflection_map.uniforms.len()
            + self.vertex.reflection_map.texture_mapping.len()
            != 0
        {
            code.push_str("    pub mod vertex {\n");
            for (wgsl_name, glsl_name) in &self.vertex.reflection_map.uniforms {
                let const_name = wgsl_name.to_uppercase();
                code.push_str(&format!(
                    "        pub const {const_name}: &str = \"{glsl_name}\";\n"
                ));
            }

            for (wgsl_name, glsl_name) in &self.vertex.reflection_map.texture_mapping {
                let const_name = wgsl_name.to_uppercase();
                code.push_str(&format!(
                    "        pub const {const_name}: &str = \"{glsl_name}\";\n"
                ));
            }
            code.push_str("    }\n");
        }

        code.push_str("    pub const FRAGMENT_SOURCE: &str = r###\"");
        code.push_str(&self.fragment.source);
        code.push_str("\"###;\n");

        // Add Fragment stage identifier mapping if there are any.
        if self.fragment.reflection_map.uniforms.len()
            + self.fragment.reflection_map.texture_mapping.len()
            != 0
        {
            code.push_str("    pub mod fragment {\n");
            for (wgsl_name, glsl_name) in &self.fragment.reflection_map.uniforms {
                let const_name = wgsl_name.to_uppercase();
                code.push_str(&format!(
                    "        pub const {const_name}: &str = \"{glsl_name}\";\n"
                ));
            }

            for (wgsl_name, glsl_name) in &self.fragment.reflection_map.texture_mapping {
                let const_name = wgsl_name.to_uppercase();
                code.push_str(&format!(
                    "        pub const {const_name}: &str = \"{glsl_name}\";\n"
                ));
            }
            code.push_str("    }\n");
        }
        code.push('}');

        code
    }
}

#[cfg(test)]
mod tests {
    use naga::{
        ShaderStage,
        back::glsl::{self, PipelineOptions, Version},
        front::wgsl,
        proc::BoundsCheckPolicies,
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
        let options = glsl::Options {
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
            BoundsCheckPolicies::default(),
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
        let options = glsl::Options {
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
            BoundsCheckPolicies::default(),
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
}
