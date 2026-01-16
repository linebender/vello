// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use naga::{
    ShaderStage,
    back::glsl::{self, PipelineOptions, Version},
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

use crate::types::{CompiledGlsl, ReflectionMap, Stage};

#[allow(
    dead_code,
    reason = "False positive as compile_wgsl_shader is used at build time."
)]
/// Compiles the given wgsl source into GLSL using [naga].
pub(crate) fn compile_wgsl_shader(
    source: &str,
    vertex_entry: &str,
    fragment_entry: &str,
) -> CompiledGlsl {
    let module = wgsl::parse_str(source).unwrap();

    let info = Validator::new(ValidationFlags::all(), Capabilities::default())
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
        .unwrap();

    let options = glsl::Options {
        version: Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };

    let mut glsl_vs = String::new();
    let reflection_vs = {
        let pipeline_options = PipelineOptions {
            entry_point: vertex_entry.into(),
            shader_stage: ShaderStage::Vertex,
            multiview: None,
        };

        let mut w_vs = glsl::Writer::new(
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
        ReflectionMap::new(
            w_vs.write().expect("failed to write vertex."),
            &module.global_variables,
        )
    };

    let pipeline_options = PipelineOptions {
        entry_point: fragment_entry.into(),
        shader_stage: ShaderStage::Fragment,
        multiview: None,
    };
    let mut glsl_fs = String::new();
    let mut w_fs = glsl::Writer::new(
        &mut glsl_fs,
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
    let reflection_fs = ReflectionMap::new(
        w_fs.write().expect("failed to write fragment."),
        &module.global_variables,
    );

    CompiledGlsl {
        vertex: Stage {
            source: glsl_vs,
            reflection_map: reflection_vs,
        },
        fragment: Stage {
            source: glsl_fs,
            reflection_map: reflection_fs,
        },
    }
}
