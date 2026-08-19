// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use naga::{
    Module, ShaderStage,
    back::glsl::{self, PipelineOptions, Version},
    compact::KeepUnused,
    front::wgsl,
    valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
};

use crate::lint::lint;
use crate::types::{CompiledGlsl, ReflectionMap, Stage};

#[allow(
    dead_code,
    reason = "False positive as compile_wgsl_shader is used at build time."
)]
/// Compiles the given wgsl source into GLSL using [naga].
pub(crate) fn compile_wgsl_shader(
    source: &str,
    shader_name: &str,
    vertex_entry: &str,
    fragment_entry: &str,
) -> CompiledGlsl {
    let module = wgsl::parse_str(source).unwrap();

    lint(shader_name, &module);

    validate(&module);

    let options = glsl::Options {
        version: Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };

    CompiledGlsl {
        vertex: compile_stage(&module, vertex_entry, ShaderStage::Vertex, &options),
        fragment: compile_stage(&module, fragment_entry, ShaderStage::Fragment, &options),
    }
}

fn compile_stage(
    module: &Module,
    entry_point: &str,
    shader_stage: ShaderStage,
    options: &glsl::Options,
) -> Stage {
    let mut module = module.clone();
    module
        .entry_points
        .retain(|entry| entry.stage == shader_stage && entry.name == entry_point);
    naga::compact::compact(&mut module, KeepUnused::No);
    let info = validate(&module);

    let pipeline_options = PipelineOptions {
        entry_point: entry_point.into(),
        shader_stage,
        multiview: None,
    };
    let mut source = String::new();
    let mut writer = glsl::Writer::new(
        &mut source,
        &module,
        &info,
        options,
        &pipeline_options,
        naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
    )
    .unwrap();
    let reflection_map = ReflectionMap::new(
        writer.write().expect("failed to write shader stage."),
        &module.global_variables,
    );

    Stage {
        source,
        reflection_map,
    }
}

fn validate(module: &Module) -> ModuleInfo {
    Validator::new(ValidationFlags::all(), Capabilities::default())
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(module)
        .unwrap()
}
