// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{BindType, ShaderInfo};
use crate::types::msl::BindingIndex;
use naga::back::msl as naga_msl;

pub fn translate(shader: &ShaderInfo) -> Result<String, naga_msl::Error> {
    let mut map = naga_msl::EntryPointResourceMap::default();
    let mut idx_iter = BindingIndexIterator::default();
    let mut binding_map = naga_msl::BindingMap::default();
    for resource in &shader.bindings {
        let binding = naga::ResourceBinding {
            group: resource.location.0,
            binding: resource.location.1,
        };
        let mut target = naga_msl::BindTarget::default();
        match idx_iter.next(resource.ty) {
            BindingIndex::Buffer(idx) => {
                target.buffer = Some(idx);
            }
            BindingIndex::Texture(idx) => {
                target.texture = Some(idx);
            }
        }
        target.mutable = resource.ty.is_mutable();
        binding_map.insert(binding, target);
    }
    map.insert(
        "main".to_string(),
        naga_msl::EntryPointResources {
            resources: binding_map,
            push_constant_buffer: None,
            sizes_buffer: Some(30),
        },
    );
    let options = naga_msl::Options {
        lang_version: (2, 0),
        per_entry_point_map: map,
        inline_samplers: vec![],
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: true,
    };
    let (source, _) = naga_msl::write_string(
        &shader.module,
        &shader.module_info,
        &options,
        &naga_msl::PipelineOptions::default(),
    )?;
    Ok(source)
}

#[derive(Default)]
pub struct BindingIndexIterator {
    buffer_idx: u8,
    tex_idx: u8,
}

impl BindingIndexIterator {
    pub fn next(&mut self, ty: BindType) -> BindingIndex {
        match ty {
            BindType::Buffer | BindType::BufReadOnly | BindType::Uniform => {
                let idx = self.buffer_idx;
                self.buffer_idx += 1;
                assert!(self.buffer_idx > 0);
                BindingIndex::Buffer(idx)
            }
            BindType::Image | BindType::ImageRead => {
                let idx = self.tex_idx;
                self.tex_idx += 1;
                assert!(self.tex_idx > 0);
                BindingIndex::Texture(idx)
            }
        }
    }
}
