use naga::back::msl;

use super::{BindType, ShaderInfo};

pub fn translate(shader: &ShaderInfo) -> Result<String, msl::Error> {
    let mut map = msl::EntryPointResourceMap::default();
    let mut buffer_index = 0u8;
    let mut image_index = 0u8;
    let mut binding_map = msl::BindingMap::default();
    for resource in &shader.bindings {
        let binding = naga::ResourceBinding {
            group: resource.location.0,
            binding: resource.location.1,
        };
        let mut target = msl::BindTarget::default();
        match resource.ty {
            BindType::Buffer | BindType::BufReadOnly | BindType::Uniform => {
                target.buffer = Some(buffer_index);
                buffer_index += 1;
            }
            BindType::Image | BindType::ImageRead => {
                target.texture = Some(image_index);
                image_index += 1;
            }
        }
        target.mutable = resource.ty.is_mutable();
        binding_map.insert(binding, target);
    }
    map.insert(
        "main".to_string(),
        msl::EntryPointResources {
            resources: binding_map,
            push_constant_buffer: None,
            sizes_buffer: Some(30),
        },
    );
    let options = msl::Options {
        lang_version: (2, 0),
        per_entry_point_map: map,
        inline_samplers: vec![],
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: false,
    };
    let (source, _) = msl::write_string(
        &shader.module,
        &shader.module_info,
        &options,
        &msl::PipelineOptions::default(),
    )?;
    Ok(source)
}
