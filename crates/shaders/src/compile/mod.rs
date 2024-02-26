// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use {
    naga::{
        front::wgsl,
        valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags},
        AddressSpace, ArraySize, ImageClass, Module, StorageAccess, WithSpan,
    },
    std::{
        collections::{HashMap, HashSet},
        path::Path,
    },
    thiserror::Error,
};

pub mod permutations;
pub mod preprocess;

#[cfg(feature = "msl")]
pub mod msl;

use crate::types::{BindType, BindingInfo, WorkgroupBufferInfo};

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to parse shader: {0}")]
    Parse(#[from] wgsl::ParseError),

    #[error("failed to validate shader: {0}")]
    Validate(#[from] WithSpan<ValidationError>),

    #[error("missing entry point function")]
    EntryPointNotFound,
}

#[derive(Debug)]
pub struct ShaderInfo {
    pub source: String,
    pub module: Module,
    pub module_info: ModuleInfo,
    pub workgroup_size: [u32; 3],
    pub bindings: Vec<BindingInfo>,
    pub workgroup_buffers: Vec<WorkgroupBufferInfo>,
}

impl ShaderInfo {
    pub fn new(source: String, entry_point: &str) -> Result<ShaderInfo, Error> {
        let module = wgsl::parse_str(&source)?;
        let module_info = naga::valid::Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)?;
        let (entry_index, entry) = module
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.name.as_str() == entry_point)
            .ok_or(Error::EntryPointNotFound)?;
        let mut bindings = vec![];
        let mut workgroup_buffers = vec![];
        let mut wg_buffer_idx = 0;
        let entry_info = module_info.get_entry_point(entry_index);
        for (var_handle, var) in module.global_variables.iter() {
            if entry_info[var_handle].is_empty() {
                continue;
            }
            let binding_ty = match module.types[var.ty].inner {
                naga::TypeInner::BindingArray { base, .. } => &module.types[base].inner,
                ref ty => ty,
            };
            let Some(binding) = &var.binding else {
                if var.space == AddressSpace::WorkGroup {
                    let index = wg_buffer_idx;
                    wg_buffer_idx += 1;
                    let size_in_bytes = match binding_ty {
                        naga::TypeInner::Array {
                            size: ArraySize::Constant(size),
                            stride,
                            ..
                        } => u32::from(*size) * stride,
                        naga::TypeInner::Struct { span, .. } => *span,
                        naga::TypeInner::Scalar { width, .. } => *width as u32,
                        naga::TypeInner::Vector { width, .. } => *width as u32,
                        naga::TypeInner::Matrix { width, .. } => *width as u32,
                        naga::TypeInner::Atomic { width, .. } => *width as u32,
                        _ => {
                            // Not a valid workgroup variable type. At least not one that is used
                            // in our shaders.
                            continue;
                        }
                    };
                    workgroup_buffers.push(WorkgroupBufferInfo {
                        size_in_bytes,
                        index,
                    });
                }
                continue;
            };
            let mut resource = BindingInfo {
                name: var.name.clone(),
                location: (binding.group, binding.binding),
                ty: BindType::Buffer,
            };
            if let naga::TypeInner::Image { class, .. } = &binding_ty {
                resource.ty = BindType::ImageRead;
                if let ImageClass::Storage { access, .. } = class {
                    if access.contains(StorageAccess::STORE) {
                        resource.ty = BindType::Image;
                    }
                }
            } else {
                resource.ty = BindType::BufReadOnly;
                match var.space {
                    AddressSpace::Storage { access } => {
                        if access.contains(StorageAccess::STORE) {
                            resource.ty = BindType::Buffer;
                        }
                    }
                    AddressSpace::Uniform => {
                        resource.ty = BindType::Uniform;
                    }
                    _ => {}
                }
            }
            bindings.push(resource);
        }
        bindings.sort_by_key(|res| res.location);
        let workgroup_size = entry.workgroup_size;
        Ok(ShaderInfo {
            source,
            module,
            module_info,
            workgroup_size,
            bindings,
            workgroup_buffers,
        })
    }

    pub fn from_dir(shader_dir: impl AsRef<Path>) -> HashMap<String, Self> {
        use std::fs;
        let shader_dir = shader_dir.as_ref();
        let permutation_map = if let Ok(permutations_source) =
            std::fs::read_to_string(shader_dir.join("permutations"))
        {
            permutations::parse(&permutations_source)
        } else {
            Default::default()
        };
        println!("{:?}", permutation_map);
        let imports = preprocess::get_imports(shader_dir);
        let mut info = HashMap::default();
        let mut defines = HashSet::default();
        defines.insert("full".to_string());
        for entry in shader_dir
            .read_dir()
            .expect("Can read shader import directory")
            .filter_map(move |e| {
                e.ok()
                    .filter(|e| e.path().extension().map(|e| e == "wgsl").unwrap_or(false))
            })
        {
            let file_name = entry.file_name();
            if let Some(name) = file_name.to_str() {
                let suffix = ".wgsl";
                if let Some(shader_name) = name.strip_suffix(suffix) {
                    let mut options = preprocess::Options::default();
                    if cfg!(feature = "force_rw_storage") {
                        // All bindings in the `fine` stage are readonly which means that a usage
                        // scope warning can never occur. Plus, the bindings must be declared as
                        // readonly to prevent a failed compilation due to uniformity analysis.
                        options.force_rw_storage = !name.starts_with("fine");
                    }
                    let contents = fs::read_to_string(shader_dir.join(&file_name))
                        .expect("Could read shader {shader_name} contents");
                    if let Some(permutations) = permutation_map.get(shader_name) {
                        for permutation in permutations {
                            let mut defines = defines.clone();
                            defines.extend(permutation.defines.iter().cloned());
                            let source =
                                preprocess::preprocess(&contents, &options, &defines, &imports);
                            let shader_info = Self::new(source.clone(), "main").unwrap();
                            info.insert(permutation.name.clone(), shader_info);
                        }
                    } else {
                        let source =
                            preprocess::preprocess(&contents, &options, &defines, &imports);
                        let shader_info = Self::new(source.clone(), "main").unwrap();
                        info.insert(shader_name.to_string(), shader_info);
                    }
                }
            }
        }
        info
    }
}
