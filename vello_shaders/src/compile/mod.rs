use naga::{
    front::wgsl,
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags},
    AddressSpace, ImageClass, Module, StorageAccess, WithSpan,
};

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

pub mod permutations;
pub mod preprocess;

pub mod msl;

use crate::types::{BindType, BindingInfo};

#[derive(Debug)]
pub enum Error {
    Parse(wgsl::ParseError),
    Validate(WithSpan<ValidationError>),
    EntryPointNotFound,
}

impl From<wgsl::ParseError> for Error {
    fn from(e: wgsl::ParseError) -> Self {
        Self::Parse(e)
    }
}

impl From<WithSpan<ValidationError>> for Error {
    fn from(e: WithSpan<ValidationError>) -> Self {
        Self::Validate(e)
    }
}

#[derive(Debug)]
pub struct ShaderInfo {
    pub source: String,
    pub module: Module,
    pub module_info: ModuleInfo,
    pub workgroup_size: [u32; 3],
    pub bindings: Vec<BindingInfo>,
}

impl ShaderInfo {
    pub fn new(source: String, entry_point: &str) -> Result<ShaderInfo, Error> {
        let module = wgsl::parse_str(&source)?;
        let module_info = naga::valid::Validator::new(
            ValidationFlags::all() & !ValidationFlags::CONTROL_FLOW_UNIFORMITY,
            Capabilities::all(),
        )
        .validate(&module)?;
        let (entry_index, entry) = module
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.name.as_str() == entry_point)
            .ok_or(Error::EntryPointNotFound)?;
        let mut bindings = vec![];
        let entry_info = module_info.get_entry_point(entry_index);
        for (var_handle, var) in module.global_variables.iter() {
            if entry_info[var_handle].is_empty() {
                continue;
            }
            let Some(binding) = &var.binding else {
                continue;
            };
            let mut resource = BindingInfo {
                name: var.name.clone(),
                location: (binding.group, binding.binding),
                ty: BindType::Buffer,
            };
            let binding_ty = match module.types[var.ty].inner {
                naga::TypeInner::BindingArray { base, .. } => &module.types[base].inner,
                ref ty => ty,
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
        {
            let entry = entry.expect("Can continue reading shader import directory");
            if entry.file_type().unwrap().is_file() {
                let file_name = entry.file_name();
                if let Some(name) = file_name.to_str() {
                    let suffix = ".wgsl";
                    if let Some(shader_name) = name.strip_suffix(suffix) {
                        let contents = fs::read_to_string(shader_dir.join(&file_name))
                            .expect("Could read shader {shader_name} contents");
                        if let Some(permutations) = permutation_map.get(shader_name) {
                            for permutation in permutations {
                                let mut defines = defines.clone();
                                defines.extend(permutation.defines.iter().cloned());
                                let source = preprocess::preprocess(&contents, &defines, &imports);
                                let shader_info = Self::new(source.clone(), "main").unwrap();
                                info.insert(permutation.name.clone(), shader_info);
                            }
                        } else {
                            let source = preprocess::preprocess(&contents, &defines, &imports);
                            let shader_info = Self::new(source.clone(), "main").unwrap();
                            info.insert(shader_name.to_string(), shader_info);
                        }
                    }
                }
            }
        }
        info
    }
}
