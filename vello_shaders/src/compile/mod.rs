// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use naga::front::wgsl;
use naga::valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags};
use naga::{AddressSpace, ArraySize, ImageClass, Module, StorageAccess, WithSpan};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use thiserror::Error;

pub mod permutations;
pub mod preprocess;

#[cfg(feature = "msl")]
pub mod msl;

use crate::types::{BindType, BindingInfo, WorkgroupBufferInfo};

pub type Result<T> = std::result::Result<T, Error>;
pub type CoalescedResult<T> = std::result::Result<T, ErrorVec>;

#[derive(Error, Debug)]
pub struct ErrorVec(Vec<Error>);

#[derive(Error, Debug)]
#[error("{source} ({name}) {msg}")]
pub struct Error {
    name: String,
    msg: String,
    source: InnerError,
}

#[derive(Error, Debug)]
enum InnerError {
    #[error("failed to parse shader")]
    Parse(#[from] wgsl::ParseError),

    #[error("failed to validate shader")]
    Validate(#[from] WithSpan<ValidationError>),

    #[error("missing entry point function")]
    EntryPointNotFound,
}

impl fmt::Display for ErrorVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in self.0.iter() {
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

impl Error {
    fn new(wgsl: &str, name: &str, error: impl Into<InnerError>) -> Self {
        let source = error.into();
        Self {
            name: name.to_owned(),
            msg: source.emit_msg(wgsl, &format!("({name} preprocessed)")),
            source,
        }
    }
}

impl InnerError {
    fn emit_msg(&self, wgsl: &str, name: &str) -> String {
        match self {
            Self::Parse(e) => e.emit_to_string_with_path(wgsl, name),
            Self::Validate(e) => e.emit_to_string_with_path(wgsl, name),
            _ => String::default(),
        }
    }
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
    #[cfg_attr(
        target_pointer_width = "64",
        allow(
            clippy::result_large_err,
            reason = "Deferred: This is a cold code path."
        )
    )]
    pub fn new(name: &str, source: String, entry_point: &str) -> Result<Self> {
        let module = wgsl::parse_str(&source).map_err(|error| Error::new(&source, name, error))?;
        let module_info = naga::valid::Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .map_err(|error| Error::new(&source, name, error))?;
        let (entry_index, entry) = module
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.name.as_str() == entry_point)
            .ok_or(Error::new(&source, name, InnerError::EntryPointNotFound))?;
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
                        naga::TypeInner::Scalar(scalar)
                        | naga::TypeInner::Vector { scalar, .. }
                        | naga::TypeInner::Matrix { scalar, .. }
                        | naga::TypeInner::Atomic(scalar) => scalar.width as u32,
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
                if let ImageClass::Storage { access, .. } = class
                    && access.contains(StorageAccess::STORE)
                {
                    resource.ty = BindType::Image;
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
        Ok(Self {
            source: postprocess(&source),
            module,
            module_info,
            workgroup_size,
            bindings,
            workgroup_buffers,
        })
    }

    /// Same as [`ShaderInfo::from_dir`] but uses the default shader directory provided by [`shader_dir`].
    pub fn from_default() -> CoalescedResult<HashMap<String, Self>> {
        Self::from_dir(shader_dir())
    }

    pub fn from_dir(shader_dir: impl AsRef<Path>) -> CoalescedResult<HashMap<String, Self>> {
        use std::fs;
        let shader_dir = shader_dir.as_ref();
        let permutation_map =
            if let Ok(permutations_source) = fs::read_to_string(shader_dir.join("permutations")) {
                permutations::parse(&permutations_source)
            } else {
                HashMap::default()
            };
        //println!("{permutation_map:?}");
        let imports = preprocess::get_imports(shader_dir);
        let mut errors = vec![];
        let mut info = HashMap::default();
        let defines: HashSet<_> = HashSet::default();
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
                    let contents = fs::read_to_string(shader_dir.join(&file_name))
                        .unwrap_or_else(|_| panic!("Couldn't read shader {shader_name} contents"));
                    if let Some(permutations) = permutation_map.get(shader_name) {
                        for permutation in permutations {
                            let mut defines = defines.clone();
                            defines.extend(permutation.defines.iter().cloned());
                            let source =
                                preprocess::preprocess(&contents, shader_name, &defines, &imports);
                            match Self::new(&permutation.name, source, "main") {
                                Ok(shader_info) => {
                                    info.insert(permutation.name.clone(), shader_info);
                                }
                                Err(e) => {
                                    errors.push(e);
                                }
                            }
                        }
                    } else {
                        let source =
                            preprocess::preprocess(&contents, shader_name, &defines, &imports);
                        match Self::new(shader_name, source, "main") {
                            Ok(shader_info) => {
                                info.insert(shader_name.to_string(), shader_info);
                            }
                            Err(e) => {
                                errors.push(e);
                            }
                        }
                    }
                }
            }
        }
        if !errors.is_empty() {
            Err(ErrorVec(errors))
        } else {
            Ok(info)
        }
    }
}

// TODO: This is a workaround for gfx-rs/wgpu#5476. Since naga can't handle the `enable` directive,
// we allow its use in other WGSL compilers using our own "#enable" post-process directive. Remove
// this mechanism once naga supports the directive.
fn postprocess(wgsl: &str) -> String {
    let mut output = String::with_capacity(wgsl.len());
    for line in wgsl.lines() {
        if line.starts_with("//__#enable") {
            output.push_str(&line["//__#".len()..]);
        } else {
            output.push_str(line);
        }
        output.push('\n');
    }
    output
}

/// Returns the absolute path to the directory containing the WGSL shaders.
///
/// The path is determined at compile time and is likely only valid on the compiling machine.
// NOTE: Embedding build environment info into the code makes reproducible builds trickier.
pub fn shader_dir() -> &'static PathBuf {
    static SHADER_DIR: OnceLock<PathBuf> = OnceLock::new();
    SHADER_DIR.get_or_init(|| manifest_dir().join("shader"))
}

// In a regular cargo build the manifest directory is simply given by CARGO_MANIFEST_DIR.
//
// Skia, an external consumer of this crate, uses Bazel rules to compile Rust code. Due to
// limitations in Bazel's rust support, Skia maintains its own build definitions
// (https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/bazel/external/vello/BUILD.bazel).
//
// Because of the current setup, Bazel sets CARGO_MANIFEST_DIR to the workspace root instead of the
// actual crate being built. This could be improved but until then, we work around this by allowing
// the absolute path to the vello_shader crate's manifest to be specified using the
// `UNSTABLE_BAZEL_VELLO_SHADERS_CRATE_MANIFEST_PATH` build script environment variable.
//
// This should never be set when using cargo.
fn manifest_dir() -> PathBuf {
    use std::env;
    env::var_os("UNSTABLE_BAZEL_VELLO_SHADERS_CRATE_MANIFEST_PATH")
        .and_then(|p| Path::new(&p).parent().map(|p| p.to_owned()))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .to_path_buf()
}
