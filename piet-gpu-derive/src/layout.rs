//! Logic for layout of structures in memory.

// This is fairly simple now, but there are some extensions that are likely:
// * Addition of f16 types
//   + These will probably have 2-byte alignments to support `packHalf2x16`
// * 1 byte tag values (so small struct fields can be packed along with tag)
// * (Possibly) reordering for better packing

use std::collections::{HashMap, HashSet};

use crate::parse::{GpuModule, GpuType, GpuTypeDef};

#[derive(Clone)]
pub struct LayoutType {
    size: Size,
    pub ty: GpuType,
}

#[derive(Clone)]
pub enum LayoutTypeDef {
    /// Name, offset, field type. Make a separate struct?
    Struct(Vec<(String, usize, LayoutType)>),
    Enum(Vec<(String, Vec<(usize, LayoutType)>)>),
}

pub struct LayoutModule {
    pub name: String,
    pub def_names: Vec<String>,
    pub defs: HashMap<String, (Size, LayoutTypeDef)>,
    enum_variants: HashSet<String>,

    /// Generate shader code to write the module.
    ///
    /// This is derived from the presence of the `gpu_write` attribute in the source module.
    pub gpu_write: bool,
    /// Generate Rust code to encode the module.
    ///
    /// This is derived from the presence of the `rust_encode` attribute in the source module.
    pub rust_encode: bool,
}

struct LayoutSession<'a> {
    enum_variants: HashSet<String>,
    orig_defs: HashMap<String, &'a GpuTypeDef>,
    defs: HashMap<String, (Size, LayoutTypeDef)>,
}

#[derive(Clone, Copy)]
pub struct Size {
    pub size: usize,
    alignment: usize,
}

impl LayoutType {
    fn from_gpu(ty: &GpuType, session: &mut LayoutSession) -> LayoutType {
        let size = session.get_size(ty);
        LayoutType {
            size,
            ty: ty.clone(),
        }
    }
}

impl LayoutTypeDef {
    // Maybe have a type representing the tuple?
    fn from_gpu(def: &GpuTypeDef, session: &mut LayoutSession) -> (Size, LayoutTypeDef) {
        match def {
            GpuTypeDef::Struct(_name, fields) => {
                // TODO: We want to be able to pack enums more tightly, in particular
                // other struct fields along with the enum tag. Structs in that category
                // (first field has an alignment < 4, serve as enum variant) will have a
                // different layout. This is why we're tracking `is_enum_variant`.
                //
                // But it's a bit of YAGNI for now; we're currently reserving 4 bytes for
                // the tag, so structure layout doesn't care.
                let mut offset = 0;
                let mut result = Vec::new();
                for field in fields {
                    let layout_ty = LayoutType::from_gpu(&field.1, session);
                    offset += align_padding(offset, layout_ty.size.alignment);
                    let size = layout_ty.size.size;
                    result.push((field.0.clone(), offset, layout_ty));
                    offset += size;
                }
                offset += align_padding(offset, 4);
                let size = Size::new_struct(offset);
                (size, LayoutTypeDef::Struct(result))
            }
            GpuTypeDef::Enum(en) => {
                let mut result = Vec::new();
                let mut max_offset = 0;
                for variant in &en.variants {
                    let mut r2 = Vec::new();
                    let mut offset = 4;
                    for field in &variant.1 {
                        let layout_ty = LayoutType::from_gpu(field, session);
                        offset += align_padding(offset, layout_ty.size.alignment);
                        let size = layout_ty.size.size;
                        r2.push((offset, layout_ty));
                        offset += size;
                    }
                    max_offset = max_offset.max(offset);
                    result.push((variant.0.clone(), r2));
                }
                max_offset += align_padding(max_offset, 4);
                let size = Size::new_struct(max_offset);
                (size, LayoutTypeDef::Enum(result))
            }
        }
    }
}

impl LayoutModule {
    pub fn from_gpu(module: &GpuModule) -> LayoutModule {
        let def_names = module
            .defs
            .iter()
            .map(|def| def.name().to_owned())
            .collect::<Vec<_>>();
        let mut session = LayoutSession::new(module);
        for def in &module.defs {
            let _ = session.layout_def(def.name());
        }
        let gpu_write = module.attrs.contains("gpu_write");
        let rust_encode = module.attrs.contains("rust_encode");
        LayoutModule {
            name: module.name.clone(),
            gpu_write,
            rust_encode,
            def_names,
            enum_variants: session.enum_variants,
            defs: session.defs,
        }
    }

    #[allow(unused)]
    pub fn is_enum_variant(&self, name: &str) -> bool {
        self.enum_variants.contains(name)
    }
}

impl<'a> LayoutSession<'a> {
    fn new(module: &GpuModule) -> LayoutSession {
        let mut orig_defs = HashMap::new();
        let mut enum_variants = HashSet::new();
        for def in &module.defs {
            orig_defs.insert(def.name().to_owned(), def.clone());
            if let GpuTypeDef::Enum(en) = def {
                for variant in &en.variants {
                    if let Some(GpuType::InlineStruct(name)) = variant.1.first() {
                        enum_variants.insert(name.clone());
                    }
                }
            }
        }
        LayoutSession {
            enum_variants,
            orig_defs,
            defs: HashMap::new(),
        }
    }

    /// Do layout of one def.
    ///
    /// This might be called recursively.
    /// Note: expect stack overflow for circular dependencies.
    fn layout_def(&mut self, name: &str) -> Size {
        if let Some(def) = self.defs.get(name) {
            return def.0;
        }
        let def = self.orig_defs.get(name).unwrap();
        let layout = LayoutTypeDef::from_gpu(def, self);
        let size = layout.0;
        self.defs.insert(name.to_owned(), layout);
        size
    }

    fn get_size(&mut self, ty: &GpuType) -> Size {
        match ty {
            GpuType::Scalar(scalar) => Size::new(scalar.size()),
            GpuType::Vector(scalar, len) => Size::new(scalar.size() * len),
            GpuType::Ref(_) => Size::new(4),
            GpuType::InlineStruct(name) => self.layout_def(name),
        }
    }

    #[allow(unused)]
    fn is_enum_variant(&self, name: &str) -> bool {
        self.enum_variants.contains(name)
    }
}

/// Compute coverage of fields.
///
/// Each element of the result represents a list of fields for one 4-byte chunk of
/// the struct layout. Inline structs are only included if requested.
pub fn struct_coverage(
    fields: &[(String, usize, LayoutType)],
    include_inline: bool,
) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    for (i, (_name, offset, ty)) in fields.iter().enumerate() {
        let size = match ty.ty {
            GpuType::Scalar(scalar) => scalar.size(),
            GpuType::Vector(scalar, len) => scalar.size() * len,
            GpuType::Ref(_) => 4,
            GpuType::InlineStruct(_) => {
                if include_inline {
                    4
                } else {
                    0
                }
            }
        };
        if size > 0 {
            for ix in (offset / 4)..(offset + size + 3) / 4 {
                if ix >= result.len() {
                    result.resize_with(ix + 1, Default::default);
                }
                result[ix].push(i);
            }
        }
    }
    result
}

impl Size {
    fn new(size: usize) -> Size {
        // Note: there is special case we could do better:
        // `(u8, u16, u8)`, where the alignment could be 1. However,
        // this case can also be solved by reordering.
        let alignment = size.min(4);
        Size { size, alignment }
    }

    fn new_struct(size: usize) -> Size {
        let alignment = 4;
        Size { size, alignment }
    }
}

fn align_padding(offset: usize, alignment: usize) -> usize {
    offset.wrapping_neg() & (alignment.max(1) - 1)
}
