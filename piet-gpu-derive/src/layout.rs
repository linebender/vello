//! Logic for layout of structures in memory.

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
                    let mut offset = if session.is_enum_variant(&en.name) {
                        4
                    } else {
                        0
                    };
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
        LayoutModule {
            name: module.name.clone(),
            def_names,
            defs: session.defs,
        }
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

    fn is_enum_variant(&self, name: &str) -> bool {
        self.enum_variants.contains(name)
    }
}

impl Size {
    fn new(size: usize) -> Size {
        let alignment = if size < 4 { 1 } else { 4 };
        Size { size, alignment }
    }

    fn new_struct(size: usize) -> Size {
        let alignment = 4;
        Size { size, alignment }
    }
}

fn align_padding(offset: usize, alignment: usize) -> usize {
    offset.wrapping_neg() & (alignment - 1)
}
