//! Parsing of the source

extern crate proc_macro;

use std::collections::HashSet;

use syn::{
    Expr, ExprLit, Fields, FieldsNamed, FieldsUnnamed, GenericArgument, ItemEnum, ItemStruct, Lit,
    PathArguments, TypeArray, TypePath,
};

/// A scalar that can be represented in a packed data structure.
#[derive(Clone, Copy, PartialEq)]
pub enum GpuScalar {
    F16,
    F32,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
}

/// An algebraic datatype.
#[derive(Clone)]
pub enum GpuType {
    Scalar(GpuScalar),
    Vector(GpuScalar, usize),
    /// Used mostly for the body of enum variants.
    InlineStruct(String),
    Ref(Box<GpuType>),
}

pub struct GpuEnum {
    pub name: String,
    pub variants: Vec<(String, Vec<GpuType>)>,
}

pub enum GpuTypeDef {
    Struct(String, Vec<(String, GpuType)>),
    Enum(GpuEnum),
}

pub struct GpuModule {
    pub name: String,
    pub attrs: HashSet<String>,
    pub defs: Vec<GpuTypeDef>,
}

impl GpuScalar {
    fn from_syn(ty: &syn::Type) -> Option<Self> {
        ty_as_single_ident(ty).and_then(|ident| match ident.as_str() {
            "f32" => Some(GpuScalar::F32),
            "f16" => Some(GpuScalar::F16),
            "i8" => Some(GpuScalar::I8),
            "i16" => Some(GpuScalar::I16),
            "i32" => Some(GpuScalar::I32),
            "u8" => Some(GpuScalar::U8),
            "u16" => Some(GpuScalar::U16),
            "u32" => Some(GpuScalar::U32),
            _ => None,
        })
    }

    /// Size of scalar type.
    ///
    /// This is arguably a concern at the layout level, not syntax, but it's here because
    /// it's not likely to be variable, so reduces the total number of types.
    pub fn size(self) -> usize {
        match self {
            GpuScalar::F32 | GpuScalar::I32 | GpuScalar::U32 => 4,
            GpuScalar::I8 | GpuScalar::U8 => 1,
            GpuScalar::F16 | GpuScalar::I16 | GpuScalar::U16 => 2,
        }
    }
}

impl GpuType {
    fn from_syn(ty: &syn::Type) -> Result<Self, String> {
        //println!("gputype {:#?}", ty);
        if let Some(scalar) = GpuScalar::from_syn(ty) {
            return Ok(GpuType::Scalar(scalar));
        }
        if let Some(name) = ty_as_single_ident(ty) {
            // Note: we're not doing any validation here.
            return Ok(GpuType::InlineStruct(name));
        }
        match ty {
            syn::Type::Path(TypePath {
                path: syn::Path { segments, .. },
                ..
            }) => {
                if segments.len() == 1 {
                    let seg = &segments[0];
                    if seg.ident == "Ref" {
                        if let PathArguments::AngleBracketed(args) = &seg.arguments {
                            if args.args.len() == 1 {
                                if let GenericArgument::Type(inner) = &args.args[0] {
                                    let inner_ty = GpuType::from_syn(inner)?;
                                    return Ok(GpuType::Ref(Box::new(inner_ty)));
                                }
                            }
                        }
                    }
                }
                Err("unknown path case".into())
            }
            syn::Type::Array(TypeArray { elem, len, .. }) => {
                if let Some(elem) = GpuScalar::from_syn(&elem) {
                    if let Some(len) = expr_int_lit(len) {
                        // maybe sanity-check length here
                        Ok(GpuType::Vector(elem, len))
                    } else {
                        Err("can't deal with variable length scalar arrays".into())
                    }
                } else {
                    Err("can't deal with non-scalar arrays".into())
                }
            }
            _ => Err("unknown type".into()),
        }
    }
}

impl GpuTypeDef {
    fn from_syn(item: &syn::Item) -> Result<Self, String> {
        match item {
            syn::Item::Struct(ItemStruct {
                ident,
                fields: Fields::Named(FieldsNamed { named, .. }),
                ..
            }) => {
                let mut fields = Vec::new();
                for field in named {
                    let field_ty = GpuType::from_syn(&field.ty)?;
                    let field_name = field.ident.as_ref().ok_or("need name".to_string())?;
                    fields.push((field_name.to_string(), field_ty));
                }
                Ok(GpuTypeDef::Struct(ident.to_string(), fields))
            }
            syn::Item::Enum(ItemEnum {
                ident, variants, ..
            }) => {
                let mut v = Vec::new();
                for variant in variants {
                    let vname = variant.ident.to_string();
                    let mut fields = Vec::new();
                    if let Fields::Unnamed(FieldsUnnamed { unnamed, .. }) = &variant.fields {
                        for field in unnamed {
                            fields.push(GpuType::from_syn(&field.ty)?);
                        }
                    }
                    v.push((vname, fields));
                }
                let en = GpuEnum {
                    name: ident.to_string(),
                    variants: v,
                };
                Ok(GpuTypeDef::Enum(en))
            }
            _ => {
                eprintln!("{:#?}", item);
                Err("unknown item".into())
            }
        }
    }

    pub fn name(&self) -> &str {
        match self {
            GpuTypeDef::Struct(name, _) => name,
            GpuTypeDef::Enum(en) => &en.name,
        }
    }
}

impl GpuModule {
    pub fn from_syn(module: &syn::ItemMod) -> Result<Self, String> {
        let name = module.ident.to_string();
        let mut attrs = HashSet::new();
        for attr in &module.attrs {
            if let Some(id) = path_as_single_ident(&attr.path) {
                attrs.insert(id.to_owned());
            }
        }
        let mut defs = Vec::new();
        if let Some((_brace, items)) = &module.content {
            for item in items {
                let def = GpuTypeDef::from_syn(item)?;
                defs.push(def);
            }
        }
        Ok(GpuModule { name, attrs, defs })
    }
}

fn path_as_single_ident(path: &syn::Path) -> Option<String> {
    if path.segments.len() == 1 {
        let seg = &path.segments[0];
        if seg.arguments == PathArguments::None {
            return Some(seg.ident.to_string());
        }
    }
    None
}

fn ty_as_single_ident(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Path(TypePath { path, .. }) = ty {
        path_as_single_ident(path)
    } else {
        None
    }
}

fn expr_int_lit(e: &Expr) -> Option<usize> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Int(lit_int),
        ..
    }) = e
    {
        lit_int.base10_parse().ok()
    } else {
        None
    }
}
