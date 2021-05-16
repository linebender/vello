//! Generation of Rust derive functions for encoding.

use quote::{format_ident, quote};

use crate::layout::{LayoutModule, LayoutTypeDef};
use crate::parse::{GpuScalar, GpuType};

pub fn gen_derive(module: &LayoutModule) -> proc_macro2::TokenStream {
    let mut ts = proc_macro2::TokenStream::new();
    let module_name = format_ident!("{}", module.name);
    for name in &module.def_names {
        let def = module.defs.get(name).unwrap();
        ts.extend(gen_derive_def(name, def.0.size, &def.1));
    }
    quote! {
        mod #module_name {
            pub trait HalfToLeBytes {
                fn to_le_bytes(&self) -> [u8; 2];
            }

            impl HalfToLeBytes for half::f16 {
                fn to_le_bytes(&self) -> [u8; 2] {
                    self.to_bits().to_le_bytes()
                }
            }

            #ts
        }
    }
}

fn gen_derive_def(name: &str, size: usize, def: &LayoutTypeDef) -> proc_macro2::TokenStream {
    let name_id = format_ident!("{}", name);
    match def {
        LayoutTypeDef::Struct(fields) => {
            let mut gen_fields = proc_macro2::TokenStream::new();
            let mut encode_fields = proc_macro2::TokenStream::new();
            for (field_name, offset, ty) in fields {
                let field_name_id = format_ident!("{}", field_name);
                let gen_ty = gen_derive_ty(&ty.ty);
                let gen_field = quote! {
                    pub #field_name_id: #gen_ty,
                };
                gen_fields.extend(gen_field);

                encode_fields.extend(gen_encode_field(field_name, *offset, &ty.ty));
            }
            quote! {
                pub struct #name_id {
                    #gen_fields
                }

                impl crate::encoder::Encode for #name_id {
                    fn fixed_size() -> usize {
                        #size
                    }
                    fn encode_to(&self, buf: &mut [u8]) {
                        #encode_fields
                    }
                }
            }
        }
        LayoutTypeDef::Enum(variants) => {
            let mut gen_variants = proc_macro2::TokenStream::new();
            let mut cases = proc_macro2::TokenStream::new();
            for (variant_ix, (variant_name, payload)) in variants.iter().enumerate() {
                let variant_id = format_ident!("{}", variant_name);
                let field_tys = payload.iter().map(|(_offset, ty)| gen_derive_ty(&ty.ty));
                let variant = quote! {
                    #variant_id(#(#field_tys),*),
                };
                gen_variants.extend(variant);

                let mut args = Vec::new();
                let mut field_encoders = proc_macro2::TokenStream::new();
                let mut tag_field = None;
                for (i, (offset, ty)) in payload.iter().enumerate() {
                    let field_id = format_ident!("f{}", i);
                    if matches!(ty.ty, GpuType::Scalar(GpuScalar::TagFlags)) {
                        tag_field = Some(field_id.clone());
                    } else {
                        let field_encoder = quote! {
                            #field_id.encode_to(&mut buf[#offset..]);
                        };
                        field_encoders.extend(field_encoder);
                    }
                    args.push(field_id);
                }
                let tag = variant_ix as u32;
                let tag_encode = match tag_field {
                    None => quote! {
                        buf[0..4].copy_from_slice(&#tag.to_le_bytes());
                    },
                    Some(tag_field) => quote! {
                        buf[0..4].copy_from_slice(&(#tag | ((*#tag_field as u32) << 16)).to_le_bytes());
                    },
                };
                let case = quote! {
                    #name_id::#variant_id(#(#args),*) => {
                        #tag_encode
                        #field_encoders
                    }
                };
                cases.extend(case);
            }
            quote! {
                pub enum #name_id {
                    #gen_variants
                }

                impl crate::encoder::Encode for #name_id {
                    fn fixed_size() -> usize {
                        #size
                    }
                    fn encode_to(&self, buf: &mut [u8]) {
                        match self {
                            #cases
                        }
                    }
                }
            }
        }
    }
}

/// Generate a Rust type.
fn gen_derive_ty(ty: &GpuType) -> proc_macro2::TokenStream {
    match ty {
        GpuType::Scalar(s) => gen_derive_scalar_ty(s),
        GpuType::Vector(s, len) => {
            let scalar = gen_derive_scalar_ty(s);
            quote! { [#scalar; #len] }
        }
        GpuType::InlineStruct(name) => {
            let name_id = format_ident!("{}", name);
            quote! { #name_id }
        }
        GpuType::Ref(ty) => {
            let gen_ty = gen_derive_ty(ty);
            quote! { crate::encoder::Ref<#gen_ty> }
        }
    }
}

fn gen_derive_scalar_ty(ty: &GpuScalar) -> proc_macro2::TokenStream {
    match ty {
        GpuScalar::F16 => quote!(half::f16),
        GpuScalar::F32 => quote!(f32),
        GpuScalar::I8 => quote!(i8),
        GpuScalar::I16 => quote!(i16),
        GpuScalar::I32 => quote!(i32),
        GpuScalar::U8 => quote!(u8),
        GpuScalar::U16 => quote!(u16),
        GpuScalar::U32 => quote!(u32),
        GpuScalar::TagFlags => quote!(u16),
    }
}

fn gen_encode_field(name: &str, offset: usize, ty: &GpuType) -> proc_macro2::TokenStream {
    let name_id = format_ident!("{}", name);
    match ty {
        // encoding of flags into tag word is handled elsewhere
        GpuType::Scalar(GpuScalar::TagFlags) => quote! {},
        GpuType::Scalar(s) => {
            let end = offset + s.size();
            quote! {
                buf[#offset..#end].copy_from_slice(&self.#name_id.to_le_bytes());
            }
        }
        GpuType::Vector(s, len) => {
            let size = s.size();
            quote! {
                for i in 0..#len {
                    let offset = #offset + i * #size;
                    buf[offset..offset + #size].copy_from_slice(&self.#name_id[i].to_le_bytes());
                }
            }
        }
        GpuType::Ref(_) => {
            quote! {
                buf[#offset..#offset + 4].copy_from_slice(&self.#name_id.offset().to_le_bytes());
            }
        }
        _ => {
            quote! {
                &self.#name_id.encode_to(&mut buf[#offset..]);
            }
        }
    }
}
