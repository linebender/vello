mod derive;
mod glsl;
mod layout;
mod parse;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use layout::LayoutModule;
use parse::GpuModule;

#[proc_macro]
pub fn piet_gpu(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemMod);
    //println!("input: {:#?}", input);
    let module = GpuModule::from_syn(&input).unwrap();
    let layout = LayoutModule::from_gpu(&module);
    let glsl = glsl::gen_glsl(&layout);
    let gen_gpu_fn = format_ident!("gen_gpu_{}", layout.name);
    let mut expanded = quote! {
        fn #gen_gpu_fn() -> String {
            #glsl.into()
        }
    };
    if layout.rust_encode {
        expanded.extend(derive::gen_derive(&layout));
    }
    expanded.into()
}
