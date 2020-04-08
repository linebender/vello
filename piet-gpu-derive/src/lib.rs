mod parse;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

use parse::GpuModule;

#[proc_macro]
pub fn piet_gpu(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemMod);
    //println!("input: {:#?}", input);
    let module = GpuModule::from_syn(&input).unwrap();
    let expanded = quote! {};
    expanded.into()
}
