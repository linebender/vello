//! Proc-macros for testing `vello_cpu` and `vello_hybrid`.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, ItemFn, Token, parse_macro_input};

#[derive(Debug)]
enum AttributeArg {
    Pair {
        key: Ident,
        eq_token: Token![=],
        expr: Expr,
    },
    Flag(Ident),
}

fn is_flag(key: &Ident) -> bool {
    matches!(key.to_string().as_str(), "transparent")
}

impl Parse for AttributeArg {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let key = input.parse()?;

        if is_flag(&key) {
            Ok(Self::Flag(key))
        } else {
            let eq_token = input.parse()?;
            let expr = input.parse()?;
            Ok(Self::Pair {
                key,
                eq_token,
                expr,
            })
        }
    }
}

#[derive(Debug)]
struct AttributeInput {
    args: Punctuated<AttributeArg, Token![,]>,
}

impl Parse for AttributeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            args: input.parse_terminated(AttributeArg::parse, Token![,])?,
        })
    }
}

/// Create a new vello snapshot test.
#[proc_macro_attribute]
pub fn v_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as AttributeInput);

    let mut width = 100;
    let mut height = 100;
    let mut threshold = 0;
    let mut transparent = false;

    for arg in &attrs.args {
        match arg {
            AttributeArg::Pair { key, expr, .. } => {
                let key_str = key.to_string();
                match key_str.as_str() {
                    "width" => {
                        if let Some(w) = parse_int_lit(expr) {
                            width = w;
                        } else {
                            panic!("invalid expression supplied to `width`");
                        }
                    }
                    "height" => {
                        if let Some(h) = parse_int_lit(expr) {
                            height = h;
                        } else {
                            panic!("invalid expression supplied to `height`");
                        }
                    }
                    "threshold" => {
                        if let Some(t) = parse_int_lit(expr) {
                            threshold = t;
                        } else {
                            panic!("invalid expression supplied to `threshold`");
                        }
                    }
                    _ => panic!("unknown pair attribute {}", key_str),
                }
            }
            AttributeArg::Flag(flag_ident) => {
                let flag_str = flag_ident.to_string();
                match flag_str.as_str() {
                    "transparent" => {
                        transparent = true;
                    }
                    _ => panic!("unknown flag attribute {}", flag_str),
                }
            }
        }
    }

    let input_fn = parse_macro_input!(item as ItemFn);
    let input_fn_name = input_fn.sig.ident.clone();
    let input_fn_name_str = input_fn_name.to_string();

    let u8_fn_name = Ident::new(&format!("{}_u8", input_fn_name), input_fn_name.span());
    let hybrid_fn_name = Ident::new(&format!("{}_hybrid", input_fn_name), input_fn_name.span());
    
    let hybrid = if input_fn_name.to_string() == "filled_unaligned_rect".to_string() {
        quote! {
            #[test]
            fn #hybrid_fn_name() {
                use crate::util::{
                    check_ref, get_ctx_inner
                };
                use vello_hybrid::Scene;
    
                let mut ctx = get_ctx_inner::<Scene>(#width, #height, #transparent);
                #input_fn_name(&mut ctx);
                check_ref(&ctx, #input_fn_name_str);
            }
        }
    }   else { 
        quote! {}
    };

    let expanded = quote! {
        #input_fn

        #[test]
        fn #u8_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };

            let mut ctx = get_ctx(#width, #height, #transparent);
            #input_fn_name(&mut ctx);
            check_ref(&ctx, #input_fn_name_str);
        }
        
        #hybrid
    };

    expanded.into()
}

fn parse_int_lit(expr: &Expr) -> Option<u16> {
    if let Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(lit_int),
        ..
    }) = expr
    {
        lit_int.base10_parse::<u16>().ok()
    } else {
        None
    }
}
