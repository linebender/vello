// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Proc-macros for testing `vello_cpu` and `vello_hybrid`.

// How much different renderers are allowed to deviate from the reference images
// per color component. A value of 0 means that it must be an exact match, i.e.
// each pixel must have the exact same value as the reference pixel. A value of 1
// means that the absolute difference of each component of a pixel must not be higher
// than 1. For example, if the target pixel is (233, 43, 64, 100), then permissible
// values are (232, 43, 65, 101) or (233, 42, 64, 100), but not (231, 43, 64, 100).
const DEFAULT_CPU_U8_TOLERANCE: u8 = 0;
const DEFAULT_HYBRID_TOLERANCE: u8 = 1;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, ItemFn, Token, parse_macro_input};

#[derive(Debug)]
enum Attribute {
    /// A key-value-like argument.
    KeyValue { key: Ident, expr: Expr },
    /// A flag-like argument.
    Flag(Ident),
}

fn is_flag(key: &Ident) -> bool {
    matches!(
        key.to_string().as_str(),
        "transparent" | "skip_cpu" | "skip_hybrid" | "no_ref" | "ignore"
    )
}

impl Parse for Attribute {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let key = input.parse()?;

        if is_flag(&key) {
            Ok(Self::Flag(key))
        } else {
            // Skip equality token.
            let _: Token![=] = input.parse()?;
            let expr = input.parse()?;
            Ok(Self::KeyValue { key, expr })
        }
    }
}

#[derive(Debug)]
struct AttributeInput {
    args: Punctuated<Attribute, Token![,]>,
}

impl Parse for AttributeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            args: input.parse_terminated(Attribute::parse, Token![,])?,
        })
    }
}

struct Arguments {
    /// The width of the scene.
    width: u16,
    /// The height of the scene.
    height: u16,
    /// The (additional) maximum tolerance for how much two pixels are allowed to deviate from each other
    /// when comparing to the reference images. Some renderers already have an existing tolerance
    /// (see the constants at the top of the file), this value will simply be added
    /// to the currently existing threshold. See the top of the file for an explanation of
    /// how exactly the tolerance is interpreted.
    cpu_tolerance: u8,
    /// Same as above, but for the hybrid renderer.
    hybrid_tolerance: u8,
    /// Whether the background should be transparent (the default is white).
    transparent: bool,
    /// Whether the test should not be run on the CPU (`vello_cpu`).
    skip_cpu: bool,
    /// Whether the test should not be run on the GPU (`vello_hybrid`).
    skip_hybrid: bool,
    /// Whether no reference image should actually be created (for tests that only check
    /// for panics, but are not interested in the actual output).
    no_ref: bool,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            cpu_tolerance: 0,
            hybrid_tolerance: 0,
            transparent: false,
            skip_cpu: false,
            skip_hybrid: false,
            no_ref: false,
        }
    }
}

/// Create a new vello snapshot test.
#[proc_macro_attribute]
pub fn v_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as AttributeInput);

    let input_fn = parse_macro_input!(item as ItemFn);

    let input_fn_name = input_fn.sig.ident.clone();
    let u8_fn_name = Ident::new(&format!("{}_cpu_u8", input_fn_name), input_fn_name.span());
    let hybrid_fn_name = Ident::new(&format!("{}_hybrid", input_fn_name), input_fn_name.span());

    // TODO: Tests with the same names in different modules can clash, see
    // https://github.com/linebender/vello/pull/925#discussion_r2070710362.
    // We should take the module path into consideration for naming the tests.
    
    let input_fn_name_str = input_fn_name.to_string();
    let u8_fn_name_str = u8_fn_name.to_string();
    let hybrid_fn_name_str = hybrid_fn_name.to_string();

    let Arguments {
        width,
        height,
        mut cpu_tolerance,
        mut hybrid_tolerance,
        transparent,
        skip_cpu,
        mut skip_hybrid,
        no_ref,
    } = parse_args(&attrs);

    // These tests currently don't work with `vello_hybrid`.
    skip_hybrid |= {
        input_fn_name_str.contains("clip")
            || input_fn_name_str.contains("compose")
            || input_fn_name_str.contains("gradient")
            || input_fn_name_str.contains("image")
            || input_fn_name_str.contains("layer")
            || input_fn_name_str.contains("mask")
            || input_fn_name_str.contains("mix")
            || input_fn_name_str.contains("opacity")
    };

    let empty_snippet = quote! {};
    let ignore_snippet = quote! {#[ignore]};

    let ignore_hybrid = if skip_hybrid {
        ignore_snippet.clone()
    } else {
        empty_snippet.clone()
    };
    let ignore_cpu = if skip_cpu {
        ignore_snippet.clone()
    } else {
        empty_snippet.clone()
    };
    cpu_tolerance += DEFAULT_CPU_U8_TOLERANCE;
     hybrid_tolerance += DEFAULT_HYBRID_TOLERANCE;

    let expanded = quote! {
        #input_fn

        #ignore_cpu
        #[test]
        fn #u8_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use vello_cpu::RenderContext;

            let mut ctx = get_ctx::<RenderContext>(#width, #height, #transparent);
            #input_fn_name(&mut ctx);
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #u8_fn_name_str, #cpu_tolerance, true);
            }
        }

        #ignore_hybrid
        #[test]
        fn #hybrid_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use vello_hybrid::Scene;

            let mut ctx = get_ctx::<Scene>(#width, #height, #transparent);
            #input_fn_name(&mut ctx);
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #hybrid_fn_name_str, #hybrid_tolerance, false);
            }
        }
    };

    expanded.into()
}

fn parse_args(attribute_input: &AttributeInput) -> Arguments {
    let mut args = Arguments::default();

    for arg in &attribute_input.args {
        match arg {
            Attribute::KeyValue { key, expr, .. } => {
                let key_str = key.to_string();
                match key_str.as_str() {
                    "width" => args.width = parse_int_lit(expr, "width"),
                    "height" => args.height = parse_int_lit(expr, "height"),
                    #[allow(clippy::cast_possible_truncation, reason = "user-supplied value")]
                    "cpu_tolerance" => args.cpu_tolerance = parse_int_lit(expr, "cpu_tolerance") as u8,
                    #[allow(clippy::cast_possible_truncation, reason = "user-supplied value")]
                    "hybrid_tolerance" => args.hybrid_tolerance = parse_int_lit(expr, "hybrid_tolerance") as u8,
                    _ => panic!("unknown pair attribute {}", key_str),
                }
            }
            Attribute::Flag(flag_ident) => {
                let flag_str = flag_ident.to_string();
                match flag_str.as_str() {
                    "transparent" => args.transparent = true,
                    "skip_cpu" => args.skip_cpu = true,
                    "skip_hybrid" => args.skip_hybrid = true,
                    "no_ref" => args.no_ref = true,
                    "ignore" => {
                        args.skip_cpu = true;
                        args.skip_hybrid = true;
                    }
                    _ => panic!("unknown flag attribute {}", flag_str),
                }
            }
        }
    }

    args
}

fn parse_int_lit(expr: &Expr, name: &str) -> u16 {
    if let Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(lit_int),
        ..
    }) = expr
    {
        lit_int.base10_parse::<u16>().unwrap()
    } else {
        panic!("invalid expression supplied to `{name}`")
    }
}
