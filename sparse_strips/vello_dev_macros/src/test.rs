// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    Attribute, AttributeInput, DEFAULT_CPU_F32_TOLERANCE, DEFAULT_CPU_U8_TOLERANCE,
    DEFAULT_HYBRID_TOLERANCE, parse_int_lit, parse_string_lit,
};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

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
    cpu_u8_tolerance: u8,
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
    /// A reason for ignoring a test.
    ignore_reason: Option<String>,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            cpu_u8_tolerance: 0,
            hybrid_tolerance: 0,
            transparent: false,
            skip_cpu: false,
            skip_hybrid: false,
            no_ref: false,
            ignore_reason: None,
        }
    }
}

pub(crate) fn vello_test_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as AttributeInput);

    let input_fn = parse_macro_input!(item as ItemFn);

    let input_fn_name = input_fn.sig.ident.clone();
    let u8_fn_name = Ident::new(&format!("{}_cpu_u8", input_fn_name), input_fn_name.span());
    let f32_fn_name = Ident::new(&format!("{}_cpu_f32", input_fn_name), input_fn_name.span());
    let hybrid_fn_name = Ident::new(&format!("{}_hybrid", input_fn_name), input_fn_name.span());
    let webgl_fn_name = Ident::new(
        &format!("{}_hybrid_webgl", input_fn_name),
        input_fn_name.span(),
    );

    // TODO: Tests with the same names in different modules can clash, see
    // https://github.com/linebender/vello/pull/925#discussion_r2070710362.
    // We should take the module path into consideration for naming the tests.

    let input_fn_name_str = input_fn_name.to_string();
    let u8_fn_name_str = u8_fn_name.to_string();
    let f32_fn_name_str = f32_fn_name.to_string();
    let hybrid_fn_name_str = hybrid_fn_name.to_string();
    let webgl_fn_name_str = webgl_fn_name.to_string();

    let Arguments {
        width,
        height,
        cpu_u8_tolerance,
        mut hybrid_tolerance,
        transparent,
        skip_cpu,
        mut skip_hybrid,
        ignore_reason,
        no_ref,
    } = parse_args(&attrs);

    // Wasm doesn't have access to the filesystem. For wasm, inline the snapshot bytes into the
    // binary.
    let reference_image_name = Ident::new(
        &format!(
            "{}_REFERENCE_IMAGE",
            input_fn_name.to_string().to_uppercase()
        ),
        input_fn_name.span(),
    );
    let reference_image_const = if !no_ref {
        quote! {
            #[cfg(target_arch = "wasm32")]
            const #reference_image_name: &[u8] = include_bytes!(
                concat!(env!("CARGO_MANIFEST_DIR"), "/snapshots/", #input_fn_name_str, ".png")
            );
            #[cfg(not(target_arch = "wasm32"))]
            const #reference_image_name: &[u8] = &[];
        }
    } else {
        quote! {
            const #reference_image_name: &[u8] = &[];
        }
    };

    let cpu_u8_tolerance = cpu_u8_tolerance + DEFAULT_CPU_U8_TOLERANCE;
    // Since f32 is our gold standard, we always require exact matches for this one.
    let cpu_f32_tolerance = DEFAULT_CPU_F32_TOLERANCE;
    hybrid_tolerance += DEFAULT_HYBRID_TOLERANCE;

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
            || input_fn_name_str.contains("blurred_rounded_rect")
    };

    let empty_snippet = quote! {};
    let ignore_snippet = if let Some(reason) = ignore_reason {
        quote! {#[ignore = #reason]}
    } else {
        quote! {#[ignore]}
    };

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

    let cpu_snippet = |fn_name: Ident,
                       fn_name_str: String,
                       tolerance: u8,
                       is_reference: bool,
                       render_mode: proc_macro2::TokenStream| {
        quote! {
            #ignore_cpu
            #[test]
            fn #fn_name() {
                use crate::util::{
                    check_ref, get_ctx
                };
                use vello_cpu::{RenderContext, RenderMode};

                let mut ctx = get_ctx::<RenderContext>(#width, #height, #transparent);
                #input_fn_name(&mut ctx);
                if !#no_ref {
                    check_ref(&ctx, #input_fn_name_str, #fn_name_str, #tolerance, #is_reference, #render_mode, #reference_image_name);
                }
            }
        }
    };

    let u8_snippet = cpu_snippet(
        u8_fn_name,
        u8_fn_name_str,
        cpu_u8_tolerance,
        false,
        quote! { RenderMode::OptimizeSpeed },
    );
    let f32_snippet = cpu_snippet(
        f32_fn_name,
        f32_fn_name_str,
        cpu_f32_tolerance,
        true,
        quote! { RenderMode::OptimizeQuality },
    );

    let expanded = quote! {
        #input_fn

        #reference_image_const

        #u8_snippet

        #f32_snippet

        #ignore_hybrid
        #[test]
        fn #hybrid_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use vello_hybrid::Scene;
            use vello_cpu::RenderMode;

            let mut ctx = get_ctx::<Scene>(#width, #height, #transparent);
            #input_fn_name(&mut ctx);
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #hybrid_fn_name_str, #hybrid_tolerance, false, RenderMode::OptimizeSpeed, #reference_image_name);
            }
        }

        #ignore_hybrid
        #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
        #[wasm_bindgen_test::wasm_bindgen_test]
        async fn #webgl_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use vello_hybrid::Scene;
            use vello_cpu::RenderMode;

            let mut ctx = get_ctx::<Scene>(#width, #height, #transparent);
            #input_fn_name(&mut ctx);
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #webgl_fn_name_str, #hybrid_tolerance, false, RenderMode::OptimizeSpeed, #reference_image_name);
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
                    "ignore" => {
                        args.skip_cpu = true;
                        args.skip_hybrid = true;
                        args.ignore_reason = Some(parse_string_lit(expr, "ignore"));
                    }
                    "width" => args.width = parse_int_lit(expr, "width"),
                    "height" => args.height = parse_int_lit(expr, "height"),
                    "cpu_u8_tolerance" => {
                        args.cpu_u8_tolerance = parse_int_lit(expr, "cpu_u8_tolerance")
                            .try_into()
                            .expect("value to fit for cpu_tolerance.");
                    }
                    "hybrid_tolerance" => {
                        args.hybrid_tolerance = parse_int_lit(expr, "hybrid_tolerance")
                            .try_into()
                            .expect("value to fit for hybrid_tolerance.");
                    }
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
