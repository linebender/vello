// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    Attribute, AttributeInput, DEFAULT_CPU_F32_TOLERANCE, DEFAULT_CPU_U8_TOLERANCE,
    DEFAULT_HYBRID_TOLERANCE, DEFAULT_SIMD_TOLERANCE, parse_int_lit, parse_string_lit,
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
    /// Whether the test should not be run on the multi-threaded CPU (`vello_cpu`).
    skip_multithreaded: bool,
    /// Whether the test should not be run on the GPU (`vello_hybrid`).
    skip_hybrid: bool,
    /// The maximum number of pixels that are allowed to completely deviate from the reference
    /// images. This attribute mainly exists because there are some test cases (like gradients),
    /// where, due to floating point inaccuracies, some pixels might land on a different color
    /// stop and thus yield a different value in CI.
    diff_pixels: u32,
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
            skip_multithreaded: false,
            skip_hybrid: false,
            no_ref: false,
            diff_pixels: 0,
            ignore_reason: None,
        }
    }
}

pub(crate) fn vello_test_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as AttributeInput);

    let input_fn = parse_macro_input!(item as ItemFn);

    let input_fn_name = input_fn.sig.ident.clone();
    let u8_fn_name_scalar = Ident::new(
        &format!("{input_fn_name}_cpu_u8_scalar"),
        input_fn_name.span(),
    );
    let f32_fn_name_scalar = Ident::new(
        &format!("{input_fn_name}_cpu_f32_scalar"),
        input_fn_name.span(),
    );
    let u8_fn_name_neon = Ident::new(
        &format!("{input_fn_name}_cpu_u8_neon"),
        input_fn_name.span(),
    );
    let f32_fn_name_neon = Ident::new(
        &format!("{input_fn_name}_cpu_f32_neon"),
        input_fn_name.span(),
    );
    let u8_fn_name_sse42 = Ident::new(
        &format!("{input_fn_name}_cpu_u8_sse42"),
        input_fn_name.span(),
    );
    let f32_fn_name_sse42 = Ident::new(
        &format!("{input_fn_name}_cpu_f32_sse42"),
        input_fn_name.span(),
    );
    let u8_fn_name_avx2 = Ident::new(
        &format!("{input_fn_name}_cpu_u8_avx2"),
        input_fn_name.span(),
    );
    let f32_fn_name_avx2 = Ident::new(
        &format!("{input_fn_name}_cpu_f32_avx2"),
        input_fn_name.span(),
    );
    let u8_fn_name_wasm = Ident::new(
        &format!("{input_fn_name}_cpu_u8_wasm"),
        input_fn_name.span(),
    );
    let f32_fn_name_wasm: Ident = Ident::new(
        &format!("{input_fn_name}_cpu_f32_wasm"),
        input_fn_name.span(),
    );
    let multithreaded_fn_name = Ident::new(
        &format!("{input_fn_name}_cpu_multithreaded"),
        input_fn_name.span(),
    );
    let hybrid_fn_name = Ident::new(&format!("{input_fn_name}_hybrid"), input_fn_name.span());
    let webgl_fn_name = Ident::new(
        &format!("{input_fn_name}_hybrid_webgl"),
        input_fn_name.span(),
    );

    // TODO: Tests with the same names in different modules can clash, see
    // https://github.com/linebender/vello/pull/925#discussion_r2070710362.
    // We should take the module path into consideration for naming the tests.

    let input_fn_name_str = input_fn_name.to_string();
    let u8_fn_name_str_scalar = u8_fn_name_scalar.to_string();
    let f32_fn_name_str_scalar = f32_fn_name_scalar.to_string();
    let u8_fn_name_str_neon = u8_fn_name_neon.to_string();
    let f32_fn_name_str_neon = f32_fn_name_neon.to_string();
    let u8_fn_name_str_sse42 = u8_fn_name_sse42.to_string();
    let f32_fn_name_str_sse42 = f32_fn_name_sse42.to_string();
    let u8_fn_name_str_avx2 = u8_fn_name_avx2.to_string();
    let f32_fn_name_str_avx2 = f32_fn_name_avx2.to_string();
    let u8_fn_name_wasm_str = u8_fn_name_wasm.to_string();
    let f32_fn_name_wasm_str = f32_fn_name_wasm.to_string();
    let multithreaded_fn_name_str = multithreaded_fn_name.to_string();
    let hybrid_fn_name_str = hybrid_fn_name.to_string();
    let webgl_fn_name_str = webgl_fn_name.to_string();

    let Arguments {
        width,
        height,
        cpu_u8_tolerance,
        mut hybrid_tolerance,
        transparent,
        skip_cpu,
        skip_multithreaded,
        mut skip_hybrid,
        ignore_reason,
        no_ref,
        diff_pixels,
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

    let cpu_u8_tolerance_scalar = cpu_u8_tolerance + DEFAULT_CPU_U8_TOLERANCE;
    let cpu_u8_tolerance_simd =
        cpu_u8_tolerance + DEFAULT_SIMD_TOLERANCE.max(DEFAULT_CPU_U8_TOLERANCE);

    // Since f32 is our gold standard, we always require exact matches for this one.
    let cpu_f32_tolerance_scalar = DEFAULT_CPU_F32_TOLERANCE;
    let cpu_f32_tolerance_simd = DEFAULT_CPU_F32_TOLERANCE + DEFAULT_SIMD_TOLERANCE;
    hybrid_tolerance += DEFAULT_HYBRID_TOLERANCE;

    // These tests currently don't work with `vello_hybrid`.
    skip_hybrid |= {
        input_fn_name_str.contains("layer_multiple_properties")
            || input_fn_name_str.contains("mask")
            || input_fn_name_str.contains("blurred_rounded_rect")
            || input_fn_name_str.contains("clip_clear")
            || input_fn_name_str.contains("mix_non_isolated")
            || input_fn_name_str.contains("compose_non_isolated")
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
    let ignore_hybrid_webgl = if skip_hybrid {
        ignore_snippet.clone()
    } else {
        empty_snippet.clone()
    };

    let cpu_snippet = |fn_name: Ident,
                       fn_name_str: String,
                       tolerance: u8,
                       is_reference: bool,
                       num_threads: u16,
                       // Need to pass as string, to avoid dependency on `fearless_simd` and also
                       // so that it works with proc_macros.
                       level: proc_macro2::TokenStream,
                       ignore: bool,
                       render_mode: proc_macro2::TokenStream| {
        // Use the name to infer if the test is running in the browser.
        let is_wasm_test = fn_name_str.contains("wasm");
        // WASM cannot create references, so force `is_reference` to be `false` unconditionally.
        let is_reference = if is_wasm_test { false } else { is_reference };
        let ignore_snippet = if ignore {
            ignore_snippet.clone()
        } else {
            quote! {}
        };

        let (cfg_attr, test_attr) = if is_wasm_test {
            assert_eq!(num_threads, 0, "wasm is single threaded");
            (
                quote! { #[cfg(target_arch = "wasm32")] },
                quote! { #[wasm_bindgen_test::wasm_bindgen_test] },
            )
        } else {
            (quote! {}, quote! { #[test] })
        };

        quote! {
            #cfg_attr
            #ignore_snippet
            #test_attr
            fn #fn_name() {
                use crate::util::{
                    check_ref, get_ctx
                };
                use vello_cpu::{RenderContext, RenderMode};

                let mut ctx = get_ctx::<RenderContext>(#width, #height, #transparent, #num_threads, #level, #render_mode);
                #input_fn_name(&mut ctx);
                ctx.flush();
                if !#no_ref {
                    check_ref(&ctx, #input_fn_name_str, #fn_name_str, #tolerance, #diff_pixels, #is_reference, #reference_image_name);
                }
            }
        }
    };

    #[cfg(target_arch = "aarch64")]
    let has_neon = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(target_arch = "aarch64"))]
    let has_neon = false;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    let has_sse42 = false;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let has_sse42 = std::arch::is_x86_feature_detected!("sse4.2");

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    let has_avx2 = false;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let has_avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");

    let wasm_simd_level = quote! {if cfg!(target_feature = "simd128") {
            "wasm_simd128"
        } else {
            "fallback"
        }
    };

    let u8_snippet = cpu_snippet(
        u8_fn_name_scalar,
        u8_fn_name_str_scalar,
        cpu_u8_tolerance_scalar,
        false,
        0,
        quote! {"fallback"},
        skip_cpu,
        quote! { RenderMode::OptimizeSpeed },
    );
    let f32_snippet = cpu_snippet(
        f32_fn_name_scalar,
        f32_fn_name_str_scalar,
        cpu_f32_tolerance_scalar,
        true,
        0,
        quote! {"fallback"},
        skip_cpu,
        quote! { RenderMode::OptimizeQuality },
    );
    let u8_snippet_wasm = cpu_snippet(
        u8_fn_name_wasm,
        u8_fn_name_wasm_str,
        cpu_u8_tolerance_scalar,
        false,
        0,
        wasm_simd_level.clone(),
        skip_cpu,
        quote! { RenderMode::OptimizeSpeed },
    );
    let f32_snippet_wasm = cpu_snippet(
        f32_fn_name_wasm,
        f32_fn_name_wasm_str,
        cpu_f32_tolerance_scalar,
        true,
        0,
        wasm_simd_level,
        skip_cpu,
        quote! { RenderMode::OptimizeQuality },
    );
    let multi_threaded_snippet = cpu_snippet(
        multithreaded_fn_name,
        multithreaded_fn_name_str,
        cpu_f32_tolerance_scalar,
        false,
        3,
        quote! {"fallback"},
        skip_cpu | skip_multithreaded,
        quote! { RenderMode::OptimizeQuality },
    );

    let neon_u8_snippet = cpu_snippet(
        u8_fn_name_neon,
        u8_fn_name_str_neon,
        cpu_u8_tolerance_simd,
        false,
        0,
        quote! {"neon"},
        skip_cpu | !has_neon,
        quote! { RenderMode::OptimizeSpeed },
    );

    let neon_f32_snippet = cpu_snippet(
        f32_fn_name_neon,
        f32_fn_name_str_neon,
        cpu_f32_tolerance_simd,
        false,
        0,
        quote! {"neon"},
        skip_cpu | !has_neon,
        quote! { RenderMode::OptimizeQuality },
    );

    let sse42_u8_snippet = cpu_snippet(
        u8_fn_name_sse42,
        u8_fn_name_str_sse42,
        cpu_u8_tolerance_simd,
        false,
        0,
        quote! {"sse42"},
        skip_cpu | !has_sse42,
        quote! { RenderMode::OptimizeSpeed },
    );

    let sse42_f32_snippet = cpu_snippet(
        f32_fn_name_sse42,
        f32_fn_name_str_sse42,
        cpu_f32_tolerance_simd,
        false,
        0,
        quote! {"sse42"},
        skip_cpu | !has_sse42,
        quote! { RenderMode::OptimizeQuality },
    );

    let avx2_u8_snippet = cpu_snippet(
        u8_fn_name_avx2,
        u8_fn_name_str_avx2,
        cpu_u8_tolerance_simd,
        false,
        0,
        quote! {"avx2"},
        skip_cpu | !has_avx2,
        quote! { RenderMode::OptimizeSpeed },
    );

    let avx2_f32_snippet = cpu_snippet(
        f32_fn_name_avx2,
        f32_fn_name_str_avx2,
        cpu_f32_tolerance_simd,
        false,
        0,
        quote! {"avx2"},
        skip_cpu | !has_avx2,
        quote! { RenderMode::OptimizeQuality },
    );

    let expanded = quote! {
        #input_fn

        #reference_image_const

        #u8_snippet

        #neon_u8_snippet

        #sse42_u8_snippet

        #avx2_u8_snippet

        #u8_snippet_wasm

        #f32_snippet

        #neon_f32_snippet

        #sse42_f32_snippet

        #avx2_f32_snippet

        #f32_snippet_wasm

        #multi_threaded_snippet

        #ignore_hybrid
        #[test]
        fn #hybrid_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use crate::renderer::HybridRenderer;
            use vello_cpu::RenderMode;

            let mut ctx = get_ctx::<HybridRenderer>(#width, #height, #transparent, 0, "fallback", RenderMode::OptimizeSpeed);
            #input_fn_name(&mut ctx);
            ctx.flush();
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #hybrid_fn_name_str, #hybrid_tolerance, #diff_pixels, false, #reference_image_name);
            }
        }

        #ignore_hybrid_webgl
        #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
        #[wasm_bindgen_test::wasm_bindgen_test]
        async fn #webgl_fn_name() {
            use crate::util::{
                check_ref, get_ctx
            };
            use crate::renderer::HybridRenderer;
            use vello_cpu::RenderMode;

            let mut ctx = get_ctx::<HybridRenderer>(#width, #height, #transparent, 0, "fallback", RenderMode::OptimizeSpeed);
            #input_fn_name(&mut ctx);
            ctx.flush();
            if !#no_ref {
                check_ref(&ctx, #input_fn_name_str, #webgl_fn_name_str, #hybrid_tolerance, #diff_pixels, false, #reference_image_name);
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
                        args.skip_multithreaded = true;
                        args.skip_hybrid = true;
                        args.ignore_reason = Some(parse_string_lit(expr, "ignore"));
                    }
                    "width" => args.width = parse_int_lit(expr, "width"),
                    "diff_pixels" => args.diff_pixels = parse_int_lit(expr, "diff_pixels"),
                    "height" => args.height = parse_int_lit(expr, "height"),
                    "cpu_u8_tolerance" => {
                        args.cpu_u8_tolerance = parse_int_lit::<u8>(expr, "cpu_u8_tolerance");
                    }
                    "hybrid_tolerance" => {
                        args.hybrid_tolerance = parse_int_lit::<u8>(expr, "hybrid_tolerance");
                    }
                    _ => panic!("unknown pair attribute {key_str}"),
                }
            }
            Attribute::Flag(flag_ident) => {
                let flag_str = flag_ident.to_string();
                match flag_str.as_str() {
                    "transparent" => args.transparent = true,
                    "skip_cpu" => args.skip_cpu = true,
                    "skip_multithreaded" => args.skip_multithreaded = true,
                    "skip_hybrid" => args.skip_hybrid = true,
                    "no_ref" => args.no_ref = true,
                    "ignore" => {
                        args.skip_cpu = true;
                        args.skip_multithreaded = true;
                        args.skip_hybrid = true;
                    }
                    _ => panic!("unknown flag attribute {flag_str}"),
                }
            }
        }
    }

    args
}
