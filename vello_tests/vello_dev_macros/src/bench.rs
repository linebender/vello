// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

pub(crate) fn vello_bench_inner(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);

    let input_fn_name = input_fn.sig.ident.clone();
    let input_fn_name_str = input_fn.sig.ident.to_string();
    let inner_fn_name = Ident::new(&format!("{input_fn_name}_inner"), input_fn_name.span());

    input_fn.sig.ident = inner_fn_name.clone();

    let expanded = quote! {
        #input_fn

        pub fn #input_fn_name(registry: &mut crate::harness::Registry) {
            use vello_cpu::fine::{F32Kernel, Fine, U8Kernel};
            use vello_common::fearless_simd::Simd;
            use vello_cpu::Level;

            fn get_bench_name(suffix1: &str, suffix2: &str) -> String {
                let module_path = module_path!();

                let module = module_path
                    .split("::")
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join("/");

                format!("{}/{}_{}", module, suffix1, suffix2)
            }

            fn run_u8<S: Simd>(b: &mut Bencher, simd: S) {
                let mut fine = Fine::<S, U8Kernel>::new(simd, crate::fine::BENCH_WIDTH);
                #inner_fn_name(b, &mut fine);
            }

            fn run_f32<S: Simd>(b: &mut Bencher, simd: S) {
                let mut fine = Fine::<S, F32Kernel>::new(simd, crate::fine::BENCH_WIDTH);
                #inner_fn_name(b, &mut fine);
            }

            fn register_variants<S: Simd + Copy + 'static>(
                registry: &mut crate::harness::Registry,
                name: &str,
                suffix: &str,
                simd: S,
            ) {
                registry.add(get_bench_name(name, &format!("u8_{suffix}")), move |b| {
                    run_u8(b, simd);
                });
                registry.f32(|registry| {
                    registry.add(get_bench_name(name, &format!("f32_{suffix}")), move |b| {
                        run_f32(b, simd);
                    });
                });
            }

            registry.non_simd(|registry| {
                register_variants(
                    registry,
                    &#input_fn_name_str,
                    "scalar",
                    vello_common::fearless_simd::Fallback::new(),
                );
            });

            #[cfg(target_arch = "aarch64")]
            if let Some(neon) = Level::new().as_neon() {
                register_variants(registry, &#input_fn_name_str, "neon", neon);
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(sse4_2) = Level::new().as_sse4_2() {
                register_variants(registry, &#input_fn_name_str, "sse4_2", sse4_2);
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(avx2) = Level::new().as_avx2() {
                register_variants(registry, &#input_fn_name_str, "avx2", avx2);
            }

            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                register_variants(
                    registry,
                    &#input_fn_name_str,
                    "wasm_simd128",
                    Level::new().as_wasm_simd128().unwrap(),
                );
            }
        }
    };

    expanded.into()
}
