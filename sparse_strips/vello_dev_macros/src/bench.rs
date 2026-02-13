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

        pub fn #input_fn_name(c: &mut criterion::Criterion) {
            use vello_cpu::fine::{Fine, U8Kernel, F32Kernel};
            use vello_common::coarse::WideTile;
            use vello_common::tile::Tile;
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

            fn run_integer<S: Simd>(b: &mut Bencher, simd: S) {
                let mut fine = Fine::<S, U8Kernel>::new(simd);
                #inner_fn_name(b, &mut fine);
            }

            fn run_float<S: Simd>(b: &mut Bencher, simd: S) {
                let mut fine = Fine::<S, F32Kernel>::new(simd);
                #inner_fn_name(b, &mut fine);
            }

            // Uncomment this to enable u8_scalar benchmarks.
            // c.bench_function(&get_bench_name(&#input_fn_name_str, "u8_scalar"), |b| {
            //     run_integer(b, vello_common::fearless_simd::Fallback::new());
            // });

            #[cfg(target_arch = "aarch64")]
            if let Some(neon) = Level::new().as_neon() {
                c.bench_function(&get_bench_name(&#input_fn_name_str, "u8_neon"), |b| {
                    run_integer(b, neon);
                });
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(sse4_2) = Level::new().as_sse4_2() {
                c.bench_function(&get_bench_name(&#input_fn_name_str, "u8_sse4_2"), |b| {
                    run_integer(b, sse4_2);
                });
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(avx2) = Level::new().as_avx2() {
                c.bench_function(&get_bench_name(&#input_fn_name_str, "u8_avx2"), |b| {
                    run_integer(b, avx2);
                });
            }

            // Uncomment this to enable f32_scalar benchmarks.
            // c.bench_function(&get_bench_name(&#input_fn_name_str, "f32_scalar"), |b| {
            //     run_float(b, vello_common::fearless_simd::Fallback::new());
            // });

            // f32 benchmarks are disabled because they are only of secondary importance and take a long time
            //
            // #[cfg(target_arch = "aarch64")]
            // if let Some(neon) = Level::new().as_neon() {
            //     c.bench_function(&get_bench_name(&#input_fn_name_str, "f32_neon"), |b| {
            //         run_float(b, neon);
            //     });
            // }

            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // if let Some(sse4_2) = Level::new().as_sse4_2() {
            //     c.bench_function(&get_bench_name(&#input_fn_name_str, "f32_sse4_2"), |b| {
            //         run_float(b, sse4_2);
            //     });
            // }

            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // if let Some(avx2) = Level::new().as_avx2() {
            //     c.bench_function(&get_bench_name(&#input_fn_name_str, "f32_avx2"), |b| {
            //         run_float(b, avx2);
            //     });
            // }
        }
    };

    expanded.into()
}
