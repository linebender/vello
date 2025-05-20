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
    let inner_fn_name = Ident::new(&format!("{}_inner", input_fn_name), input_fn_name.span());

    input_fn.sig.ident = inner_fn_name.clone();

    let expanded = quote! {
        #input_fn

        pub fn #input_fn_name(c: &mut criterion::Criterion) {
            use vello_cpu::fine::Fine;
            use vello_common::coarse::WideTile;
            use vello_common::tile::Tile;

            fn get_bench_name(suffix1: &str, suffix2: &str) -> String {
                let module_path = module_path!();

                let module = module_path
                    .split("::")
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join("/");

                format!("{}/{}_{}", module, suffix1, suffix2)
            }

            c.bench_function(&get_bench_name(&#input_fn_name_str, "u8_scalar"), |b| {
                let mut fine = Fine::<u8>::new(WideTile::WIDTH, Tile::HEIGHT);
                #inner_fn_name(b, &mut fine);
            });

            c.bench_function(&get_bench_name(&#input_fn_name_str, "f32_scalar"), |b| {
                let mut fine = Fine::<f32>::new(WideTile::WIDTH, Tile::HEIGHT);
                #inner_fn_name(b, &mut fine);
            });
        }
    };

    expanded.into()
}
