// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(rustdoc::private_intra_doc_links, reason = "not a public-facing crate")]
#![allow(rustdoc::broken_intra_doc_links, reason = "not a public-facing crate")]

//! Proc-macros for testing `vello_cpu` and `vello_hybrid`.

mod bench;
mod test;

// How much renderers are allowed to deviate from the reference images
// per color component. A value of 0 means that it must be an exact match, i.e.
// each pixel must have the exact same value as the reference pixel. A value of 1
// means that the absolute difference of each component of a pixel must not be higher
// than 1. For example, if the target pixel is (233, 43, 64, 100), then permissible
// values are (232, 43, 65, 101) or (233, 42, 64, 100), but not (231, 43, 64, 100).
// If we set this to 1 for CPU_U8, we would need to adjust it manually for nearly all
// bilinear image test cases.
const DEFAULT_CPU_U8_TOLERANCE: u8 = 2;
const DEFAULT_SIMD_TOLERANCE: u8 = 1;
const DEFAULT_CPU_F32_TOLERANCE: u8 = 0;
const DEFAULT_HYBRID_TOLERANCE: u8 = 1;

use crate::bench::vello_bench_inner;
use crate::test::vello_test_inner;
use proc_macro::TokenStream;
use std::fmt::Display;
use std::str::FromStr;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, Token};

#[derive(Debug)]
enum Attribute {
    /// A key-value-like argument.
    KeyValue { key: Ident, expr: Expr },
    /// A flag-like argument.
    Flag(Ident),
}

impl Parse for Attribute {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let key = input.parse()?;

        let is_flag = !input.peek(Token![=]);

        if is_flag {
            Ok(Self::Flag(key))
        } else {
            // Skip equality token.
            let _: Token![=] = input.parse()?;
            let expr = input.parse()?;
            Ok(Self::KeyValue { key, expr })
        }
    }
}

/// Create a new Vello snapshot test.
/// See [`test::Arguments`] for documentation of the arguments, which are comma-separated.
/// Boolean flags are set on their own, and others are in the form of key-value pairs.
///
/// ## Example
/// ```ignore
/// use vello_dev_macros::vello_test;
/// use vello_api::kurbo::Rect;
/// use vello_api::color::palette::css::REBECCA_PURPLE;
///
/// #[vello_test(width = 10, height = 10)]
/// fn rectangle_above_viewport(ctx: &mut impl Renderer) {
///     let rect = Rect::new(2.0, -5.0, 8.0, 8.0);
///
///     ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
///     ctx.fill_rect(&rect);
/// }
/// ```
#[proc_macro_attribute]
pub fn vello_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    vello_test_inner(attr, item)
}

/// Create a new Vello benchmark for fine rasterization.
/// This macro expects a function hat takes a `Bencher` and `Fine` as input, and will generate one benchmark
/// for each possible instantiation of `Fine`.
///
/// ## Example
/// ```ignore
/// #[vello_bench]
/// pub fn transparent_short<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
///     let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE.with_alpha(0.3)));
///     let width = 32;
///
///     fill_single(&paint, &[], width, b, default_blend(), fine);
/// }
/// ```
#[proc_macro_attribute]
pub fn vello_bench(attr: TokenStream, item: TokenStream) -> TokenStream {
    vello_bench_inner(attr, item)
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

fn parse_int_lit<N>(expr: &Expr, name: &str) -> N
where
    N: FromStr,
    N::Err: Display,
{
    if let Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(lit_int),
        ..
    }) = expr
    {
        lit_int.base10_parse::<N>().unwrap()
    } else {
        panic!("invalid expression supplied to `{name}`")
    }
}

fn parse_string_lit(expr: &Expr, name: &str) -> String {
    if let Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit_str),
        ..
    }) = expr
    {
        lit_str.value()
    } else {
        panic!("invalid expression supplied to `{name}`")
    }
}
