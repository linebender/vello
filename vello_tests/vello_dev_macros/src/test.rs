// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    DEFAULT_CPU_F32_TOLERANCE, DEFAULT_CPU_U8_TOLERANCE, DEFAULT_GPU_TOLERANCE,
    DEFAULT_SIMD_TOLERANCE,
};
use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::Parser;
use syn::{ItemFn, LitInt, LitStr, parse_macro_input};

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
    /// Same as above, but for the gpu renderer.
    gpu_tolerance: u8,
    /// Whether the background should be transparent (the default is white).
    transparent: bool,
    /// Whether the test should not be run on the CPU (`vello_cpu`).
    skip_cpu: bool,
    /// Whether the test should not be run on the multi-threaded CPU (`vello_cpu`).
    skip_multithreaded: bool,
    /// Whether the test should not be run on the GPU (`vello_gpu`).
    skip_gpu: bool,
    /// Whether the test should not be run using the WebGL backend.
    skip_webgl: bool,
    /// Whether only `vello_gpu` should run and generate the reference image.
    gpu_only: bool,
    /// Whether to additionally run `vello_gpu` with depth buffering disabled.
    gpu_no_depth: bool,
    /// The maximum number of pixels that are allowed to completely deviate from the reference
    /// images. This attribute mainly exists because there are some test cases (like gradients),
    /// where, due to floating point inaccuracies, some pixels might land on a different color
    /// stop and thus yield a different value in CI.
    diff_pixels: u32,
    /// Whether no reference image should actually be created (for tests that only check
    /// for panics, but are not interested in the actual output).
    no_ref: bool,
    /// Whether this is a glyph test and should generate the additional caching variants.
    glyph: bool,
    /// A reason for ignoring a test.
    ignore_reason: Option<String>,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            cpu_u8_tolerance: 0,
            gpu_tolerance: 0,
            transparent: false,
            skip_cpu: false,
            skip_multithreaded: false,
            skip_gpu: false,
            skip_webgl: false,
            gpu_only: false,
            gpu_no_depth: false,
            no_ref: false,
            glyph: false,
            diff_pixels: 0,
            ignore_reason: None,
        }
    }
}

#[derive(Clone, Copy)]
enum Pipeline {
    U8,
    F32,
}

impl Pipeline {
    const ALL: [Self; 2] = [Self::U8, Self::F32];

    fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::F32 => "f32",
        }
    }

    fn render_mode(self) -> TokenStream2 {
        match self {
            Self::U8 => quote! { vello_cpu::RenderMode::OptimizeSpeed },
            Self::F32 => quote! { vello_cpu::RenderMode::OptimizeQuality },
        }
    }
}

#[derive(Clone, Copy)]
enum CpuLevel {
    Scalar,
    Neon,
    Sse2,
    Sse42,
    Avx2,
    Avx512,
    Wasm,
}

impl CpuLevel {
    const ALL: [Self; 7] = [
        Self::Scalar,
        Self::Neon,
        Self::Sse2,
        Self::Sse42,
        Self::Avx2,
        Self::Avx512,
        Self::Wasm,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Sse2 => "sse2",
            Self::Sse42 => "sse42",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Wasm => "wasm",
        }
    }

    fn value(self) -> TokenStream2 {
        match self {
            Self::Scalar => quote! { "fallback" },
            Self::Neon => quote! { "neon" },
            Self::Sse2 => quote! { "sse2" },
            Self::Sse42 => quote! { "sse42" },
            Self::Avx2 => quote! { "avx2" },
            Self::Avx512 => quote! { "avx512" },
            Self::Wasm => quote! {
                if cfg!(target_feature = "simd128") {
                    "wasm_simd128"
                } else {
                    "fallback"
                }
            },
        }
    }

    fn tolerance(self, scalar: u8, simd: u8) -> TokenStream2 {
        match self {
            Self::Scalar => quote! { #scalar },
            Self::Wasm => quote! {
                if cfg!(target_feature = "simd128") {
                    #simd
                } else {
                    #scalar
                }
            },
            Self::Neon | Self::Sse2 | Self::Sse42 | Self::Avx2 | Self::Avx512 => {
                quote! { #simd }
            }
        }
    }

    fn is_available(self) -> bool {
        match self {
            Self::Scalar | Self::Wasm => true,
            Self::Neon => {
                #[cfg(target_arch = "aarch64")]
                return std::arch::is_aarch64_feature_detected!("neon");
                #[cfg(not(target_arch = "aarch64"))]
                return false;
            }
            Self::Sse2 => {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                return std::arch::is_x86_feature_detected!("sse2")
                    && std::arch::is_x86_feature_detected!("fxsr");
                #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                return false;
            }
            Self::Sse42 => {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                return std::arch::is_x86_feature_detected!("sse4.2");
                #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                return false;
            }
            Self::Avx2 => {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                return std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
                #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                return false;
            }
            Self::Avx512 => {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                return std::arch::is_x86_feature_detected!("avx512f")
                    || std::env::var_os("VELLO_TEST_AVX512").is_some();
                #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                return false;
            }
        }
    }
}

struct Tolerances {
    cpu_u8_scalar: u8,
    cpu_u8_simd: u8,
    cpu_f32_scalar: u8,
    cpu_f32_simd: u8,
    gpu: u8,
}

impl Tolerances {
    fn new(args: &Arguments) -> Self {
        Self {
            cpu_u8_scalar: args.cpu_u8_tolerance + DEFAULT_CPU_U8_TOLERANCE,
            cpu_u8_simd: args.cpu_u8_tolerance
                + DEFAULT_SIMD_TOLERANCE.max(DEFAULT_CPU_U8_TOLERANCE),
            cpu_f32_scalar: DEFAULT_CPU_F32_TOLERANCE,
            cpu_f32_simd: DEFAULT_CPU_F32_TOLERANCE + DEFAULT_SIMD_TOLERANCE,
            gpu: args.gpu_tolerance + DEFAULT_GPU_TOLERANCE,
        }
    }

    fn for_cpu(&self, pipeline: Pipeline, level: CpuLevel) -> TokenStream2 {
        let (scalar, simd) = match pipeline {
            Pipeline::U8 => (self.cpu_u8_scalar, self.cpu_u8_simd),
            Pipeline::F32 => (self.cpu_f32_scalar, self.cpu_f32_simd),
        };
        level.tolerance(scalar, simd)
    }
}

#[derive(Clone, Copy)]
enum CpuVariant {
    Pipeline { pipeline: Pipeline, level: CpuLevel },
    Multithreaded,
    Cached,
}

#[derive(Clone, Copy)]
enum GpuBackend {
    Wgpu,
    WebGl,
}

#[derive(Clone, Copy)]
struct GpuVariant {
    backend: GpuBackend,
    cached: bool,
    no_depth: bool,
}

enum Renderer {
    Cpu(CpuVariant),
    Gpu(GpuVariant),
}

struct TestCase {
    suffix: String,
    renderer: Renderer,
    tolerance: TokenStream2,
    is_reference: bool,
    ignore: bool,
}

impl Renderer {
    fn cached(&self) -> bool {
        match self {
            Self::Cpu(variant) => variant.is_cached(),
            Self::Gpu(variant) => variant.cached,
        }
    }
}

impl CpuVariant {
    fn config(self) -> (Pipeline, CpuLevel, u16) {
        match self {
            Self::Pipeline { pipeline, level } => (pipeline, level, 0),
            Self::Multithreaded => (Pipeline::F32, CpuLevel::Scalar, 3),
            Self::Cached => (Pipeline::F32, CpuLevel::Scalar, 0),
        }
    }

    fn is_cached(self) -> bool {
        matches!(self, Self::Cached)
    }

    fn resolve(self, args: &Arguments, tolerances: &Tolerances) -> TestCase {
        let (pipeline, level, _) = self.config();
        let (suffix, is_reference, ignore) = match self {
            Self::Pipeline { pipeline, level } => (
                format!("cpu_{}_{}", pipeline.name(), level.name()),
                matches!((pipeline, level), (Pipeline::F32, CpuLevel::Scalar)) && !args.gpu_only,
                args.skip_cpu || !level.is_available(),
            ),
            Self::Multithreaded => (
                "cpu_multithreaded".to_owned(),
                false,
                args.skip_cpu || args.skip_multithreaded,
            ),
            Self::Cached => (
                "cpu_f32_scalar_cached".to_owned(),
                !args.gpu_only,
                args.skip_cpu,
            ),
        };
        TestCase {
            suffix,
            renderer: Renderer::Cpu(self),
            tolerance: tolerances.for_cpu(pipeline, level),
            is_reference,
            ignore,
        }
    }
}

impl GpuVariant {
    fn wgpu() -> Self {
        Self {
            backend: GpuBackend::Wgpu,
            cached: false,
            no_depth: false,
        }
    }

    fn webgl() -> Self {
        Self {
            backend: GpuBackend::WebGl,
            cached: false,
            no_depth: false,
        }
    }

    fn cached(mut self) -> Self {
        self.cached = true;
        self
    }

    fn without_depth(mut self) -> Self {
        self.no_depth = true;
        self
    }

    fn is_webgl(self) -> bool {
        matches!(self.backend, GpuBackend::WebGl)
    }

    fn resolve(self, args: &Arguments, tolerance: u8) -> TestCase {
        let webgl = self.is_webgl();
        let mut suffix = if webgl { "gpu_webgl" } else { "gpu" }.to_owned();
        if self.no_depth {
            suffix.push_str("_no_depth");
        }
        if self.cached {
            suffix.push_str("_cached");
        }
        let tolerance = quote! { #tolerance };
        TestCase {
            suffix,
            renderer: Renderer::Gpu(self),
            tolerance,
            is_reference: args.gpu_only && !webgl && !self.no_depth,
            ignore: args.skip_gpu || (webgl && args.skip_webgl),
        }
    }
}

struct TestContext<'a> {
    input_fn_name: &'a Ident,
    input_fn_name_str: &'a str,
    reference_image_name: &'a Ident,
    cached_reference_image_name: &'a Ident,
    args: &'a Arguments,
}

impl TestContext<'_> {
    fn test_name(&self, suffix: &str) -> (Ident, String) {
        let name = format!("{}_{}", self.input_fn_name, suffix);
        (Ident::new(&name, self.input_fn_name.span()), name)
    }

    fn reference_test_name(&self, cached: bool) -> String {
        if cached {
            format!("{}_cached", self.input_fn_name)
        } else {
            self.input_fn_name_str.to_owned()
        }
    }

    fn invocation(&self, cached: bool) -> TokenStream2 {
        let input_fn_name = self.input_fn_name;
        match (self.args.glyph, cached) {
            (false, false) => quote! { #input_fn_name(&mut ctx); },
            (true, false) => quote! { #input_fn_name(&mut ctx, false); },
            (true, true) => quote! { #input_fn_name(&mut ctx, true); },
            (false, true) => unreachable!("only glyph tests have cached variants"),
        }
    }

    fn ignore_attribute(&self, ignore: bool) -> TokenStream2 {
        if !ignore {
            quote! {}
        } else if let Some(reason) = &self.args.ignore_reason {
            quote! { #[ignore = #reason] }
        } else {
            quote! { #[ignore] }
        }
    }

    fn generate_test(&self, case: TestCase) -> TokenStream2 {
        let Self { args, .. } = self;
        let Arguments {
            width,
            height,
            transparent,
            no_ref,
            diff_pixels,
            ..
        } = args;
        let cached = case.renderer.cached();
        let (fn_name, fn_name_str) = self.test_name(&case.suffix);
        let test_name = self.reference_test_name(cached);
        let TestCase {
            renderer,
            tolerance,
            mut is_reference,
            ignore,
            ..
        } = case;
        let reference_image_name = if cached {
            self.cached_reference_image_name
        } else {
            self.reference_image_name
        };
        let invoke_input = self.invocation(cached);
        let ignore_attribute = self.ignore_attribute(ignore);
        let (cfg_attribute, test_attribute, asyncness, create_ctx) = match renderer {
            Renderer::Cpu(variant) => {
                let (pipeline, level, num_threads) = variant.config();
                let render_mode = pipeline.render_mode();
                let is_wasm = matches!(level, CpuLevel::Wasm);
                let level = level.value();
                let attributes = if is_wasm {
                    assert_eq!(num_threads, 0, "wasm is single threaded");
                    is_reference = false;
                    (
                        quote! { #[cfg(target_arch = "wasm32")] },
                        quote! { #[wasm_bindgen_test::wasm_bindgen_test] },
                    )
                } else {
                    (quote! {}, quote! { #[test] })
                };
                (
                    attributes.0,
                    attributes.1,
                    quote! {},
                    quote! {
                        crate::util::get_ctx::<crate::renderer::CpuRenderer>(
                            #width,
                            #height,
                            #transparent,
                            #num_threads,
                            #level,
                            #render_mode,
                        )
                    },
                )
            }
            Renderer::Gpu(variant) => {
                let webgl = variant.is_webgl();
                let (cfg_attribute, test_attribute, asyncness) = if webgl {
                    (
                        quote! { #[cfg(all(target_arch = "wasm32", feature = "webgl"))] },
                        quote! { #[wasm_bindgen_test::wasm_bindgen_test] },
                        quote! { async },
                    )
                } else {
                    (quote! {}, quote! { #[test] }, quote! {})
                };
                let create_ctx = if variant.no_depth {
                    quote! {
                        crate::util::get_ctx_with_depth_buffer::<crate::renderer::GpuRenderer>(
                            #width,
                            #height,
                            #transparent,
                            0,
                            "fallback",
                            vello_cpu::RenderMode::OptimizeSpeed,
                            false,
                        )
                    }
                } else {
                    quote! {
                        crate::util::get_ctx::<crate::renderer::GpuRenderer>(
                            #width,
                            #height,
                            #transparent,
                            0,
                            "fallback",
                            vello_cpu::RenderMode::OptimizeSpeed,
                        )
                    }
                };
                (cfg_attribute, test_attribute, asyncness, create_ctx)
            }
        };

        quote! {
            #cfg_attribute
            #ignore_attribute
            #test_attribute
            #asyncness fn #fn_name() {
                use crate::util::check_ref;

                let mut ctx = #create_ctx;
                #invoke_input
                ctx.flush();
                if !#no_ref {
                    check_ref(
                        &mut ctx,
                        #test_name,
                        #fn_name_str,
                        #tolerance,
                        #diff_pixels,
                        #is_reference,
                        #reference_image_name,
                    );
                }
            }
        }
    }
}

fn reference_image(input_fn_name: &Ident, suffix: &str, no_ref: bool) -> (Ident, TokenStream2) {
    let const_name = Ident::new(
        &format!(
            "{}{}_REFERENCE_IMAGE",
            input_fn_name.to_string().to_uppercase(),
            suffix.to_uppercase()
        ),
        input_fn_name.span(),
    );
    let snapshot_name = format!("{input_fn_name}{suffix}.png");
    let declaration = if no_ref {
        quote! {
            const #const_name: &[u8] = &[];
        }
    } else {
        quote! {
            #[cfg(target_arch = "wasm32")]
            const #const_name: &[u8] = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/snapshots/",
                #snapshot_name
            ));
            #[cfg(not(target_arch = "wasm32"))]
            const #const_name: &[u8] = &[];
        }
    };
    (const_name, declaration)
}

pub(crate) fn vello_test_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let input_fn_name = input_fn.sig.ident.clone();
    let input_fn_name_str = input_fn_name.to_string();
    let mut args = match parse_args(attr) {
        Ok(args) => args,
        Err(error) => return error.into_compile_error().into(),
    };

    match (args.glyph, input_fn.sig.inputs.len()) {
        (false, 1) | (true, 2) => {}
        (true, 1) => panic!("glyph tests must take two arguments"),
        (false, 2) => panic!("method has unexpected second parameter"),
        _ => panic!(
            "test functions must take either one renderer argument or renderer + enable_caching"
        ),
    }

    // These tests currently don't work with `vello_gpu`.
    args.skip_gpu |= input_fn_name_str.contains("layer_multiple_properties")
        || input_fn_name_str.contains("mask");
    assert!(
        !(args.gpu_only && args.skip_gpu),
        "`gpu_only` cannot be combined with `skip_gpu`"
    );

    // Wasm doesn't have access to the filesystem. For wasm, inline the snapshot bytes into the
    // binary.
    let (reference_image_name, reference_image_const) =
        reference_image(&input_fn_name, "", args.no_ref);
    let (cached_reference_image_name, cached_reference_image_const) = if args.glyph {
        reference_image(&input_fn_name, "_cached", args.no_ref)
    } else {
        (reference_image_name.clone(), quote! {})
    };

    let tolerances = Tolerances::new(&args);

    let context = TestContext {
        input_fn_name: &input_fn_name,
        input_fn_name_str: &input_fn_name_str,
        reference_image_name: &reference_image_name,
        cached_reference_image_name: &cached_reference_image_name,
        args: &args,
    };

    let mut cpu_variants = Pipeline::ALL
        .into_iter()
        .flat_map(|pipeline| {
            CpuLevel::ALL
                .into_iter()
                .map(move |level| CpuVariant::Pipeline { pipeline, level })
        })
        .collect::<Vec<_>>();
    cpu_variants.push(CpuVariant::Multithreaded);
    if args.glyph {
        cpu_variants.push(CpuVariant::Cached);
    }
    let cpu_tests = cpu_variants
        .into_iter()
        .map(|variant| variant.resolve(&args, &tolerances))
        .map(|case| context.generate_test(case));

    let mut gpu_variants = Vec::new();
    for variant in [GpuVariant::wgpu(), GpuVariant::webgl()] {
        gpu_variants.push(variant);
        if args.glyph {
            gpu_variants.push(variant.cached());
        }
        if args.gpu_no_depth {
            gpu_variants.push(variant.without_depth());
        }
    }
    let gpu_tests = gpu_variants
        .into_iter()
        .map(|variant| variant.resolve(&args, tolerances.gpu))
        .map(|case| context.generate_test(case));

    // TODO: Tests with the same names in different modules can clash, see
    // https://github.com/linebender/vello/pull/925#discussion_r2070710362.
    // We should take the module path into consideration for naming the tests.
    quote! {
        #input_fn
        #reference_image_const
        #cached_reference_image_const
        #(#cpu_tests)*
        #(#gpu_tests)*
    }
    .into()
}

fn parse_args(attr: TokenStream) -> syn::Result<Arguments> {
    let mut args = Arguments::default();
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("width") {
            args.width = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("height") {
            args.height = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("diff_pixels") {
            args.diff_pixels = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("cpu_u8_tolerance") {
            args.cpu_u8_tolerance = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("gpu_tolerance") {
            args.gpu_tolerance = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("transparent") {
            args.transparent = true;
        } else if meta.path.is_ident("skip_cpu") {
            args.skip_cpu = true;
        } else if meta.path.is_ident("skip_multithreaded") {
            args.skip_multithreaded = true;
        } else if meta.path.is_ident("skip_gpu") {
            args.skip_gpu = true;
        } else if meta.path.is_ident("skip_webgl") {
            args.skip_webgl = true;
        } else if meta.path.is_ident("gpu_only") {
            args.skip_cpu = true;
            args.gpu_only = true;
        } else if meta.path.is_ident("gpu_no_depth") {
            args.gpu_no_depth = true;
        } else if meta.path.is_ident("no_ref") {
            args.no_ref = true;
        } else if meta.path.is_ident("glyph") {
            args.glyph = true;
        } else if meta.path.is_ident("ignore") {
            args.skip_cpu = true;
            args.skip_multithreaded = true;
            args.skip_gpu = true;
            if meta.input.peek(syn::Token![=]) {
                args.ignore_reason = Some(meta.value()?.parse::<LitStr>()?.value());
            }
        } else {
            return Err(meta.error("unknown `vello_test` attribute"));
        }
        Ok(())
    });
    parser.parse(attr)?;
    Ok(args)
}
