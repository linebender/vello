// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    DEFAULT_CPU_F32_TOLERANCE, DEFAULT_CPU_U8_TOLERANCE, DEFAULT_GPU_TOLERANCE,
    DEFAULT_SIMD_TOLERANCE,
};
use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::quote;
use syn::meta::ParseNestedMeta;
use syn::parse::Parser;
use syn::{ItemFn, LitInt, LitStr, Meta, parse_macro_input};

/// The arguments of `#[vello_test(...)]`.
///
/// The arguments in [`PlatformArgs`] can additionally be overridden for some compilation
/// targets by wrapping them in a platform block:
///
/// ```ignore
/// #[vello_test(gpu_tolerance = 1, windows(gpu_tolerance = 3, diff_pixels = 2))]
/// #[vello_test(macos(skip_gpu), wasm(ignore = "reason"))]
/// #[vello_test(cfg_attr(all(target_os = "linux", target_arch = "aarch64"), cpu_u8_tolerance = 4))]
/// ```
///
/// `windows`, `macos`, `linux` and `wasm` are shorthands for `cfg_attr(<predicate>, ...)`,
/// which accepts any `cfg` predicate. The predicate is checked against the target the tests
/// are compiled for, so the WebGL variants match `wasm` regardless of the browser's OS.
/// A value set in a block replaces the base value of that argument rather than adding to it.
/// If several blocks that set the same value match, the last one wins; skips from all matching
/// blocks apply.
struct Arguments {
    /// The width of the scene.
    width: u16,
    /// The height of the scene.
    height: u16,
    /// The arguments that apply on every platform, unless overridden in `platforms`.
    base: PlatformArgs,
    /// Platform-specific overrides, in the order they were written.
    platforms: Vec<PlatformOverride>,
    /// Whether the background should be transparent (the default is white).
    transparent: bool,
    /// Whether only `vello_gpu` should run and generate the reference image.
    gpu_only: bool,
    /// Whether to additionally run `vello_gpu` with depth buffering disabled.
    gpu_no_depth: bool,
    /// Whether to additionally run the WebGL backend rendering into a texture.
    webgl_texture: bool,
    /// Whether no reference image should actually be created (for tests that only check
    /// for panics, but are not interested in the actual output).
    no_ref: bool,
    /// Whether this is a glyph test and should generate the additional caching variants.
    glyph: bool,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            base: PlatformArgs::default(),
            platforms: Vec::new(),
            transparent: false,
            gpu_only: false,
            gpu_no_depth: false,
            webgl_texture: false,
            no_ref: false,
            glyph: false,
        }
    }
}

/// The arguments that can be overridden per platform. Unset values default to 0.
#[derive(Default)]
struct PlatformArgs {
    /// The (additional) maximum tolerance for how much two pixels are allowed to deviate from each other
    /// when comparing to the reference images. Some renderers already have an existing tolerance
    /// (see the constants at the top of the file), this value will simply be added
    /// to the currently existing threshold. See the top of the file for an explanation of
    /// how exactly the tolerance is interpreted.
    cpu_u8_tolerance: Option<u8>,
    /// Same as above, but for the gpu renderer.
    gpu_tolerance: Option<u8>,
    /// The maximum number of pixels that are allowed to completely deviate from the reference
    /// images. This attribute mainly exists because there are some test cases (like gradients),
    /// where, due to floating point inaccuracies, some pixels might land on a different color
    /// stop and thus yield a different value in CI.
    diff_pixels: Option<u32>,
    /// Additional differing pixels allowed for all CPU variants.
    cpu_diff_pixels: Option<u32>,
    /// Additional differing pixels allowed for all GPU variants.
    gpu_diff_pixels: Option<u32>,
    /// Which renderers the test should not be run on.
    skip: Skips,
    /// A reason for ignoring a test.
    ignore_reason: Option<String>,
}

#[derive(Clone, Copy, Default)]
struct Skips {
    /// Whether the test should not be run on the CPU (`vello_cpu`).
    cpu: bool,
    /// Whether the test should not be run on the multi-threaded CPU (`vello_cpu`).
    multithreaded: bool,
    /// Whether the test should not be run on the GPU (`vello_gpu`).
    gpu: bool,
    /// Whether the test should not be run using the WebGL backend.
    webgl: bool,
}

struct PlatformOverride {
    /// The `cfg` predicate selecting the platform.
    cfg: TokenStream2,
    args: PlatformArgs,
}

/// A value that may differ per platform.
struct PerPlatform<T> {
    base: T,
    /// `(cfg predicate, value)` pairs; the last one whose predicate holds wins.
    overrides: Vec<(TokenStream2, T)>,
}

impl<T: Copy + quote::ToTokens> PerPlatform<T> {
    fn new(base: T) -> Self {
        Self {
            base,
            overrides: Vec::new(),
        }
    }

    fn map<U>(&self, f: impl Fn(T) -> U) -> PerPlatform<U> {
        PerPlatform {
            base: f(self.base),
            overrides: self
                .overrides
                .iter()
                .map(|(cfg, value)| (cfg.clone(), f(*value)))
                .collect(),
        }
    }

    /// An expression evaluating to the value for the target platform.
    fn to_expr(&self) -> TokenStream2 {
        let base = self.base;
        self.overrides
            .iter()
            .fold(quote! { #base }, |otherwise, (cfg, value)| {
                quote! { if cfg!(#cfg) { #value } else { #otherwise } }
            })
    }
}

impl Arguments {
    fn per_platform<T: Copy + Default>(
        &self,
        get: impl Fn(&PlatformArgs) -> Option<T>,
    ) -> PerPlatform<T> {
        PerPlatform {
            base: get(&self.base).unwrap_or_default(),
            overrides: self
                .platforms
                .iter()
                .filter_map(|platform| Some((platform.cfg.clone(), get(&platform.args)?)))
                .collect(),
        }
    }

    /// The `#[ignore]` attributes for a test variant that is skipped when `skipped` returns true.
    fn ignore_attributes(&self, skipped: impl Fn(Skips) -> bool) -> TokenStream2 {
        fn ignore(reason: Option<&String>) -> TokenStream2 {
            match reason {
                Some(reason) => quote! { ignore = #reason },
                None => quote! { ignore },
            }
        }

        if skipped(self.base.skip) {
            let ignore = ignore(self.base.ignore_reason.as_ref());
            return quote! { #[#ignore] };
        }
        let platforms = self
            .platforms
            .iter()
            .filter(|platform| skipped(platform.args.skip))
            .collect::<Vec<_>>();
        // At most one attribute may apply, as duplicate `#[ignore]` attributes trigger a
        // warning. As for values, the last matching platform wins.
        platforms
            .iter()
            .enumerate()
            .map(|(i, platform)| {
                let cfg = &platform.cfg;
                let later = platforms[i + 1..].iter().map(|platform| &platform.cfg);
                let ignore = ignore(platform.args.ignore_reason.as_ref());
                quote! { #[cfg_attr(all(#cfg, not(any(#(#later),*))), #ignore)] }
            })
            .collect()
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

    fn dispatch_cfg_attribute(self) -> TokenStream2 {
        match self {
            Self::Neon => quote! { #[cfg(target_arch = "aarch64")] },
            Self::Sse2 => {
                quote! { #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] }
            }
            Self::Sse42 => quote! {
                #[cfg(all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    not(disable_dispatch_sse4_2)
                ))]
            },
            Self::Avx2 => quote! {
                #[cfg(all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    not(disable_dispatch_avx2)
                ))]
            },
            Self::Avx512 => quote! {
                #[cfg(all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    not(disable_dispatch_avx512)
                ))]
            },
            Self::Scalar | Self::Wasm => quote! {},
        }
    }

    fn tolerance(self, scalar: TokenStream2, simd: TokenStream2) -> TokenStream2 {
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
    cpu_u8_scalar: PerPlatform<u8>,
    cpu_u8_simd: PerPlatform<u8>,
    cpu_f32_scalar: PerPlatform<u8>,
    cpu_f32_simd: PerPlatform<u8>,
    gpu: PerPlatform<u8>,
}

impl Tolerances {
    fn new(args: &Arguments) -> Self {
        let cpu_u8 = args.per_platform(|args| args.cpu_u8_tolerance);
        Self {
            cpu_u8_scalar: cpu_u8.map(|tolerance| tolerance + DEFAULT_CPU_U8_TOLERANCE),
            cpu_u8_simd: cpu_u8
                .map(|tolerance| tolerance + DEFAULT_SIMD_TOLERANCE.max(DEFAULT_CPU_U8_TOLERANCE)),
            cpu_f32_scalar: PerPlatform::new(DEFAULT_CPU_F32_TOLERANCE),
            cpu_f32_simd: PerPlatform::new(DEFAULT_CPU_F32_TOLERANCE + DEFAULT_SIMD_TOLERANCE),
            gpu: args
                .per_platform(|args| args.gpu_tolerance)
                .map(|tolerance| tolerance + DEFAULT_GPU_TOLERANCE),
        }
    }

    fn for_cpu(&self, pipeline: Pipeline, level: CpuLevel) -> TokenStream2 {
        let (scalar, simd) = match pipeline {
            Pipeline::U8 => (&self.cpu_u8_scalar, &self.cpu_u8_simd),
            Pipeline::F32 => (&self.cpu_f32_scalar, &self.cpu_f32_simd),
        };
        level.tolerance(scalar.to_expr(), simd.to_expr())
    }
}

/// An expression evaluating to the number of differing pixels allowed on the target platform.
fn diff_pixels(
    args: &Arguments,
    renderer_specific: impl Fn(&PlatformArgs) -> Option<u32>,
) -> TokenStream2 {
    let shared = args.per_platform(|args| args.diff_pixels).to_expr();
    let renderer_specific = args.per_platform(renderer_specific).to_expr();
    quote! { (#shared) + (#renderer_specific) }
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
    texture: bool,
}

enum Renderer {
    Cpu(CpuVariant),
    Gpu(GpuVariant),
}

struct TestCase {
    suffix: String,
    renderer: Renderer,
    tolerance: TokenStream2,
    diff_pixels: TokenStream2,
    is_reference: bool,
    /// The `#[ignore]` attributes of the test.
    ignore: TokenStream2,
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
                args.ignore_attributes(|skip| skip.cpu || !level.is_available()),
            ),
            Self::Multithreaded => (
                "cpu_multithreaded".to_owned(),
                false,
                args.ignore_attributes(|skip| skip.cpu || skip.multithreaded),
            ),
            Self::Cached => (
                "cpu_f32_scalar_cached".to_owned(),
                !args.gpu_only,
                args.ignore_attributes(|skip| skip.cpu),
            ),
        };
        TestCase {
            suffix,
            renderer: Renderer::Cpu(self),
            tolerance: tolerances.for_cpu(pipeline, level),
            diff_pixels: diff_pixels(args, |args| args.cpu_diff_pixels),
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
            texture: false,
        }
    }

    fn webgl() -> Self {
        Self {
            backend: GpuBackend::WebGl,
            cached: false,
            no_depth: false,
            texture: false,
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

    fn with_texture(mut self) -> Self {
        self.texture = true;
        self
    }

    fn is_webgl(self) -> bool {
        matches!(self.backend, GpuBackend::WebGl)
    }

    fn resolve(self, args: &Arguments, tolerance: &PerPlatform<u8>) -> TestCase {
        let webgl = self.is_webgl();
        let mut suffix = if webgl { "gpu_webgl" } else { "gpu" }.to_owned();
        if self.texture {
            suffix.push_str("_texture");
        }
        if self.no_depth {
            suffix.push_str("_no_depth");
        }
        if self.cached {
            suffix.push_str("_cached");
        }
        TestCase {
            suffix,
            renderer: Renderer::Gpu(self),
            tolerance: tolerance.to_expr(),
            diff_pixels: diff_pixels(args, |args| args.gpu_diff_pixels),
            is_reference: args.gpu_only && !webgl && !self.no_depth,
            ignore: args.ignore_attributes(|skip| skip.gpu || (webgl && skip.webgl)),
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

    fn generate_test(&self, case: TestCase) -> TokenStream2 {
        let Self { args, .. } = self;
        let Arguments {
            width,
            height,
            transparent,
            no_ref,
            ..
        } = args;
        let cached = case.renderer.cached();
        let (fn_name, fn_name_str) = self.test_name(&case.suffix);
        let test_name = self.reference_test_name(cached);
        let TestCase {
            renderer,
            tolerance,
            diff_pixels,
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
        let use_texture_target = match &renderer {
            Renderer::Gpu(variant) if variant.texture => quote! { ctx.use_texture_target(); },
            _ => quote! {},
        };
        let (cfg_attribute, test_attribute, asyncness, create_ctx) = match renderer {
            Renderer::Cpu(variant) => {
                let (pipeline, level, num_threads) = variant.config();
                let render_mode = pipeline.render_mode();
                let is_wasm = matches!(level, CpuLevel::Wasm);
                let dispatch_cfg_attribute = level.dispatch_cfg_attribute();
                let requires_fallback = matches!(
                    variant,
                    CpuVariant::Pipeline {
                        level: CpuLevel::Scalar,
                        ..
                    } | CpuVariant::Multithreaded
                        | CpuVariant::Cached
                );
                let level = level.value();
                let (target_cfg_attribute, test_attribute) = if is_wasm {
                    assert_eq!(num_threads, 0, "wasm is single threaded");
                    is_reference = false;
                    (
                        quote! { #[cfg(target_arch = "wasm32")] },
                        quote! { #[wasm_bindgen_test::wasm_bindgen_test] },
                    )
                } else if requires_fallback {
                    (
                        quote! { #[cfg(feature = "force_support_fallback")] },
                        quote! { #[test] },
                    )
                } else {
                    (quote! {}, quote! { #[test] })
                };
                (
                    quote! {
                        #dispatch_cfg_attribute
                        #target_cfg_attribute
                    },
                    test_attribute,
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
                            "baseline",
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
                            "baseline",
                            vello_cpu::RenderMode::OptimizeSpeed,
                        )
                    }
                };
                (cfg_attribute, test_attribute, asyncness, create_ctx)
            }
        };

        quote! {
            #cfg_attribute
            #ignore
            #test_attribute
            #asyncness fn #fn_name() {
                use crate::util::check_ref;

                let mut ctx = #create_ctx;
                #use_texture_target
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
    let mut args = match parse_args(attr.into()) {
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
    args.base.skip.gpu |= input_fn_name_str.contains("layer_multiple_properties")
        || input_fn_name_str.contains("mask");
    assert!(
        !(args.gpu_only && args.base.skip.gpu),
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
        if args.webgl_texture && variant.is_webgl() {
            gpu_variants.push(variant.with_texture());
            if args.gpu_no_depth {
                gpu_variants.push(variant.with_texture().without_depth());
            }
        }
    }
    let gpu_tests = gpu_variants
        .into_iter()
        .map(|variant| variant.resolve(&args, &tolerances.gpu))
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

fn parse_args(attr: TokenStream2) -> syn::Result<Arguments> {
    let mut args = Arguments::default();
    let parser = syn::meta::parser(|meta| {
        if parse_platform_arg(&meta, &mut args.base)? {
            return Ok(());
        }
        if let Some(platform) = parse_platform_block(&meta)? {
            args.platforms.push(platform);
        } else if meta.path.is_ident("width") {
            args.width = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("height") {
            args.height = meta.value()?.parse::<LitInt>()?.base10_parse()?;
        } else if meta.path.is_ident("transparent") {
            args.transparent = true;
        } else if meta.path.is_ident("gpu_only") {
            args.base.skip.cpu = true;
            args.gpu_only = true;
        } else if meta.path.is_ident("gpu_no_depth") {
            args.gpu_no_depth = true;
        } else if meta.path.is_ident("webgl_texture") {
            args.webgl_texture = true;
        } else if meta.path.is_ident("no_ref") {
            args.no_ref = true;
        } else if meta.path.is_ident("glyph") {
            args.glyph = true;
        } else {
            return Err(meta.error("unknown `vello_test` attribute"));
        }
        Ok(())
    });
    parser.parse2(attr)?;
    Ok(args)
}

/// Parses an argument that can be set per platform, returning whether `meta` was one.
fn parse_platform_arg(meta: &ParseNestedMeta<'_>, args: &mut PlatformArgs) -> syn::Result<bool> {
    if meta.path.is_ident("diff_pixels") {
        args.diff_pixels = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
    } else if meta.path.is_ident("cpu_diff_pixels") {
        args.cpu_diff_pixels = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
    } else if meta.path.is_ident("gpu_diff_pixels") {
        args.gpu_diff_pixels = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
    } else if meta.path.is_ident("cpu_u8_tolerance") {
        args.cpu_u8_tolerance = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
    } else if meta.path.is_ident("gpu_tolerance") {
        args.gpu_tolerance = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
    } else if meta.path.is_ident("skip_cpu") {
        args.skip.cpu = true;
    } else if meta.path.is_ident("skip_multithreaded") {
        args.skip.multithreaded = true;
    } else if meta.path.is_ident("skip_gpu") {
        args.skip.gpu = true;
    } else if meta.path.is_ident("skip_webgl") {
        args.skip.webgl = true;
    } else if meta.path.is_ident("ignore") {
        args.skip.cpu = true;
        args.skip.multithreaded = true;
        args.skip.gpu = true;
        if meta.input.peek(syn::Token![=]) {
            args.ignore_reason = Some(meta.value()?.parse::<LitStr>()?.value());
        }
    } else {
        return Ok(false);
    }
    Ok(true)
}

/// Parses a platform block, returning `None` if `meta` is not one.
fn parse_platform_block(meta: &ParseNestedMeta<'_>) -> syn::Result<Option<PlatformOverride>> {
    let Some(name) = meta.path.get_ident() else {
        return Ok(None);
    };
    let mut args = PlatformArgs::default();
    let parse_arg = |meta: ParseNestedMeta<'_>| {
        if parse_platform_arg(&meta, &mut args)? {
            Ok(())
        } else {
            Err(meta.error("this `vello_test` attribute cannot be set per platform"))
        }
    };
    let cfg = match name.to_string().as_str() {
        "windows" => quote! { target_os = "windows" },
        "macos" => quote! { target_os = "macos" },
        "linux" => quote! { target_os = "linux" },
        "wasm" => quote! { target_arch = "wasm32" },
        "cfg_attr" => {
            let content;
            syn::parenthesized!(content in meta.input);
            let cfg = content.parse::<Meta>()?;
            content.parse::<syn::Token![,]>()?;
            syn::meta::parser(parse_arg).parse2(content.parse()?)?;
            return Ok(Some(PlatformOverride {
                cfg: quote! { #cfg },
                args,
            }));
        }
        _ => return Ok(None),
    };
    meta.parse_nested_meta(parse_arg)?;
    Ok(Some(PlatformOverride { cfg, args }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(attr: TokenStream2) -> Arguments {
        parse_args(attr).unwrap()
    }

    fn compact(tokens: TokenStream2) -> String {
        tokens.to_string().replace(' ', "")
    }

    #[test]
    fn platform_value_replaces_base_value() {
        let args = parse(quote! { gpu_tolerance = 1, windows(gpu_tolerance = 3) });
        let tolerance = Tolerances::new(&args).gpu.to_expr();
        assert_eq!(
            compact(tolerance),
            r#"ifcfg!(target_os="windows"){4u8}else{2u8}"#
        );
    }

    #[test]
    fn platform_block_may_come_before_base_value() {
        let args = parse(quote! { macos(cpu_u8_tolerance = 5), cpu_u8_tolerance = 1 });
        let tolerance = Tolerances::new(&args).cpu_u8_scalar;
        assert_eq!(tolerance.base, 1 + DEFAULT_CPU_U8_TOLERANCE);
        assert_eq!(tolerance.overrides[0].1, 5 + DEFAULT_CPU_U8_TOLERANCE);
    }

    #[test]
    fn last_matching_platform_wins() {
        let args = parse(quote! {
            linux(gpu_tolerance = 2),
            cfg_attr(target_arch = "aarch64", gpu_tolerance = 3),
        });
        assert_eq!(
            compact(args.per_platform(|args| args.gpu_tolerance).to_expr()),
            concat!(
                r#"ifcfg!(target_arch="aarch64"){3u8}"#,
                r#"else{ifcfg!(target_os="linux"){2u8}else{0u8}}"#
            )
        );
    }

    #[test]
    fn platforms_without_the_value_are_not_checked() {
        let args = parse(quote! { windows(skip_gpu), wasm(diff_pixels = 4) });
        assert_eq!(
            compact(diff_pixels(&args, |args| args.cpu_diff_pixels)),
            r#"(ifcfg!(target_arch="wasm32"){4u32}else{0u32})+(0u32)"#
        );
    }

    #[test]
    fn cfg_attr_accepts_any_predicate() {
        let args = parse(quote! {
            cfg_attr(all(target_os = "linux", not(target_arch = "x86_64")), skip_webgl, gpu_diff_pixels = 7)
        });
        let platform = &args.platforms[0];
        assert_eq!(
            compact(platform.cfg.clone()),
            r#"all(target_os="linux",not(target_arch="x86_64"))"#
        );
        assert!(platform.args.skip.webgl);
        assert_eq!(platform.args.gpu_diff_pixels, Some(7));
    }

    #[test]
    fn platform_skips_ignore_only_on_that_platform() {
        let args = parse(quote! { windows(skip_gpu), macos(ignore = "flaky") });
        assert_eq!(
            compact(args.ignore_attributes(|skip| skip.gpu)),
            concat!(
                r#"#[cfg_attr(all(target_os="windows",not(any(target_os="macos"))),ignore)]"#,
                r#"#[cfg_attr(all(target_os="macos",not(any())),ignore="flaky")]"#
            )
        );
        assert_eq!(
            compact(args.ignore_attributes(|skip| skip.cpu)),
            r#"#[cfg_attr(all(target_os="macos",not(any())),ignore="flaky")]"#
        );
    }

    #[test]
    fn base_skip_ignores_everywhere() {
        let args = parse(quote! { skip_gpu, windows(ignore = "reason") });
        assert_eq!(
            compact(args.ignore_attributes(|skip| skip.gpu)),
            "#[ignore]"
        );
    }

    #[test]
    fn rejects_arguments_that_cannot_differ_per_platform() {
        let error = parse_args(quote! { windows(width = 10) }).err().unwrap();
        assert_eq!(
            error.to_string(),
            "this `vello_test` attribute cannot be set per platform"
        );
        assert!(parse_args(quote! { cfg_attr(windows) }).is_err());
    }
}
