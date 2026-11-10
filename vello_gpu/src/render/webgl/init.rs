// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Context creation, shader compilation and renderer initialization.

use crate::render::webgl::WebGlRenderer;
use crate::render::webgl::gl::resource::{
    FragmentShader, Program, VertexShader, WebGlFragmentShader, WebGlVertexShader,
};
use crate::render::webgl::passes::WebGlPrograms;
use crate::render::webgl::passes::blend::get_blend_uniforms;
use crate::render::webgl::passes::copy::get_copy_uniforms;
use crate::render::webgl::passes::filter::get_filter_pass_uniforms;
use crate::render::webgl::passes::strip::get_strip_uniforms;
use crate::render::webgl::resources::{WebGlResources, create_webgl_resources};
use crate::render::webgl::target::REQUIRED_DEPTH_BITS;
use crate::render::webgl::{
    IncompatibleContextReason, IncompatiblePrecompiledShadersReason, WebGlContextOperation,
    WebGlError, WebGlOperation, WebGlResultExt, WebGlShaderProgramOperation, WebGlShaderStage,
};
use crate::{
    RenderSettings, Resources,
    render::common::{DeviceLimits, ScenePreparer},
};
use core::cell::Cell;
use core::time::Duration;
use vello_gpu_shaders::{blend, copy, filter as filter_shader, render};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext, WebGlProgram, WebGlShader};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Query the WebGL context for the max texture size.
pub(crate) fn get_max_texture_dimension_2d(gl: &WebGl2RenderingContext) -> Result<u32, WebGlError> {
    Ok(gl
        .get_parameter(WebGl2RenderingContext::MAX_TEXTURE_SIZE)
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Query))?
        .as_f64()
        .unwrap() as u32)
}

/// Result of polling WebGL renderer initialization.
#[derive(Debug)]
pub enum WebGlRendererInitStatus {
    /// Shader compilation or program linking is still in progress.
    Pending(WebGlRendererInit),
    /// Initialization has completed and the renderer is ready to use.
    Complete(WebGlRenderer),
}

/// WebGL renderer initialization that may still be compiling shader programs.
///
/// Its [`Resources`] are already usable, so CPU-side work such as glyph preparation can proceed
/// while compilation continues. Check [`WebGlRendererInit::is_ready`] (or poll
/// [`WebGlRendererInit::try_finish`]) from successive animation frames to avoid blocking the
/// browser's main thread when `KHR_parallel_shader_compile` is available. On browsers without the
/// extension, finishing falls back to synchronously completing compilation.
#[derive(Debug)]
pub struct WebGlRendererInit {
    shaders: WebGlShaders,
    resources: WebGlResources,
    scene_preparer: ScenePreparer,
}

/// Vello's WebGL shader programs, whose compilation and linking may still be in progress.
///
/// The programs belong to the canvas's WebGL context and cannot be used with any other context.
#[derive(Debug)]
pub(crate) struct WebGlShaders {
    gl: WebGl2RenderingContext,
    use_depth_buffer: bool,
    parallel_shader_compile: bool,
    compilation_start_ms: f64,
    poll_count: Cell<u32>,
    ready: Cell<bool>,
    strip_program: PendingShaderProgram,
    filter_program: PendingShaderProgram,
    blend_program: PendingShaderProgram,
    copy_program: PendingShaderProgram,
}

/// Shader compilation statistics recorded while creating a [`WebGlRenderer`].
///
/// Only the WebGL backend reports these, because `KHR_parallel_shader_compile` lets it detect
/// when compilation actually finishes. WebGPU signals completion through
/// `createRenderPipelineAsync`, but wgpu exposes only synchronous pipeline creation, which may
/// return before the driver finishes compiling. Timing those calls would underreport the cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShaderCompilationStats {
    /// Number of vertex and fragment shaders compiled.
    pub shader_count: u32,
    /// Number of non-blocking completion-status polls.
    ///
    /// This is zero when initialization is completed through [`WebGlRendererInit::finish`]
    /// without first checking [`WebGlRendererInit::is_ready`].
    pub poll_count: u32,
    /// Wall-clock time from issuing the first compilation until all programs were confirmed
    /// linked.
    pub elapsed: Duration,
}

impl WebGlRenderer {
    /// Creates a new WebGL2 renderer and its persistent resources.
    ///
    /// This blocks until shader compilation and linking finish. Use [`WebGlRenderer::begin`] when
    /// initialization must not block the browser's main thread.
    pub fn new(canvas: &HtmlCanvasElement) -> Result<(Self, Resources), WebGlError> {
        Self::new_with(canvas, RenderSettings::default(), true)
    }

    /// Creates a new WebGL2 renderer and its persistent resources with specific settings.
    ///
    /// When `use_depth_buffer` is true, the WebGL context allocates a depth buffer and the renderer
    /// uses it for opaque-strip occlusion. Disabling it reduces GPU memory usage, but can increase
    /// overdraw.
    pub fn new_with(
        canvas: &HtmlCanvasElement,
        settings: RenderSettings,
        use_depth_buffer: bool,
    ) -> Result<(Self, Resources), WebGlError> {
        let (init, resources) = Self::begin_with(canvas, settings, use_depth_buffer)?;

        Ok((init.finish()?, resources))
    }

    /// Begins creating a WebGL2 renderer and its persistent resources.
    ///
    /// When `KHR_parallel_shader_compile` is available, shader compilation and program linking can
    /// continue while the caller yields to the browser. Check [`WebGlRendererInit::is_ready`] (or
    /// poll [`WebGlRendererInit::try_finish`]) on successive animation frames.
    ///
    /// If the extension is unavailable, finishing synchronously completes shader compilation and
    /// linking.
    pub fn begin(canvas: &HtmlCanvasElement) -> Result<(WebGlRendererInit, Resources), WebGlError> {
        Self::begin_with(canvas, RenderSettings::default(), true)
    }

    /// Begins creating a WebGL2 renderer with specific settings.
    ///
    /// See [`WebGlRenderer::begin`] for polling and fallback behavior.
    pub fn begin_with(
        canvas: &HtmlCanvasElement,
        settings: RenderSettings,
        use_depth_buffer: bool,
    ) -> Result<(WebGlRendererInit, Resources), WebGlError> {
        Self::begin_with_shaders(WebGlShaders::compile(canvas, use_depth_buffer)?, settings)
    }

    /// Begins creating a WebGL2 renderer from the shader programs that the `compileVelloShaders`
    /// JavaScript function from [`vello_gpu_shaders::webgl_bootstrap::JS_MODULE`] started
    /// compiling.
    ///
    /// Running that module before the WebAssembly module is downloaded and instantiated lets shader
    /// compilation overlap with loading it. The renderer uses the canvas and WebGL context that
    /// `compileVelloShaders` created, and a depth buffer only if that context has one. See
    /// [`WebGlRenderer::begin`] for polling and fallback behavior.
    ///
    /// The renderer takes ownership of the shaders and programs in `precompiled` and deletes them
    /// when it's dropped, so they must not be used elsewhere, and the same result must not be
    /// passed to this function twice.
    ///
    /// Returns [`WebGlError::IncompatiblePrecompiledShaders`] if the programs can't be adopted,
    /// including when `precompiled` is `null` because WebGL2 was unavailable.
    /// [`IncompatiblePrecompiledShadersReason::DifferentModule`] can also happen in
    /// production, for example when a cached JavaScript module from an older build is served
    /// with a newer WebAssembly module, so callers should fall back to
    /// [`WebGlRenderer::begin_with`] rather than give up on rendering.
    pub fn begin_with_precompiled(
        precompiled: &JsValue,
        settings: RenderSettings,
    ) -> Result<(WebGlRendererInit, Resources), WebGlError> {
        Self::begin_with_shaders(WebGlShaders::from_precompiled(precompiled)?, settings)
    }

    /// Allocates the renderer's persistent resources without waiting for `shaders` to compile.
    fn begin_with_shaders(
        shaders: WebGlShaders,
        mut settings: RenderSettings,
    ) -> Result<(WebGlRendererInit, Resources), WebGlError> {
        let gl = &shaders.gl;
        let device_limits = DeviceLimits {
            max_texture_dimension_2d: u16::try_from(get_max_texture_dimension_2d(gl)?)
                .unwrap_or(u16::MAX),
        };
        let (scene_preparer, resources) = ScenePreparer::new(&mut settings, &device_limits);
        let resource_texture_dimension_2d = device_limits.resource_texture_dimension_2d();

        let webgl_resources = create_webgl_resources(
            gl,
            &resources.image_cache,
            scene_preparer.layers_config,
            resource_texture_dimension_2d,
            shaders.use_depth_buffer,
        )?;

        let init = WebGlRendererInit {
            shaders,
            resources: webgl_resources,
            scene_preparer,
        };

        Ok((init, resources))
    }
}

/// Creates the canvas's WebGL2 context with the attributes the renderer depends on.
///
/// If the canvas already has a WebGL2 context, the browser returns it and ignores the requested
/// attributes, so the ones that can be queried are validated afterwards. The attributes must match
/// the defaults of `compileVelloShaders` in `vello_gpu_shaders/src/webgl_bootstrap.js`, which
/// creates the context when shaders are compiled from JavaScript;
/// `webgl_contexts_created_by_javascript_and_rust_match` in `vello_tests` checks this.
pub(crate) fn create_context(
    canvas: &HtmlCanvasElement,
    use_depth_buffer: bool,
) -> Result<WebGl2RenderingContext, WebGlError> {
    #[allow(
        clippy::assertions_on_constants,
        reason = "intentional guard against non-wasm32 use"
    )]
    {
        debug_assert!(
            cfg!(target_arch = "wasm32"),
            "`WebGlRenderer` can only be constructed when targeting `wasm32`",
        );
    }
    crate::render::common::maybe_warn_about_webgl_feature_conflict();

    // We do our own anti-aliasing, so no need to enable it in the WebGL
    // context.
    let context_options = js_sys::Object::new();
    js_sys::Reflect::set(&context_options, &"antialias".into(), &JsValue::FALSE)
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Configure))?;
    // Avoid running Vello GPU if it's gonna be super slow.
    js_sys::Reflect::set(
        &context_options,
        &"failIfMajorPerformanceCaveat".into(),
        &JsValue::TRUE,
    )
    .map_js_error(WebGlOperation::Context(WebGlContextOperation::Configure))?;
    // The opaque-strip optimization requires a 24+ bit depth buffer. If the hardware falls
    // back to a 16 bit depth buffer, correctness issues will arise. For all intents and
    // purposes, a device manufactured in the past 10 years should support 24+ bit depth
    // buffers (certainly those within the realm of what we consider "supported" devices) but:
    //
    // Relevant code for default depth buffer behaviour can be found here:
    // - Chromium defaults to 24 bit with no fallback: https://github.com/chromium/chromium/blob/86bafb3aab8e999690d310b201d0b5489f512b08/third_party/blink/renderer/platform/graphics/gpu/drawing_buffer.cc#L1376-L1400
    // - Firefox defaults to 24 bit with no fallback: https://github.com/mozilla/gecko-dev/blob/5836a062726f715fda621338a17b51aff30d0a8c/gfx/gl/MozFramebuffer.cpp#L155-L161
    // - Safari defaults to 24 bit _with 16 bit_ fallback: https://github.com/WebKit/WebKit/blob/a6d6c154bbee0643f5ad1e55c071558c0df9aef7/Source/WebCore/platform/graphics/angle/GraphicsContextGLANGLE.cpp#L393-L416
    //
    // TODO: The above understanding is encoded in a below assertion, but this should be encapsulated within a
    // "this device can run Vello correctly" check function.
    let depth = if use_depth_buffer {
        &JsValue::TRUE
    } else {
        &JsValue::FALSE
    };
    js_sys::Reflect::set(&context_options, &"depth".into(), depth)
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Configure))?;

    let context = canvas
        .get_context_with_context_options("webgl2", &context_options)
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Create))?
        .ok_or(WebGlError::ContextUnavailable)?;
    let gl = context
        .dyn_into::<WebGl2RenderingContext>()
        .expect("Context to be a WebGL2 context");

    // See https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices#use_webgl_provoking_vertex_when_its_available
    // for why we need to set this. This has been confirmed by benchmarks, where doing just a
    // couple of draw calls with external textures tanks performance on any
    // Apple device, especially low-tier iOS devices.
    // IMPORTANT: This assumes that any use of flat parameters in shaders is the same across
    // each vertex! In case this ever changes, we need to revisit this.
    // https://web3dsurvey.com/webgl2/extensions/WEBGL_provoking_vertex
    if let Some(ext) = gl
        .get_extension("WEBGL_provoking_vertex")
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Extension))?
    {
        // See https://registry.khronos.org/webgl/extensions/WEBGL_provoking_vertex
        const FIRST_VERTEX_CONVENTION_WEBGL: u32 = 0x8E4D;

        // In WebGL, the _last_ vertex supplies the value for flat shader varyings by default.
        // According to the spec, implementations should only expose this flag if the _first_
        // convention is faster by default. Therefore, if the extension is available, we always
        // set it to _first_.
        js_sys::Reflect::get(ext.as_ref(), &"provokingVertexWEBGL".into())
            .map_js_error(WebGlOperation::Context(WebGlContextOperation::Extension))?
            .unchecked_into::<js_sys::Function>()
            .call1(ext.as_ref(), &JsValue::from(FIRST_VERTEX_CONVENTION_WEBGL))
            .map_js_error(WebGlOperation::Context(WebGlContextOperation::Extension))?;
    }

    // Note: It is not entirely clear whether we really _have_ to ensure anti-aliasing is disabled.
    // This code is inherited from a similar snippet in wgpu
    // (https://github.com/gfx-rs/wgpu/blob/56e4a389ddd02403e232beef3d3ff305625e6485/wgpu-hal/src/gles/web.rs#L101-L106),
    // which itself seems to have been copied from the older `gfx` crate, where it was first introduced
    // in https://github.com/gfx-rs/gfx/pull/2554/changes#diff-a47711d61df7a43fe6dd99c39b936d17ff817cbc2238d7e3ae6698ffde9b88f7R79,
    // without any comment on why.
    // From my (Laurenz) testing, tests seem to work even when anti-aliasing is enabled,
    // but Andrew previously got errors similar to the ones outlined in
    // https://github.com/gfx-rs/wgpu/issues/5263. Therefore, we just leave it as is for now.
    #[cfg(debug_assertions)]
    {
        // If a WebGL context already exists on this canvas, it will be returned instead of
        // creating a new one with the correct context_options set.
        // See this comment for why we still care about non-antialiased context:
        // https://github.com/linebender/vello/pull/1546/changes#r3008692535
        if context_attribute(&gl, "antialias")? {
            return Err(WebGlError::IncompatibleContext(
                IncompatibleContextReason::AntialiasingEnabled,
            ));
        }
    }

    if use_depth_buffer {
        let actual_bits = gl
            .get_parameter(WebGl2RenderingContext::DEPTH_BITS)
            .map_js_error(WebGlOperation::Context(WebGlContextOperation::Query))?
            .as_f64()
            .unwrap() as u32;

        if actual_bits < REQUIRED_DEPTH_BITS {
            return Err(WebGlError::IncompatibleContext(
                IncompatibleContextReason::InsufficientDepthBuffer {
                    actual_bits,
                    required_bits: REQUIRED_DEPTH_BITS,
                },
            ));
        }
    }

    Ok(gl)
}

/// Returns a boolean attribute that `gl` was actually created with.
pub(crate) fn context_attribute(
    gl: &WebGl2RenderingContext,
    name: &str,
) -> Result<bool, WebGlError> {
    let context_attributes = gl
        .get_context_attributes()
        .ok_or(WebGlError::OperationFailed {
            operation: WebGlOperation::Context(WebGlContextOperation::Query),
            message: None,
        })?;

    Ok(js_sys::Reflect::get(&context_attributes, &name.into())
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Query))?
        .as_bool()
        .unwrap_or(false))
}

impl WebGlRendererInit {
    /// Returns whether shader compilation and program linking have finished, without blocking.
    ///
    /// Without `KHR_parallel_shader_compile`, completion cannot be queried without blocking, so
    /// this returns `true` immediately and [`WebGlRendererInit::finish`] completes compilation
    /// synchronously.
    pub fn is_ready(&self) -> bool {
        self.shaders.is_ready()
    }

    /// Polls shader compilation and program linking without blocking when
    /// `KHR_parallel_shader_compile` is available.
    ///
    /// On browsers without the extension, this synchronously finishes initialization and returns
    /// [`WebGlRendererInitStatus::Complete`].
    pub fn try_finish(self) -> Result<WebGlRendererInitStatus, WebGlError> {
        if self.is_ready() {
            Ok(WebGlRendererInitStatus::Complete(self.finish()?))
        } else {
            Ok(WebGlRendererInitStatus::Pending(self))
        }
    }

    /// Finishes initialization, blocking until shader compilation and program linking finish.
    ///
    /// This does not block once [`WebGlRendererInit::is_ready`] has returned `true`.
    pub fn finish(self) -> Result<WebGlRenderer, WebGlError> {
        let Self {
            shaders,
            resources,
            scene_preparer,
        } = self;
        let WebGlShaders {
            gl,
            compilation_start_ms,
            poll_count,
            strip_program,
            filter_program,
            blend_program,
            copy_program,
            ..
        } = shaders;

        let strip_program = strip_program.finish(&gl)?;
        let filter_program = filter_program.finish(&gl)?;
        let blend_program = blend_program.finish(&gl)?;
        let copy_program = copy_program.finish(&gl)?;
        let shader_compilation_stats = ShaderCompilationStats {
            shader_count: WebGlShaders::PROGRAM_COUNT * PendingShaderProgram::SHADER_COUNT,
            poll_count: poll_count.get(),
            elapsed: Duration::from_secs_f64((performance_now() - compilation_start_ms) / 1_000.0),
        };

        let filter_uniforms = get_filter_pass_uniforms(&gl, &filter_program)?;
        let blend_uniforms = get_blend_uniforms(&gl, &blend_program)?;
        let copy_uniforms = get_copy_uniforms(&gl, &copy_program)?;
        let strip_uniforms = get_strip_uniforms(&gl, &strip_program)?;

        gl.enable(WebGl2RenderingContext::BLEND);
        gl.blend_func(
            WebGl2RenderingContext::ONE,
            WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
        );

        let programs = WebGlPrograms {
            strip_program,
            strip_uniforms,
            filter_program,
            filter_uniforms,
            blend_program,
            blend_uniforms,
            copy_program,
            copy_uniforms,
            shader_compilation_stats,
        };

        Ok(WebGlRenderer {
            programs,
            resources,
            gl,
            scene_preparer,
        })
    }
}

#[derive(Debug)]
pub(crate) struct PendingShaderProgram {
    vertex_shader: VertexShader,
    fragment_shader: FragmentShader,
    program: Program,
}

impl PendingShaderProgram {
    /// One vertex and one fragment shader.
    const SHADER_COUNT: u32 = 2;

    fn new(
        gl: &WebGl2RenderingContext,
        vertex_src: &str,
        fragment_src: &str,
    ) -> Result<Self, WebGlError> {
        let vertex_shader = VertexShader::new(gl)?;
        gl.shader_source(&vertex_shader, vertex_src);
        gl.compile_shader(&vertex_shader);

        let fragment_shader = FragmentShader::new(gl)?;
        gl.shader_source(&fragment_shader, fragment_src);
        gl.compile_shader(&fragment_shader);

        let program = Program::new(gl)?;
        gl.attach_shader(&program, &vertex_shader);
        gl.attach_shader(&program, &fragment_shader);
        gl.link_program(&program);

        Ok(Self {
            vertex_shader,
            fragment_shader,
            program,
        })
    }

    fn is_complete(&self, gl: &WebGl2RenderingContext, parallel_shader_compile: bool) -> bool {
        /// `COMPLETION_STATUS_KHR` from the `KHR_parallel_shader_compile` specification.
        const COMPLETION_STATUS_KHR: u32 = 0x91B1;
        if !parallel_shader_compile {
            return true;
        }
        gl.get_program_parameter(&self.program, COMPLETION_STATUS_KHR)
            .as_bool()
            .unwrap_or(false)
    }

    fn finish(self, gl: &WebGl2RenderingContext) -> Result<Program, WebGlError> {
        if !gl
            .get_program_parameter(&self.program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            if !gl
                .get_shader_parameter(&self.vertex_shader, WebGl2RenderingContext::COMPILE_STATUS)
                .as_bool()
                .unwrap_or(false)
            {
                let info = gl
                    .get_shader_info_log(&self.vertex_shader)
                    .unwrap_or_else(|| "Unknown error creating vertex shader".into());

                return Err(WebGlError::OperationFailed {
                    operation: WebGlOperation::ShaderProgram(WebGlShaderProgramOperation::Compile(
                        WebGlShaderStage::Vertex,
                    )),
                    message: Some(info),
                });
            }

            if !gl
                .get_shader_parameter(
                    &self.fragment_shader,
                    WebGl2RenderingContext::COMPILE_STATUS,
                )
                .as_bool()
                .unwrap_or(false)
            {
                let info = gl
                    .get_shader_info_log(&self.fragment_shader)
                    .unwrap_or_else(|| "Unknown error creating fragment shader".into());

                return Err(WebGlError::OperationFailed {
                    operation: WebGlOperation::ShaderProgram(WebGlShaderProgramOperation::Compile(
                        WebGlShaderStage::Fragment,
                    )),
                    message: Some(info),
                });
            }

            let info = gl
                .get_program_info_log(&self.program)
                .unwrap_or_else(|| "Unknown error creating program".into());

            return Err(WebGlError::OperationFailed {
                operation: WebGlOperation::ShaderProgram(WebGlShaderProgramOperation::Link),
                message: Some(info),
            });
        }

        Ok(self.program)
    }
}

impl WebGlShaders {
    const PROGRAM_COUNT: u32 = 4;

    /// Creates the canvas's WebGL2 context and starts compiling and linking all shader programs,
    /// returning without waiting for them to finish.
    fn compile(canvas: &HtmlCanvasElement, use_depth_buffer: bool) -> Result<Self, WebGlError> {
        let gl = create_context(canvas, use_depth_buffer)?;
        let (compilation_start_ms, programs) = compile_programs(&gl)?;

        Self::from_programs(gl, use_depth_buffer, compilation_start_ms, programs)
    }

    /// See [`WebGlRenderer::begin_with_precompiled`].
    fn from_precompiled(precompiled: &JsValue) -> Result<Self, WebGlError> {
        let not_a_result = || {
            WebGlError::IncompatiblePrecompiledShaders(
                IncompatiblePrecompiledShadersReason::NotCompileVelloShadersResult,
            )
        };
        let context = precompiled_field(precompiled, "gl")
            .ok()
            .and_then(|gl| gl.dyn_into::<WebGl2RenderingContext>().ok())
            .ok_or_else(not_a_result)?;
        let canvas = context
            .canvas()
            .and_then(|canvas| canvas.dyn_into::<HtmlCanvasElement>().ok())
            .ok_or_else(not_a_result)?;
        let use_depth_buffer = context_attribute(&context, "depth")?;

        // The canvas already has the context `compileVelloShaders` created, so this validates and
        // returns it rather than creating a new one.
        let gl = create_context(&canvas, use_depth_buffer)?;
        let (compilation_start_ms, programs) = adopt_precompiled_programs(&gl, precompiled)
            .map_err(WebGlError::IncompatiblePrecompiledShaders)?;

        Self::from_programs(gl, use_depth_buffer, compilation_start_ms, programs)
    }

    fn from_programs(
        gl: WebGl2RenderingContext,
        use_depth_buffer: bool,
        compilation_start_ms: f64,
        [strip_program, filter_program, blend_program, copy_program]: [PendingShaderProgram; 4],
    ) -> Result<Self, WebGlError> {
        let parallel_shader_compile = gl
            .get_extension("KHR_parallel_shader_compile")
            .map_js_error(WebGlOperation::Context(WebGlContextOperation::Extension))?
            .is_some();

        Ok(Self {
            gl,
            use_depth_buffer,
            parallel_shader_compile,
            compilation_start_ms,
            poll_count: Cell::new(0),
            ready: Cell::new(false),
            strip_program,
            filter_program,
            blend_program,
            copy_program,
        })
    }

    /// See [`WebGlRendererInit::is_ready`].
    fn is_ready(&self) -> bool {
        if !self.ready.get() {
            self.poll_count.set(self.poll_count.get() + 1);
            let ready = [
                &self.strip_program,
                &self.filter_program,
                &self.blend_program,
                &self.copy_program,
            ]
            .into_iter()
            .all(|program| program.is_complete(&self.gl, self.parallel_shader_compile));
            self.ready.set(ready);
        }
        self.ready.get()
    }
}

/// Starts compiling all programs, returning when compilation started and the programs in
/// [`WebGlShaders`]' field order.
pub(crate) fn compile_programs(
    gl: &WebGl2RenderingContext,
) -> Result<(f64, [PendingShaderProgram; 4]), WebGlError> {
    // Must be enabled before compiling: besides the completion status polled by
    // `WebGlShaders::is_ready`, enabling it is what makes Chromium compile on background threads.
    gl.get_extension("KHR_parallel_shader_compile")
        .map_js_error(WebGlOperation::Context(WebGlContextOperation::Extension))?;

    let compilation_start_ms = performance_now();
    let programs = [
        PendingShaderProgram::new(gl, render::VERTEX_SOURCE, render::FRAGMENT_SOURCE)?,
        PendingShaderProgram::new(
            gl,
            filter_shader::VERTEX_SOURCE,
            filter_shader::FRAGMENT_SOURCE,
        )?,
        PendingShaderProgram::new(gl, blend::VERTEX_SOURCE, blend::FRAGMENT_SOURCE)?,
        PendingShaderProgram::new(gl, copy::VERTEX_SOURCE, copy::FRAGMENT_SOURCE)?,
    ];
    // Browsers may buffer WebGL commands until the next flush, which would delay compilation
    // until the caller yields if it keeps the main thread busy instead.
    gl.flush();

    Ok((compilation_start_ms, programs))
}

/// Takes ownership of the programs in a `compileVelloShaders` result whose context is `gl`,
/// returning when compilation started and the programs in [`WebGlShaders`]' field order.
///
/// This avoids WebGL queries, which may block until compilation finishes.
pub(crate) fn adopt_precompiled_programs(
    gl: &WebGl2RenderingContext,
    precompiled: &JsValue,
) -> Result<(f64, [PendingShaderProgram; 4]), IncompatiblePrecompiledShadersReason> {
    use IncompatiblePrecompiledShadersReason::{DifferentModule, NotCompileVelloShadersResult};

    let module_hash = precompiled_field(precompiled, "moduleHash")?
        .as_string()
        .ok_or(NotCompileVelloShadersResult)?;
    if module_hash != vello_gpu_shaders::webgl_bootstrap::MODULE_HASH {
        return Err(DifferentModule);
    }
    let compilation_start_ms = precompiled_field(precompiled, "startTime")?
        .as_f64()
        .ok_or(NotCompileVelloShadersResult)?;
    let programs = precompiled_field(precompiled, "programs")?;

    let adopt = |name: &str| {
        let program = precompiled_field(&programs, name)?;
        let shader = |key| {
            precompiled_field(&program, key)?
                .dyn_into::<WebGlShader>()
                .map_err(|_| NotCompileVelloShadersResult)
        };
        let vertex_shader = shader("vertexShader")?;
        let fragment_shader = shader("fragmentShader")?;
        let program = precompiled_field(&program, "program")?
            .dyn_into::<WebGlProgram>()
            .map_err(|_| NotCompileVelloShadersResult)?;

        Ok(PendingShaderProgram {
            vertex_shader: VertexShader::from_raw(gl, WebGlVertexShader(vertex_shader)),
            fragment_shader: FragmentShader::from_raw(gl, WebGlFragmentShader(fragment_shader)),
            program: Program::from_raw(gl, program),
        })
    };

    Ok((
        compilation_start_ms,
        [
            adopt("render")?,
            adopt("filter")?,
            adopt("blend")?,
            adopt("copy")?,
        ],
    ))
}

/// Returns a field of a `compileVelloShaders` result, or of an object nested in it.
pub(crate) fn precompiled_field(
    object: &JsValue,
    key: &str,
) -> Result<JsValue, IncompatiblePrecompiledShadersReason> {
    js_sys::Reflect::get(object, &key.into())
        .ok()
        .filter(|value| !value.is_undefined() && !value.is_null())
        .ok_or(IncompatiblePrecompiledShadersReason::NotCompileVelloShadersResult)
}
