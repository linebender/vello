// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renderer backends driven by the harness.

use std::collections::VecDeque;

use vello_common::ResetOptions;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Rect};
use vello_common::multi_atlas::AtlasConfig;
use vello_common::pixmap::Pixmap;
use vello_gpu::{MemorySettings, RenderSettings, WebGlRenderer, WebGlRendererInitStatus};
use wasm_bindgen::{Clamped, JsCast, JsValue};
use web_sys::{
    CanvasRenderingContext2d, ExtDisjointTimerQuery, HtmlCanvasElement, ImageData,
    WebGl2RenderingContext as Gl, WebGlQuery, WebglLoseContext,
};

use crate::fonts::Fonts;
use crate::workload::{Run, TextSink, Workload};
use crate::{InitStats, now, yield_to_event_loop};

/// Per-frame CPU timings in milliseconds.
pub(crate) struct FrameTimings {
    pub(crate) glyphs: usize,
    pub(crate) encode_ms: f64,
    pub(crate) render_ms: f64,
    pub(crate) present_ms: f64,
    pub(crate) sync_ms: f64,
    /// Most recent GPU timer query result, which typically lags a few frames behind.
    pub(crate) gpu_ms: Option<f64>,
}

/// Inputs for drawing one frame.
pub(crate) struct FrameInput<'a> {
    pub(crate) workload: &'a Workload,
    pub(crate) fonts: &'a Fonts,
    pub(crate) view: Affine,
    pub(crate) time_s: f64,
    pub(crate) hint: bool,
    pub(crate) atlas_cache: bool,
    pub(crate) sync_gpu: bool,
}

impl FrameInput<'_> {
    fn draw(&self, sink: &mut impl TextSink) -> usize {
        self.workload.draw(
            sink,
            self.fonts,
            self.view,
            self.time_s,
            self.hint,
            self.atlas_cache,
        )
    }
}

pub(crate) enum Backend {
    Cpu(Box<CpuBackend>),
    Gpu(Box<GpuBackend>),
}

impl Backend {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "Vello CPU",
            Self::Gpu(_) => "Vello GPU (WebGL2)",
        }
    }

    pub(crate) fn size(&self) -> (u16, u16) {
        match self {
            Self::Cpu(backend) => (backend.pixmap.width(), backend.pixmap.height()),
            Self::Gpu(backend) => (backend.scene.width(), backend.scene.height()),
        }
    }

    pub(crate) fn resize(&mut self, width: u16, height: u16) {
        match self {
            Self::Cpu(backend) => backend.resize(width, height),
            Self::Gpu(backend) => backend.resize(width, height),
        }
    }

    pub(crate) fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameTimings, JsValue> {
        match self {
            Self::Cpu(backend) => backend.render(input),
            Self::Gpu(backend) => backend.render(input),
        }
    }

    pub(crate) fn destroy(&mut self) {
        if let Self::Gpu(backend) = self {
            backend.destroy();
        }
    }
}

pub(crate) struct CpuBackend {
    canvas: HtmlCanvasElement,
    context_2d: CanvasRenderingContext2d,
    ctx: vello_cpu::Scene,
    resources: vello_cpu::Resources,
    pixmap: Pixmap,
}

impl CpuBackend {
    pub(crate) fn new(canvas: HtmlCanvasElement) -> Result<(Self, InitStats), JsValue> {
        let start = now();
        let context_2d = canvas
            .get_context("2d")?
            .ok_or("2D canvas context is unavailable")?
            .dyn_into::<CanvasRenderingContext2d>()?;
        let (width, height) = canvas_size(&canvas);
        let backend = Self {
            canvas,
            context_2d,
            ctx: Self::cpu_scene(width, height),
            resources: vello_cpu::Resources::new(),
            pixmap: Pixmap::new(width, height),
        };
        let elapsed = now() - start;
        let stats = InitStats {
            context_ms: elapsed,
            compile_ms: 0.0,
            shader_count: 0,
            poll_count: 0,
            total_ms: elapsed,
        };
        Ok((backend, stats))
    }

    fn cpu_scene(width: u16, height: u16) -> vello_cpu::Scene {
        vello_cpu::Scene::new_with(
            width,
            height,
            vello_cpu::RenderSettings {
                num_threads: 0,
                ..Default::default()
            },
        )
    }

    fn resize(&mut self, width: u16, height: u16) {
        self.canvas.set_width(width.into());
        self.canvas.set_height(height.into());
        self.ctx = Self::cpu_scene(width, height);
        self.pixmap.resize(width, height);
    }

    fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameTimings, JsValue> {
        let start = now();
        self.ctx.reset();
        let glyphs = input.draw(&mut CpuSink {
            ctx: &mut self.ctx,
            resources: &mut self.resources,
        });
        let encoded = now();
        self.ctx.render(&mut self.pixmap, &mut self.resources);
        let rendered = now();
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(self.pixmap.data_as_u8_slice()),
            self.pixmap.width().into(),
            self.pixmap.height().into(),
        )?;
        self.context_2d.put_image_data(&image_data, 0.0, 0.0)?;
        let presented = now();
        Ok(FrameTimings {
            glyphs,
            encode_ms: encoded - start,
            render_ms: rendered - encoded,
            present_ms: presented - rendered,
            sync_ms: 0.0,
            gpu_ms: None,
        })
    }
}

struct CpuSink<'a> {
    ctx: &'a mut vello_cpu::Scene,
    resources: &'a mut vello_cpu::Resources,
}

impl TextSink for CpuSink<'_> {
    fn set_transform(&mut self, transform: Affine) {
        self.ctx.set_transform(transform);
    }

    fn set_color(&mut self, color: AlphaColor<Srgb>) {
        self.ctx.set_paint(color);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.ctx.fill_rect(rect);
    }

    fn fill_glyphs(&mut self, fonts: &Fonts, run: &Run, hint: bool, atlas_cache: bool) {
        let _ = self
            .ctx
            .glyph_run(self.resources, &fonts.face(run.face).data)
            .font_size(run.size)
            .hint(hint)
            .atlas_cache(atlas_cache)
            .fill_glyphs(run.glyphs.iter().copied());
    }
}

pub(crate) struct GpuBackend {
    canvas: HtmlCanvasElement,
    renderer: WebGlRenderer,
    resources: vello_gpu::Resources,
    scene: vello_gpu::Scene,
    timer: Option<GpuTimer>,
    last_gpu_ms: Option<f64>,
}

impl GpuBackend {
    pub(crate) async fn new(
        canvas: HtmlCanvasElement,
        atlas_size: u16,
    ) -> Result<(Self, InitStats), JsValue> {
        let start = now();
        let settings = RenderSettings::default().with_memory_settings(MemorySettings {
            glyph_atlas_config: AtlasConfig {
                atlas_size: (atlas_size, atlas_size),
                ..AtlasConfig::default()
            },
            ..MemorySettings::default()
        });
        let (mut init, resources) =
            WebGlRenderer::begin_with(&canvas, settings, true).map_err(js_error)?;
        let context_ms = now() - start;
        let renderer = loop {
            match init.try_finish().map_err(js_error)? {
                WebGlRendererInitStatus::Complete(renderer) => break renderer,
                WebGlRendererInitStatus::Pending(pending) => {
                    init = pending;
                    yield_to_event_loop().await;
                }
            }
        };
        let total_ms = now() - start;
        let compilation = renderer.shader_compilation_stats();
        let timer = GpuTimer::new(renderer.gl_context().clone());
        let (width, height) = canvas_size(&canvas);
        let backend = Self {
            canvas,
            renderer,
            resources,
            scene: vello_gpu::Scene::new(width, height),
            timer,
            last_gpu_ms: None,
        };
        let stats = InitStats {
            context_ms,
            compile_ms: compilation.elapsed.as_secs_f64() * 1000.0,
            shader_count: compilation.shader_count,
            poll_count: compilation.poll_count,
            total_ms,
        };
        Ok((backend, stats))
    }

    fn gl(&self) -> &Gl {
        self.renderer.gl_context()
    }

    fn resize(&mut self, width: u16, height: u16) {
        self.canvas.set_width(width.into());
        self.canvas.set_height(height.into());
        self.scene.reset_with(ResetOptions {
            size: Some((width, height)),
            ..ResetOptions::default()
        });
        if let Some(timer) = &mut self.timer {
            timer.reset();
        }
        self.last_gpu_ms = None;
    }

    fn render(&mut self, input: &FrameInput<'_>) -> Result<FrameTimings, JsValue> {
        if let Some(gpu_ms) = self.timer.as_mut().and_then(GpuTimer::poll) {
            self.last_gpu_ms = Some(gpu_ms);
        }
        let start = now();
        self.scene.reset();
        let glyphs = input.draw(&mut GpuSink {
            scene: &mut self.scene,
            resources: &mut self.resources,
        });
        let encoded = now();
        if let Some(timer) = &mut self.timer {
            timer.begin();
        }
        self.renderer
            .render(
                &self.scene,
                &mut self.resources,
                vello_gpu::WebGlRenderTarget::DrawingBuffer,
                &vello_gpu::WebGlTextureBindings::new(),
                vello_gpu::WebGlRenderOptions::default(),
            )
            .map_err(js_error)?;
        if let Some(timer) = &mut self.timer {
            timer.end();
        }
        let rendered = now();
        if input.sync_gpu {
            let mut texel = [0_u8; 4];
            self.gl().read_pixels_with_opt_u8_array(
                0,
                0,
                1,
                1,
                Gl::RGBA,
                Gl::UNSIGNED_BYTE,
                Some(&mut texel),
            )?;
        }
        let synced = now();
        Ok(FrameTimings {
            glyphs,
            encode_ms: encoded - start,
            render_ms: rendered - encoded,
            present_ms: 0.0,
            sync_ms: synced - rendered,
            gpu_ms: self.last_gpu_ms,
        })
    }

    fn destroy(&mut self) {
        if let Some(timer) = &mut self.timer {
            timer.reset();
        }
        if let Ok(Some(extension)) = self.gl().get_extension("WEBGL_lose_context") {
            extension
                .unchecked_into::<WebglLoseContext>()
                .lose_context();
        }
    }
}

struct GpuSink<'a> {
    scene: &'a mut vello_gpu::Scene,
    resources: &'a mut vello_gpu::Resources,
}

impl TextSink for GpuSink<'_> {
    fn set_transform(&mut self, transform: Affine) {
        self.scene.set_transform(transform);
    }

    fn set_color(&mut self, color: AlphaColor<Srgb>) {
        self.scene.set_paint(color);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    fn fill_glyphs(&mut self, fonts: &Fonts, run: &Run, hint: bool, atlas_cache: bool) {
        let _ = self
            .scene
            .glyph_run(self.resources, &fonts.face(run.face).data)
            .font_size(run.size)
            .hint(hint)
            .atlas_cache(atlas_cache)
            .fill_glyphs(run.glyphs.iter().copied());
    }
}

const MAX_PENDING_GPU_QUERIES: usize = 8;

/// Non-blocking `EXT_disjoint_timer_query_webgl2` measurements of render submissions.
struct GpuTimer {
    gl: Gl,
    pending: VecDeque<WebGlQuery>,
    active: Option<WebGlQuery>,
}

impl GpuTimer {
    fn new(gl: Gl) -> Option<Self> {
        gl.get_extension("EXT_disjoint_timer_query_webgl2")
            .ok()
            .flatten()?;
        Some(Self {
            gl,
            pending: VecDeque::with_capacity(MAX_PENDING_GPU_QUERIES),
            active: None,
        })
    }

    fn begin(&mut self) {
        if self.active.is_some() || self.pending.len() == MAX_PENDING_GPU_QUERIES {
            return;
        }
        if let Some(query) = self.gl.create_query() {
            self.gl
                .begin_query(ExtDisjointTimerQuery::TIME_ELAPSED_EXT, &query);
            self.active = Some(query);
        }
    }

    fn end(&mut self) {
        if let Some(query) = self.active.take() {
            self.gl.end_query(ExtDisjointTimerQuery::TIME_ELAPSED_EXT);
            self.pending.push_back(query);
        }
    }

    /// Returns the oldest completed measurement in milliseconds, if any.
    ///
    /// All pending results are discarded when the GPU reports a disjoint event, since their
    /// durations are unreliable.
    fn poll(&mut self) -> Option<f64> {
        let disjoint = self
            .gl
            .get_parameter(ExtDisjointTimerQuery::GPU_DISJOINT_EXT)
            .ok()
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if disjoint {
            self.reset();
            return None;
        }
        let available = self
            .gl
            .get_query_parameter(self.pending.front()?, Gl::QUERY_RESULT_AVAILABLE)
            .as_bool()
            .unwrap_or(false);
        if !available {
            return None;
        }
        let query = self.pending.pop_front()?;
        let nanos = self
            .gl
            .get_query_parameter(&query, Gl::QUERY_RESULT)
            .as_f64();
        self.gl.delete_query(Some(&query));
        nanos.map(|nanos| nanos / 1_000_000.0)
    }

    fn reset(&mut self) {
        for query in self.pending.drain(..) {
            self.gl.delete_query(Some(&query));
        }
    }
}

fn canvas_size(canvas: &HtmlCanvasElement) -> (u16, u16) {
    let clamp = |value: u32| u16::try_from(value.max(1)).unwrap_or(u16::MAX);
    (clamp(canvas.width()), clamp(canvas.height()))
}

fn js_error(error: impl core::fmt::Debug) -> JsValue {
    JsValue::from_str(&format!("{error:?}"))
}
