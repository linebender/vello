// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Browser harness for benchmarking and inspecting Glifo text rendering.
//!
//! The same workloads run on Vello CPU (presented via `putImageData`) and Vello GPU (native
//! WebGL2) and report per-frame timings to the pages under `web/`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "canvas sizes and glyph counts fit comfortably in the target types"
)]
#![allow(
    missing_debug_implementations,
    reason = "wasm-bindgen exported types wrap renderer state without Debug"
)]
#![allow(
    missing_docs,
    reason = "the JavaScript-facing API is documented in the README and consumed by web/"
)]
#![cfg(target_arch = "wasm32")]

mod backend;
mod fonts;
mod workload;

use vello_common::kurbo::Affine;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{HtmlCanvasElement, MessageChannel};

use backend::{Backend, CpuBackend, FrameInput, GpuBackend};
use fonts::Fonts;
use workload::{LayoutParams, WORKLOADS, Workload};

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
}

/// Renderer start-up costs, in milliseconds.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub struct InitStats {
    /// Time to create the context and issue shader compilation.
    pub context_ms: f64,
    /// Time from issuing shader compilation until all programs were linked.
    pub compile_ms: f64,
    pub shader_count: u32,
    pub poll_count: u32,
    /// Wall-clock time until the renderer was ready, including event-loop yields.
    pub total_ms: f64,
}

/// Timings of one frame.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub frame: u32,
    pub glyphs: u32,
    pub encode_ms: f64,
    pub render_ms: f64,
    pub present_ms: f64,
    pub sync_ms: f64,
    /// Latest GPU timer result, `NaN` when unavailable. Usually lags a few frames.
    pub gpu_ms: f64,
    pub total_ms: f64,
}

/// One renderer plus the workload state it draws.
#[wasm_bindgen]
pub struct Harness {
    backend: Backend,
    fonts: Fonts,
    workload: Workload,
    font_scale: f32,
    density: f32,
    hinting: bool,
    atlas_cache: bool,
    view: Affine,
    frame: u32,
    init: InitStats,
}

#[wasm_bindgen]
impl Harness {
    /// Creates a Vello GPU harness rendering into `canvas` with WebGL2.
    ///
    /// `atlas_size` is the side length of each square atlas page, in texels.
    #[wasm_bindgen(js_name = createGpu)]
    pub async fn create_gpu(canvas: HtmlCanvasElement, atlas_size: u32) -> Result<Self, JsValue> {
        let atlas_size = u16::try_from(atlas_size.clamp(256, 8192)).unwrap_or(4096);
        let (backend, init) = GpuBackend::new(canvas, atlas_size).await?;
        Ok(Self::new(Backend::Gpu(Box::new(backend)), init))
    }

    /// Creates a Vello CPU harness that presents into `canvas` with `putImageData`.
    #[wasm_bindgen(js_name = createCpu)]
    pub async fn create_cpu(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let (backend, init) = CpuBackend::new(canvas)?;
        Ok(Self::new(Backend::Cpu(Box::new(backend)), init))
    }

    fn new(backend: Backend, init: InitStats) -> Self {
        Self {
            backend,
            fonts: Fonts::new(),
            workload: Workload::new(WORKLOADS[0].0).expect("default workload exists"),
            font_scale: 1.0,
            density: 1.0,
            hinting: true,
            atlas_cache: true,
            view: Affine::IDENTITY,
            frame: 0,
            init,
        }
    }

    /// Returns `[name, description]` pairs for every workload.
    pub fn workloads() -> js_sys::Array {
        WORKLOADS
            .iter()
            .map(|(name, description)| {
                js_sys::Array::of2(&JsValue::from_str(name), &JsValue::from_str(description))
            })
            .collect()
    }

    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        self.backend.name().into()
    }

    #[wasm_bindgen(js_name = initStats)]
    pub fn init_stats(&self) -> InitStats {
        self.init
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.backend.size().0.into()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.backend.size().1.into()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let clamp = |value: u32| u16::try_from(value.clamp(1, 8192)).unwrap_or(1);
        let size = (clamp(width), clamp(height));
        if size != self.backend.size() {
            self.backend.resize(size.0, size.1);
            self.workload.invalidate();
        }
    }

    #[wasm_bindgen(js_name = setWorkload)]
    pub fn set_workload(&mut self, name: &str) -> Result<(), JsValue> {
        self.workload = Workload::new(name).ok_or_else(|| JsValue::from_str("unknown workload"))?;
        Ok(())
    }

    pub fn configure(&mut self, font_scale: f32, density: f32, hinting: bool, atlas_cache: bool) {
        self.font_scale = font_scale.clamp(0.25, 8.0);
        self.density = density.clamp(0.05, 20.0);
        self.hinting = hinting;
        self.atlas_cache = atlas_cache;
    }

    /// Sets a pan/zoom transform applied on top of the workload's own animation.
    #[wasm_bindgen(js_name = setView)]
    pub fn set_view(&mut self, scale: f64, translate_x: f64, translate_y: f64) {
        self.view = Affine::translate((translate_x, translate_y)) * Affine::scale(scale);
    }

    /// Renders one frame at animation time `time_ms`.
    ///
    /// With `sync_gpu`, the GPU backend waits for rendering to finish so `total_ms` includes GPU
    /// work; this stalls the pipeline and is intended for benchmarking.
    pub fn render(&mut self, time_ms: f64, sync_gpu: bool) -> Result<FrameStats, JsValue> {
        let start = now();
        let (width, height) = self.backend.size();
        self.workload.prepare(
            &self.fonts,
            LayoutParams {
                width: width.into(),
                height: height.into(),
                font_scale: self.font_scale,
                density: self.density,
            },
        );
        let timings = self.backend.render(&FrameInput {
            workload: &self.workload,
            fonts: &self.fonts,
            view: self.view,
            time_s: time_ms / 1000.0,
            hint: self.hinting,
            atlas_cache: self.atlas_cache,
            sync_gpu,
        })?;
        let total_ms = now() - start;
        self.frame += 1;
        Ok(FrameStats {
            frame: self.frame,
            glyphs: timings.glyphs as u32,
            encode_ms: timings.encode_ms,
            render_ms: timings.render_ms,
            present_ms: timings.present_ms,
            sync_ms: timings.sync_ms,
            gpu_ms: timings.gpu_ms.unwrap_or(f64::NAN),
            total_ms,
        })
    }

    /// Releases GPU resources eagerly instead of waiting for garbage collection.
    pub fn destroy(&mut self) {
        self.backend.destroy();
    }
}

pub(crate) fn now() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or(0.0, |performance| performance.now())
}

/// Yields to the browser event loop for one task.
///
/// Uses a `MessageChannel` rather than `setTimeout`, which is clamped to 4 ms when nested and
/// throttled to as little as once per minute in hidden tabs; either would distort the measured
/// shader compilation time.
pub(crate) async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let Ok(channel) = MessageChannel::new() else {
            let _ = reject.call0(&JsValue::NULL);
            return;
        };
        channel.port1().set_onmessage(Some(&resolve));
        if channel.port2().post_message(&JsValue::NULL).is_err() {
            let _ = reject.call0(&JsValue::NULL);
        }
    });
    let _ = JsFuture::from(promise).await;
}
