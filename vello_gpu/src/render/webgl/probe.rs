// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::render::webgl::resource::{Buffer, Framebuffer, Renderbuffer, SyncFence};
use crate::render::webgl::{
    ViewFramebuffer, WebGlOperation, WebGlProbeOperation, WebGlResultExt, WebGlTextureBindings,
    create_framebuffer_for_texture, create_texture_storage, elapsed, now_ms,
};
use crate::target::RootTarget;
use crate::{ClearSettings, RenderError, RenderSize, Scene, TargetInit, WebGlError, WebGlRenderer};
use alloc::{borrow::Cow, format, vec::Vec};
use core::{ops::Deref, time::Duration};
use thiserror::Error;
use vello_common::TextureId;
use vello_common::color::palette::css;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::image_cache::ImageCache;
use vello_common::kurbo::{Affine, BezPath, Rect};
use vello_common::paint::{ImageSource, PaintType};
use vello_common::peniko::BlendMode;
use vello_common::pixmap::Pixmap;
use vello_common::probe::{Probe, ProbeFeature};
use web_sys::WebGl2RenderingContext;

/// A WebGL probe whose pixel readback has been queued but not completed.
#[derive(Debug)]
pub struct WebGlPendingProbe {
    gl: WebGl2RenderingContext,
    sync: SyncFence,
    buffer: Buffer,
    width: u16,
    height: u16,
    elements: Vec<ProbeFeature>,
    timing: ProbeTimingState,
}

#[derive(Debug)]
struct ProbeTimingState {
    probe_started_at_ms: f64,
    readback_submitted_at_ms: f64,
    setup_duration: Duration,
    render_submission_duration: Duration,
    readback_submission_duration: Duration,
    poll_count: u32,
}

impl ProbeTimingState {
    fn new(
        probe_started_at_ms: f64,
        readback_submitted_at_ms: f64,
        setup_duration: Duration,
        render_submission_duration: Duration,
        readback_submission_duration: Duration,
    ) -> Self {
        Self {
            probe_started_at_ms,
            readback_submitted_at_ms,
            setup_duration,
            render_submission_duration,
            readback_submission_duration,
            poll_count: 0,
        }
    }

    fn record_poll(&mut self) {
        self.poll_count = self.poll_count.saturating_add(1);
    }

    fn finish(
        &self,
        fence_signal_observed_at_ms: f64,
        readback_duration: Duration,
        probe_result_produced_at_ms: f64,
    ) -> WebGlProbeTimings {
        WebGlProbeTimings {
            setup: self.setup_duration,
            render_submission: self.render_submission_duration,
            readback_submission: self.readback_submission_duration,
            completion_latency: elapsed(self.readback_submitted_at_ms, fence_signal_observed_at_ms),
            readback: readback_duration,
            total: elapsed(self.probe_started_at_ms, probe_result_produced_at_ms),
        }
    }
}

/// Completed WebGL probe output.
#[derive(Debug)]
pub struct WebGlProbeReport {
    /// Whether the rendered probe matched the reference image.
    pub outcome: Probe<RenderError>,
    /// Time spent in each phase of the probe.
    pub timings: WebGlProbeTimings,
    /// Number of calls to [`WebGlPendingProbe::try_finish`] before completion.
    pub poll_count: u32,
}

impl WebGlProbeReport {
    /// Returns `true` when the probe matched the bundled reference image.
    pub fn is_success(&self) -> bool {
        self.outcome.is_success()
    }
}

/// Time spent in each phase of a WebGL probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGlProbeTimings {
    /// CPU time spent allocating resources, uploading the probe image, and building the scene.
    pub setup: Duration,
    /// CPU time spent executing the probe's internal `render_scene` call.
    pub render_submission: Duration,
    /// CPU time spent allocating the pixel-pack buffer, queuing the pixel readback, creating its
    /// fence, and flushing WebGL commands.
    pub readback_submission: Duration,
    /// Time from submitting the pixel readback until its fence was first observed as signaled.
    ///
    /// This includes GPU rendering and readback as well as delays between polls.
    pub completion_latency: Duration,
    /// CPU time spent reading the completed pixel buffer back, copying it into WASM memory, and
    /// flipping it vertically.
    pub readback: Duration,
    /// Wall-clock time from starting the probe until its result was produced.
    pub total: Duration,
}

/// Error returned while running a WebGL probe.
#[derive(Debug, Clone, Error)]
pub enum WebGlProbeError {
    /// The probe was requested without any elements.
    #[error("probe requires at least one element")]
    NoElements,
    /// Finishing the probe failed.
    #[error("probe failed to finish: {}", webgl_error_name(*.0))]
    FinishFailed(u32),
    /// A WebGL operation failed.
    #[error(transparent)]
    WebGl(#[from] WebGlError),
}

/// Result of polling the WebGL probe.
#[derive(Debug)]
pub enum WebGlProbeStatus {
    /// The probe is still pending.
    Pending(WebGlPendingProbe),
    /// The probe has finished and the result is available.
    Complete(WebGlProbeReport),
}

impl WebGlRenderer {
    /// Conduct a probing operation.
    ///
    /// The WebGL drivers of certain devices are known to be buggy and might therefore not work correctly
    /// with Vello GPU. In the best case, it will simply result in a program crash, but in the worst
    /// case it can instead result in a silent failure, meaning that no explicit error is
    /// thrown, but the rendered contents of Vello GPU will either be completely empty or look glitchy.
    ///
    /// The purpose of this method is to run a sanity check to ensure that running Vello GPU on this
    /// device actually results in visible and correct output. How this achieved is by drawing a selection
    /// of small elements into a small canvas, and comparing the final output against a reference image.
    ///
    /// This method will return a handle that allows inspecting the results of the probe once the
    /// results of the probe scene can be copied back from GPU to CPU. For performance reasons,
    /// anything in-between mostly happens asynchronously.
    pub fn probe(
        &mut self,
        elements: &[ProbeFeature],
    ) -> Result<WebGlPendingProbe, WebGlProbeError> {
        if elements.is_empty() {
            return Err(WebGlProbeError::NoElements);
        }

        self.probe_inner(elements).map_err(WebGlProbeError::WebGl)
    }

    fn probe_inner(&mut self, elements: &[ProbeFeature]) -> Result<WebGlPendingProbe, WebGlError> {
        let probe_started_at_ms = now_ms();
        // Whenever making changes here, make sure to unignore the `webgl_probe_succeeds_` and
        // run them locally!
        let (width, height) = vello_common::probe::canvas_size(elements);
        let render_size = RenderSize { width, height };
        // Check whether the canvas framebuffer was configured by the user to
        // hold a depth buffer, and apply the same to the probe.
        let use_depth_buffer = self.programs.resources.view_framebuffer.use_depth_buffer();

        let probe_texture = create_texture_storage(
            &self.gl,
            WebGl2RenderingContext::RGBA8,
            u32::from(render_size.width),
            u32::from(render_size.height),
            WebGl2RenderingContext::NEAREST,
            WebGl2RenderingContext::NEAREST,
        )?;
        let probe_framebuffer = create_framebuffer_for_texture(&self.gl, &probe_texture)?;
        // We need to keep the renderbuffer around until we finished rendering!
        let _probe_depth = if use_depth_buffer {
            let probe_depth = Renderbuffer::new(&self.gl)?;
            self.gl
                .bind_renderbuffer(WebGl2RenderingContext::RENDERBUFFER, Some(&probe_depth));
            self.gl.renderbuffer_storage(
                WebGl2RenderingContext::RENDERBUFFER,
                WebGl2RenderingContext::DEPTH_COMPONENT24,
                i32::from(width),
                i32::from(height),
            );
            self.gl.framebuffer_renderbuffer(
                WebGl2RenderingContext::FRAMEBUFFER,
                WebGl2RenderingContext::DEPTH_ATTACHMENT,
                WebGl2RenderingContext::RENDERBUFFER,
                Some(&probe_depth),
            );
            Some(probe_depth)
        } else {
            None
        };

        let probe_image = vello_common::probe::probe_image_pixmap();
        let probe_image_texture = create_texture_storage(
            &self.gl,
            WebGl2RenderingContext::RGBA8,
            u32::from(probe_image.width()),
            u32::from(probe_image.height()),
            WebGl2RenderingContext::NEAREST,
            WebGl2RenderingContext::NEAREST,
        )?;

        self.gl
            .tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                0,
                0,
                i32::from(probe_image.width()),
                i32::from(probe_image.height()),
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(probe_image.data_as_u8_slice()),
            )
            .map_js_error(WebGlOperation::Probe(WebGlProbeOperation::ImageUpload))?;

        let probe_texture_id = TextureId(0);
        let mut texture_bindings = WebGlTextureBindings::new();
        texture_bindings.insert(probe_texture_id, probe_image_texture.deref().clone());
        let mut scene = Scene::new(width, height);
        vello_common::probe::draw_scene(
            &mut scene,
            ImageSource::external_texture(
                probe_texture_id,
                RectU16::new(0, 0, probe_image.width(), probe_image.height()),
                probe_image.may_have_transparency(),
            ),
            elements,
        );
        let setup_duration = elapsed(probe_started_at_ms, now_ms());

        let previous_view_framebuffer = core::mem::replace(
            &mut self.programs.resources.view_framebuffer,
            ViewFramebuffer::offscreen(probe_framebuffer, use_depth_buffer),
        );
        let render_submission_started_at_ms = now_ms();
        let render_result = self.render_scene(
            &scene,
            &ImageCache::new_dummy(),
            &render_size,
            TargetInit::Clear(ClearSettings::Viewport { color: css::WHITE }),
            RootTarget::UserSurface,
            &texture_bindings,
            Some(&probe_texture),
        );
        let render_submission_duration = elapsed(render_submission_started_at_ms, now_ms());
        let probe_framebuffer = core::mem::replace(
            &mut self.programs.resources.view_framebuffer,
            previous_view_framebuffer,
        )
        .into_framebuffer()
        .unwrap();

        // Propagate render failures only after restoring the previous framebuffer.
        render_result?;

        let pending = launch_probe(
            &self.gl,
            &probe_framebuffer,
            width,
            height,
            elements,
            probe_started_at_ms,
            setup_duration,
            render_submission_duration,
        )?;

        Ok(pending)
    }
}

#[cfg(feature = "probe")]
impl WebGlPendingProbe {
    /// Try to finish the probe.
    ///
    /// In case the result is not available yet, a new pending probe object will be returned
    /// which can be checked again in the future. Otherwise, the probe result or an error will be
    /// returned.
    pub fn try_finish(mut self) -> Result<WebGlProbeStatus, WebGlProbeError> {
        self.timing.record_poll();

        let status = self.gl.client_wait_sync_with_u32(&self.sync, 0, 0);

        if status == WebGl2RenderingContext::TIMEOUT_EXPIRED {
            return Ok(WebGlProbeStatus::Pending(self));
        }

        if status == WebGl2RenderingContext::ALREADY_SIGNALED
            || status == WebGl2RenderingContext::CONDITION_SATISFIED
        {
            let fence_signal_observed_at_ms = now_ms();
            let (outcome, readback_duration, probe_result_produced_at_ms) = self.finish_success();
            Ok(WebGlProbeStatus::Complete(WebGlProbeReport {
                outcome,
                timings: self.timing.finish(
                    fence_signal_observed_at_ms,
                    readback_duration,
                    probe_result_produced_at_ms,
                ),
                poll_count: self.timing.poll_count,
            }))
        } else {
            Err(self.finish_failure())
        }
    }

    fn finish_success(&mut self) -> (Probe<RenderError>, Duration, f64) {
        let readback_started_at_ms = now_ms();
        let mut pixmap = Pixmap::new(self.width, self.height);

        self.gl.bind_buffer(
            WebGl2RenderingContext::PIXEL_PACK_BUFFER,
            Some(&self.buffer),
        );
        // Safari 15 crashes the tab when attempting to read from a pixel pack buffer directly
        // into WASM-allocated memory. Therefore, we first read it into a JS-allocated buffer and
        // only then transfer it into the buffer backing the pixmap in WASM memory.
        let readback =
            js_sys::Uint8Array::new_with_length(u32::from(self.width) * u32::from(self.height) * 4);
        self.gl.get_buffer_sub_data_with_i32_and_js_u8_array(
            WebGl2RenderingContext::PIXEL_PACK_BUFFER,
            0,
            &readback,
        );
        readback.copy_to(pixmap.data_as_u8_slice_mut());

        // Need to flip the resulting image upside down.
        let row_len = usize::from(self.width) * 4;
        let height = usize::from(self.height);
        let pixels = pixmap.data_as_u8_slice_mut();
        for row in 0..height / 2 {
            let opposite_row = height - 1 - row;
            let (top, bottom) = pixels.split_at_mut(opposite_row * row_len);
            top[row * row_len..(row + 1) * row_len].swap_with_slice(&mut bottom[..row_len]);
        }

        let readback_duration = elapsed(readback_started_at_ms, now_ms());
        let outcome = Probe::from_actual(pixmap, &self.elements);
        let probe_result_produced_at_ms = now_ms();
        (outcome, readback_duration, probe_result_produced_at_ms)
    }

    fn finish_failure(&self) -> WebGlProbeError {
        WebGlProbeError::FinishFailed(self.gl.get_error())
    }
}

fn webgl_error_name(error: u32) -> Cow<'static, str> {
    let name = match error {
        WebGl2RenderingContext::NO_ERROR => "NO_ERROR",
        WebGl2RenderingContext::INVALID_ENUM => "INVALID_ENUM",
        WebGl2RenderingContext::INVALID_VALUE => "INVALID_VALUE",
        WebGl2RenderingContext::INVALID_OPERATION => "INVALID_OPERATION",
        WebGl2RenderingContext::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FRAMEBUFFER_OPERATION",
        WebGl2RenderingContext::OUT_OF_MEMORY => "OUT_OF_MEMORY",
        WebGl2RenderingContext::CONTEXT_LOST_WEBGL => "CONTEXT_LOST_WEBGL",
        _ => return Cow::Owned(format!("UNKNOWN_WEBGL_ERROR ({error:#06x})")),
    };

    Cow::Borrowed(name)
}

#[cfg(feature = "probe")]
fn launch_probe(
    gl: &WebGl2RenderingContext,
    framebuffer: &Framebuffer,
    width: u16,
    height: u16,
    elements: &[ProbeFeature],
    probe_started_at_ms: f64,
    setup_duration: Duration,
    render_submission_duration: Duration,
) -> Result<WebGlPendingProbe, WebGlError> {
    let readback_submission_started_at_ms = now_ms();
    let pixel_pack_buffer = Buffer::new(gl)?;
    let byte_len = i32::from(width) * i32::from(height) * 4;

    gl.bind_buffer(
        WebGl2RenderingContext::PIXEL_PACK_BUFFER,
        Some(&pixel_pack_buffer),
    );
    gl.buffer_data_with_i32(
        WebGl2RenderingContext::PIXEL_PACK_BUFFER,
        byte_len,
        WebGl2RenderingContext::STREAM_READ,
    );
    gl.bind_framebuffer(
        WebGl2RenderingContext::FRAMEBUFFER,
        Some(framebuffer.deref()),
    );
    gl.read_pixels_with_i32(
        0,
        0,
        i32::from(width),
        i32::from(height),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        0,
    )
    .map_js_error(WebGlOperation::Probe(WebGlProbeOperation::Readback))?;
    // Create a fence that notifies us once rendering is complete and the contents have been
    // transferred from the framebuffer to the pixel pack buffer.
    let sync = SyncFence::new(gl)?;
    // https://wikis.khronos.org/opengl/Sync_Object
    // "It is important that syncs are properly flushed into the GPU's command queue. Without
    // proper flushing, the sync object may never be signaled."
    gl.flush();
    gl.bind_buffer(WebGl2RenderingContext::PIXEL_PACK_BUFFER, None);
    let readback_submitted_at_ms = now_ms();
    let readback_submission_duration =
        elapsed(readback_submission_started_at_ms, readback_submitted_at_ms);

    Ok(WebGlPendingProbe {
        gl: gl.clone(),
        sync,
        buffer: pixel_pack_buffer,
        width,
        height,
        elements: elements.to_vec(),
        timing: ProbeTimingState::new(
            probe_started_at_ms,
            readback_submitted_at_ms,
            setup_duration,
            render_submission_duration,
            readback_submission_duration,
        ),
    })
}

impl vello_common::probe::ProbeRenderer for Scene {
    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }

    fn set_paint(&mut self, paint: PaintType) {
        Self::set_paint(self, paint);
    }

    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    fn push_layer(&mut self, blend_mode: Option<BlendMode>, opacity: Option<f32>) {
        Self::push_layer(self, None, blend_mode, opacity, None, None);
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        Self::push_filter_layer(self, filter);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    fn set_paint_transform(&mut self, paint_transform: Affine) {
        Self::set_paint_transform(self, paint_transform);
    }

    fn reset_paint_transform(&mut self) {
        Self::reset_paint_transform(self);
    }
}
