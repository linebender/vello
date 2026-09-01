// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::render::webgl::resource::Framebuffer;
use crate::render::webgl::{
    WebGlStateConfig, WebGlStateGuard, WebGlTextureBindings, create_framebuffer_for_texture,
    create_texture_storage,
};
use crate::target::RootTarget;
use crate::{ClearSettings, RenderError, RenderSize, Scene, TargetInit, WebGlRenderer};
use alloc::{borrow::Cow, format};
use core::ops::Deref;
use thiserror::Error;
use vello_common::TextureId;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::image_cache::ImageCache;
use vello_common::kurbo::{Affine, BezPath, Rect};
use vello_common::paint::{ImageSource, PaintType};
use vello_common::peniko::BlendMode;
use vello_common::pixmap::Pixmap;
use vello_common::probe::Probe;
use web_sys::{WebGl2RenderingContext, WebGlBuffer, WebGlSync};

/// A WebGL probe whose pixel readback has been queued but not completed.
#[derive(Debug)]
pub struct WebGlPendingProbe {
    gl: WebGl2RenderingContext,
    sync: Option<WebGlSync>,
    buffer: Option<WebGlBuffer>,
    width: u16,
    height: u16,
}

/// Error returned while running a WebGL probe.
#[derive(Debug, Clone, Error)]
pub enum WebGlProbeError {
    /// Rendering the probe scene failed.
    #[error("probe render failed: {0}")]
    Render(RenderError),
    /// Finishing the probe failed.
    #[error("probe failed to finish: {}", webgl_error_name(*.0))]
    FinishFailed(u32),
}

/// Result of polling the WebGL probe.
#[derive(Debug)]
pub enum WebGlProbeStatus {
    /// The probe is still pending.
    Pending(WebGlPendingProbe),
    /// The probe has finished and the result is available.
    Complete(Probe<RenderError>),
}

impl WebGlRenderer {
    /// Conduct a probing operation.
    ///
    /// The WebGL drivers of certain devices are known to be buggy and might therefore not work correctly
    /// with Vello Hybrid. In the best case, it will simply result in a program crash, but in the worst
    /// case it can instead result in a silent failure, meaning that no explicit error is
    /// thrown, but the rendered contents of Vello Hybrid will either be completely empty or look glitchy.
    ///
    /// The purpose of this method is to run a sanity check to ensure that running Vello Hybrid on this
    /// device actually results in visible and correct output. How this achieved is by drawing a selection
    /// of small elements into a small canvas, and comparing the final output against a reference image.
    ///
    /// This method will return a handle that allows inspecting the results of the probe once the
    /// results of the probe scene can be copied back from GPU to CPU. For performance reasons,
    /// anything in-between mostly happens asynchronously.
    pub fn probe(&mut self) -> Result<WebGlPendingProbe, WebGlProbeError> {
        self.probe_inner().map_err(WebGlProbeError::Render)
    }

    fn probe_inner(&mut self) -> Result<WebGlPendingProbe, RenderError> {
        // IMPORTANT NOTE: When making any changes to the probe, make sure to
        // unignore and rerun the "webgl_probe_succeeds" test locally.

        let _state_guard = WebGlStateGuard::with_config(
            &self.gl,
            WebGlStateConfig {
                framebuffer: true,
                active_texture: true,
                texture_2d: true,
                pixel_pack_buffer: true,
                viewport: true,
                ..Default::default()
            },
        );

        let (width, height) = vello_common::probe::canvas_size();
        let render_size = RenderSize {
            width: u32::from(width),
            height: u32::from(height),
        };

        let probe_texture = create_texture_storage(
            &self.gl,
            WebGl2RenderingContext::RGBA8,
            render_size.width,
            render_size.height,
            WebGl2RenderingContext::NEAREST,
            WebGl2RenderingContext::NEAREST,
        );
        let probe_framebuffer = create_framebuffer_for_texture(&self.gl, &probe_texture);

        let probe_image = vello_common::probe::probe_image_pixmap();
        let probe_image_texture = create_texture_storage(
            &self.gl,
            WebGl2RenderingContext::RGBA8,
            u32::from(probe_image.width()),
            u32::from(probe_image.height()),
            WebGl2RenderingContext::NEAREST,
            WebGl2RenderingContext::NEAREST,
        );

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
            .unwrap();

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
        );

        let previous_view_framebuffer = self
            .programs
            .resources
            .view_framebuffer_override
            .replace(probe_framebuffer);
        let render_result = self.render_scene(
            &scene,
            &ImageCache::new_dummy(),
            &render_size,
            TargetInit::Clear(ClearSettings::default()),
            RootTarget::AtlasLayer,
            &texture_bindings,
            Some(&probe_texture),
        );
        let probe_framebuffer = self
            .programs
            .resources
            .view_framebuffer_override
            .take()
            .expect("probe framebuffer must be restored after rendering");
        self.programs.resources.view_framebuffer_override = previous_view_framebuffer;

        // Propagate render failures only after restoring the framebuffer override.
        render_result?;

        let pending = launch_probe(&self.gl, &probe_framebuffer, width, height);

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
        let status = self.gl.client_wait_sync_with_u32(
            self.sync.as_ref().expect("probe sync must exist"),
            0,
            0,
        );

        if status == WebGl2RenderingContext::TIMEOUT_EXPIRED {
            return Ok(WebGlProbeStatus::Pending(self));
        }

        if status == WebGl2RenderingContext::ALREADY_SIGNALED
            || status == WebGl2RenderingContext::CONDITION_SATISFIED
        {
            Ok(WebGlProbeStatus::Complete(self.finish_success()))
        } else {
            Err(self.finish_failure())
        }
    }

    fn finish_success(&mut self) -> Probe<RenderError> {
        let _state_guard = WebGlStateGuard::with_config(
            &self.gl,
            WebGlStateConfig {
                pixel_pack_buffer: true,
                ..Default::default()
            },
        );
        let mut pixmap = Pixmap::new(self.width, self.height);

        self.gl.bind_buffer(
            WebGl2RenderingContext::PIXEL_PACK_BUFFER,
            self.buffer.as_ref(),
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

        Probe::from_actual(pixmap)
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

impl Drop for WebGlPendingProbe {
    fn drop(&mut self) {
        if let Some(sync) = self.sync.take() {
            self.gl.delete_sync(Some(&sync));
        }
        if let Some(buffer) = self.buffer.take() {
            self.gl.delete_buffer(Some(&buffer));
        }
    }
}

#[cfg(feature = "probe")]
fn launch_probe(
    gl: &WebGl2RenderingContext,
    framebuffer: &Framebuffer,
    width: u16,
    height: u16,
) -> WebGlPendingProbe {
    let pixel_pack_buffer = gl.create_buffer().unwrap();
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
    .unwrap();
    // Create a fence that notifies us once rendering is complete and the contents have been
    // transferred from the framebuffer to the pixel pack buffer.
    let sync = gl
        .fence_sync(WebGl2RenderingContext::SYNC_GPU_COMMANDS_COMPLETE, 0)
        .unwrap();
    // https://wikis.khronos.org/opengl/Sync_Object
    // "It is important that syncs are properly flushed into the GPU's command queue. Without
    // proper flushing, the sync object may never be signaled."
    gl.flush();
    gl.bind_buffer(WebGl2RenderingContext::PIXEL_PACK_BUFFER, None);

    WebGlPendingProbe {
        gl: gl.clone(),
        sync: Some(sync),
        buffer: Some(pixel_pack_buffer),
        width,
        height,
    }
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
