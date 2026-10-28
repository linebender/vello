// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Renderer, TextureBindings};
use crate::target::RootTarget;
use crate::{ClearSettings, RenderError, RenderSize, Scene, TargetInit};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, Ordering};
use thiserror::Error;
use vello_common::TextureId;
use vello_common::color::palette::css;
use vello_common::image_cache::ImageCache;
use vello_common::paint::{ImageSource, TextureRegion};
use vello_common::pixmap::Pixmap;
use vello_common::probe::{Probe, ProbeFeature};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, Device, TextureFormat};

const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// A wgpu probe whose pixel readback has been submitted but not completed.
#[derive(Debug)]
pub struct WgpuPendingProbe {
    device: Device,
    buffer: Buffer,
    map_state: Arc<AtomicU8>,
    format: TextureFormat,
    width: u16,
    height: u16,
    padded_bytes_per_row: u32,
    elements: Vec<ProbeFeature>,
}

/// Error returned while running a wgpu probe.
#[derive(Debug, Clone, Error)]
pub enum WgpuProbeError {
    /// The probe was requested without any elements.
    #[error("probe requires at least one element")]
    NoElements,
    /// The renderer's render target format can't be read back by the probe.
    ///
    /// Only `Rgba8Unorm` and `Bgra8Unorm` render targets are supported.
    #[error("probing render targets with format {0:?} is not supported")]
    UnsupportedFormat(TextureFormat),
    /// Rendering the probe scene failed.
    #[error(transparent)]
    Render(#[from] RenderError),
    /// Reading the probe result back from the GPU failed.
    #[error("reading the probe result back failed")]
    ReadbackFailed,
}

/// Result of polling the wgpu probe.
#[derive(Debug)]
pub enum WgpuProbeStatus {
    /// The probe is still pending.
    Pending(WgpuPendingProbe),
    /// The probe has finished and the result is available.
    Complete(Probe<RenderError>),
}

impl Renderer {
    /// Conduct a probing operation.
    ///
    /// The drivers of certain devices are known to be buggy and might therefore not work correctly
    /// with Vello GPU. In the best case, this results in an explicit error or a crash, but in the
    /// worst case it can result in a silent failure, where the rendered contents are empty or look
    /// glitchy.
    ///
    /// This method runs a sanity check by drawing a selection of small elements into a small
    /// offscreen texture with the renderer's render target format, and comparing the output
    /// against a reference image. Pass `use_depth_buffer` to match whether your renders use a depth
    /// buffer.
    ///
    /// The probe scene is submitted to the renderer's queue immediately. The returned handle allows
    /// inspecting the result once it has been read back from the GPU, see
    /// [`WgpuPendingProbe::try_finish`].
    ///
    /// Call this before [`render`](Self::render), or submit any encoder passed to `render` first;
    /// otherwise that render will draw with the probe's data.
    pub fn probe(
        &mut self,
        elements: &[ProbeFeature],
        use_depth_buffer: bool,
    ) -> Result<WgpuPendingProbe, WgpuProbeError> {
        let device = self.device.clone();
        let queue = self.queue.clone();
        if elements.is_empty() {
            return Err(WgpuProbeError::NoElements);
        }
        let format = self.target_format;
        if !matches!(
            format,
            TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm
        ) {
            return Err(WgpuProbeError::UnsupportedFormat(format));
        }

        let (width, height) = vello_common::probe::canvas_size(elements);
        let render_size = RenderSize { width, height };
        let size = wgpu::Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        };
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Probe Target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view =
            use_depth_buffer.then(|| Self::create_depth_texture_view(&device, &render_size));

        let probe_image = vello_common::probe::probe_image_pixmap();
        let probe_image_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Probe Image"),
                size: wgpu::Extent3d {
                    width: u32::from(probe_image.width()),
                    height: u32::from(probe_image.height()),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            probe_image.data_as_u8_slice(),
        );
        let probe_texture_id = TextureId(0);
        let mut texture_bindings = TextureBindings::new();
        texture_bindings.insert(
            probe_texture_id,
            probe_image_texture.create_view(&wgpu::TextureViewDescriptor::default()),
        );

        let mut scene = Scene::new(width, height);
        vello_common::probe::draw_scene(
            &mut scene,
            ImageSource::external_texture(
                probe_texture_id,
                TextureRegion::Full,
                probe_image.may_have_transparency(),
            ),
            elements,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Probe Encoder"),
        });
        self.render_scene(
            &scene,
            &mut encoder,
            &render_size,
            &target_view,
            depth_view.as_ref(),
            &ImageCache::new_dummy(),
            &scene.encoded_paints,
            TargetInit::Clear(ClearSettings::Viewport { color: css::WHITE }),
            RootTarget::UserSurface,
            &texture_bindings,
            (0, 0),
        )?;

        let padded_bytes_per_row =
            (u32::from(width) * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Probe Readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            size,
        );
        let map_state = Arc::new(AtomicU8::new(PENDING));
        let callback_state = map_state.clone();
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let state = if result.is_ok() { MAPPED } else { FAILED };
            callback_state.store(state, Ordering::Release);
        });
        queue.submit([encoder.finish()]);

        Ok(WgpuPendingProbe {
            device,
            buffer,
            map_state,
            format,
            width,
            height,
            padded_bytes_per_row,
            elements: elements.to_vec(),
        })
    }
}

impl WgpuPendingProbe {
    /// Try to finish the probe.
    ///
    /// This polls the renderer's device without blocking. In case the result is not available yet,
    /// the pending probe is returned so that it can be checked again later. Otherwise, the probe
    /// result or an error is returned.
    pub fn try_finish(self) -> Result<WgpuProbeStatus, WgpuProbeError> {
        let _ = self.device.poll(wgpu::PollType::Poll);

        match self.map_state.load(Ordering::Acquire) {
            PENDING => Ok(WgpuProbeStatus::Pending(self)),
            MAPPED => Ok(WgpuProbeStatus::Complete(self.finish_success())),
            _ => Err(WgpuProbeError::ReadbackFailed),
        }
    }

    fn finish_success(self) -> Probe<RenderError> {
        let mut pixmap = Pixmap::new(self.width, self.height);
        let row_len = usize::from(self.width) * 4;
        {
            let mapped = self
                .buffer
                .get_mapped_range(..)
                .expect("probe readback buffer must be mapped");
            for (src, dst) in mapped
                .chunks_exact(self.padded_bytes_per_row as usize)
                .zip(pixmap.data_as_u8_slice_mut().chunks_exact_mut(row_len))
            {
                dst.copy_from_slice(&src[..row_len]);
            }
        }
        self.buffer.unmap();

        if self.format == TextureFormat::Bgra8Unorm {
            for pixel in pixmap.data_as_u8_slice_mut().as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }

        Probe::from_actual(pixmap, &self.elements)
    }
}
