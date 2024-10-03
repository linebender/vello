// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple helpers for managing wgpu state and surfaces.

use std::future::Future;

use wgpu::{
    Adapter, Device, Instance, Limits, Queue, Surface, SurfaceConfiguration, SurfaceTarget,
    TextureFormat,
};

use crate::{Error, Result};

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct RenderContext {
    pub instance: Instance,
    pub devices: Vec<DeviceHandle>,
}

pub struct DeviceHandle {
    adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl RenderContext {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            ..Default::default()
        });
        Self {
            instance,
            devices: Vec::new(),
        }
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<RenderSurface<'w>> {
        self.create_render_surface(
            self.instance.create_surface(window.into())?,
            width,
            height,
            present_mode,
        )
        .await
    }

    /// Creates a new render surface for the specified window and dimensions.
    pub async fn create_render_surface<'w>(
        &mut self,
        surface: Surface<'w>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<RenderSurface<'w>> {
        let dev_id = self
            .device(Some(&surface))
            .await
            .ok_or(Error::NoCompatibleDevice)?;

        let device_handle = &self.devices[dev_id];
        let capabilities = surface.get_capabilities(&device_handle.adapter);
        let format = capabilities
            .formats
            .into_iter()
            .find(|it| matches!(it, TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm))
            .ok_or(Error::UnsupportedSurfaceFormat)?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let surface = RenderSurface {
            surface,
            config,
            dev_id,
            format,
        };
        self.configure_surface(&surface);
        Ok(surface)
    }

    /// Resizes the surface to the new dimensions.
    pub fn resize_surface(&self, surface: &mut RenderSurface, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        self.configure_surface(surface);
    }

    pub fn set_present_mode(&self, surface: &mut RenderSurface, present_mode: wgpu::PresentMode) {
        surface.config.present_mode = present_mode;
        self.configure_surface(surface);
    }

    fn configure_surface(&self, surface: &RenderSurface) {
        let device = &self.devices[surface.dev_id].device;
        surface.surface.configure(device, &surface.config);
    }

    /// Finds or creates a compatible device handle id.
    pub async fn device(&mut self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        let compatible = match compatible_surface {
            Some(s) => self
                .devices
                .iter()
                .enumerate()
                .find(|(_, d)| d.adapter.is_surface_supported(s))
                .map(|(i, _)| i),
            None => (!self.devices.is_empty()).then_some(0),
        };
        if compatible.is_none() {
            return self.new_device(compatible_surface).await;
        }
        compatible
    }

    /// Creates a compatible device handle id.
    async fn new_device(&mut self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&self.instance, compatible_surface)
                .await?;
        let features = adapter.features();
        let limits = Limits::default();
        #[allow(unused_mut)]
        let mut maybe_features = wgpu::Features::CLEAR_TEXTURE;
        #[cfg(feature = "wgpu-profiler")]
        {
            maybe_features |= wgpu_profiler::GpuProfiler::ALL_WGPU_TIMER_FEATURES;
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features & maybe_features,
                    required_limits: limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;
        let device_handle = DeviceHandle {
            adapter,
            device,
            queue,
        };
        self.devices.push(device_handle);
        Some(self.devices.len() - 1)
    }
}

impl DeviceHandle {
    /// Returns the adapter associated with the device.
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }
}

/// Combination of surface and its configuration.
#[derive(Debug)]
pub struct RenderSurface<'s> {
    pub surface: Surface<'s>,
    pub config: SurfaceConfiguration,
    pub dev_id: usize,
    pub format: TextureFormat,
}

struct NullWake;

impl std::task::Wake for NullWake {
    fn wake(self: std::sync::Arc<Self>) {}
}

/// Block on a future, polling the device as needed.
///
/// This will deadlock if the future is awaiting anything other than GPU progress.
#[cfg_attr(docsrs, doc(hidden))]
pub fn block_on_wgpu<F: Future>(device: &Device, mut fut: F) -> F::Output {
    if cfg!(target_arch = "wasm32") {
        panic!("Blocking can't work on WASM, so don't try");
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(NullWake));
    let mut context = std::task::Context::from_waker(&waker);
    // Same logic as `pin_mut!` macro from `pin_utils`.
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut context) {
            std::task::Poll::Pending => {
                device.poll(wgpu::Maintain::Wait);
            }
            std::task::Poll::Ready(item) => break item,
        }
    }
}
