// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple helpers for managing wgpu state and surfaces.

use std::{future::Future, io::ErrorKind, path::PathBuf, sync::Arc};

use super::Result;

use wgpu::{
    Adapter, AdapterInfo, Device, Instance, Limits, PipelineCache, Queue, Surface,
    SurfaceConfiguration, SurfaceTarget, TextureFormat,
};

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct RenderContext {
    pub instance: Instance,
    pub devices: Vec<DeviceHandle>,
    pub pipeline_cache_directory: Option<PathBuf>,
}

pub struct DeviceHandle {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub pipeline_cache: Option<Arc<PipelineCache>>,
    pub adapter_info: AdapterInfo,
    cache_filename: Option<PathBuf>,
}

impl RenderContext {
    pub fn new() -> Result<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            ..Default::default()
        });
        Ok(Self {
            instance,
            devices: Vec::new(),
            pipeline_cache_directory: None,
        })
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        width: u32,
        height: u32,
        present_mode: wgpu::PresentMode,
    ) -> Result<RenderSurface<'w>> {
        let surface = self.instance.create_surface(window.into())?;
        let dev_id = self
            .device(Some(&surface))
            .await
            .ok_or("Error creating device")?;

        let device_handle = &self.devices[dev_id];
        let capabilities = surface.get_capabilities(&device_handle.adapter);
        let format = capabilities
            .formats
            .into_iter()
            .find(|it| matches!(it, TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm))
            .expect("surface should support Rgba8Unorm or Bgra8Unorm");

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
        let mut maybe_features = wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE;

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
                },
                None,
            )
            .await
            .ok()?;
        let adapter_info = adapter.get_info();
        let (pipeline_cache, cache_filename) = if features.contains(wgpu::Features::PIPELINE_CACHE)
        {
            if let Some(cache_directory) = self.pipeline_cache_directory.as_ref() {
                let cache_key = wgpu::util::pipeline_cache_key(&adapter_info)
                    .expect("Adapter supports pipeline cache");
                let cache_file = cache_directory.join(cache_key);
                let contents = std::fs::read(&cache_file);
                match contents {
                    Ok(data) => {
                        let cache = unsafe {
                            device.create_pipeline_cache_init(&wgpu::PipelineCacheInitDescriptor {
                                label: Some("Vello Pipeline cache"),
                                data: &data,
                                fallback: true,
                            })
                        };
                        log::debug!("Making pipeline cache with {} bytes", data.len());
                        (Some(Arc::new(cache)), Some(cache_file))
                    }
                    Err(e) => {
                        if e.kind() != ErrorKind::NotFound {
                            log::error!("Got unexpected error {e} trying to open pipeline cache at {cache_file:?}");
                        } else {
                            log::info!("Didn't get pipeline cache at {cache_file:?}");
                        }
                        let cache = device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                            label: Some("Vello Pipeline cache"),
                        });
                        (Some(Arc::new(cache)), Some(cache_file))
                    }
                }
            } else {
                log::debug!("Not using pipeline cache as cache directory not provided");
                (None, None)
            }
        } else {
            log::debug!("Not using pipeline cache as device doesn't support it");
            (None, None)
        };
        let device_handle = DeviceHandle {
            adapter,
            device,
            queue,
            adapter_info,
            pipeline_cache,
            cache_filename,
        };
        self.devices.push(device_handle);
        Some(self.devices.len() - 1)
    }
}

impl DeviceHandle {
    pub fn store_pipeline_cache(&self) {
        if let Some(cache) = self.pipeline_cache.as_ref() {
            let Some(cache_filename) = self.cache_filename.as_ref() else {
                log::warn!(
                    "Unexpectedly didn't have pipeline cache filename, despite having cache"
                );
                return;
            };
            let Some(data) = cache.get_data() else {
                log::warn!("Unexpectedly got None from pipeline cache data");
                return;
            };
            let temp_filename = cache_filename.with_extension("temp");
            if let Err(e) = std::fs::write(&temp_filename, data) {
                log::error!("Got {e} whilst writing pipeline cache data to {temp_filename:?}");
                return;
            };
            if let Err(e) = std::fs::rename(&temp_filename, cache_filename) {
                log::error!("Got {e} whilst moving pipeline cache data from {temp_filename:?} to {cache_filename:?}");
                return;
            };
            log::info!("Stored pipeline cache at {cache_filename:?}");
        }
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
pub fn block_on_wgpu<F: Future>(device: &Device, mut fut: F) -> F::Output {
    if cfg!(target_arch = "wasm32") {
        panic!("Blocking can't work on WASM, so");
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
