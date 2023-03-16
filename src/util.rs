// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Simple helpers for managing wgpu state and surfaces.

use std::future::Future;

use super::Result;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    Adapter, Device, Instance, Limits, Queue, Surface, SurfaceConfiguration, TextureFormat,
};

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
    pub fn new() -> Result<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        Ok(Self {
            instance,
            devices: Vec::new(),
        })
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<W>(&mut self, window: &W, width: u32, height: u32) -> RenderSurface
    where
        W: HasRawWindowHandle + HasRawDisplayHandle,
    {
        let surface = unsafe { self.instance.create_surface(window) }.unwrap();
        let dev_id = self.device(Some(&surface)).await.unwrap();

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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&self.devices[dev_id].device, &config);
        RenderSurface {
            surface,
            config,
            dev_id,
            format,
        }
    }

    /// Resizes the surface to the new dimensions.
    pub fn resize_surface(&self, surface: &mut RenderSurface, width: u32, height: u32) {
        surface.config.width = width;
        surface.config.height = height;
        surface
            .surface
            .configure(&self.devices[surface.dev_id].device, &surface.config);
    }

    pub fn set_present_mode(&self, surface: &mut RenderSurface, present_mode: wgpu::PresentMode) {
        surface.config.present_mode = present_mode;
        surface
            .surface
            .configure(&self.devices[surface.dev_id].device, &surface.config);
    }

    /// Finds or creates a compatible device handle id.
    pub async fn device(&mut self, compatible_surface: Option<&Surface>) -> Option<usize> {
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
    async fn new_device(&mut self, compatible_surface: Option<&Surface>) -> Option<usize> {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(
            &self.instance,
            wgpu::Backends::PRIMARY,
            compatible_surface,
        )
        .await?;
        let features = adapter.features();
        let limits = Limits::default();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features
                        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_TEXTURE),
                    limits,
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

/// Combination of surface and its configuration.
#[derive(Debug)]
pub struct RenderSurface {
    pub surface: Surface,
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
    let waker = std::task::Waker::from(std::sync::Arc::new(NullWake));
    let mut context = std::task::Context::from_waker(&waker);
    // Same logic as `pin_mut!` macro from `pin_utils`.
    let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
    loop {
        match fut.as_mut().poll(&mut context) {
            std::task::Poll::Pending => {
                device.poll(wgpu::Maintain::Wait);
            }
            std::task::Poll::Ready(item) => break item,
        }
    }
}
