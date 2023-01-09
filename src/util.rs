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

use super::Result;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    Adapter, Device, Instance, Limits, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration,
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
        let instance = Instance::new(wgpu::Backends::PRIMARY);
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
        let surface = unsafe { self.instance.create_surface(window) };
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        let dev_id = self.device(Some(&surface)).await.unwrap();
        surface.configure(&self.devices[dev_id].device, &config);
        RenderSurface {
            surface,
            config,
            dev_id,
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

    /// Finds or creates a compatible device handle id.
    async fn device(&mut self, compatible_surface: Option<&Surface>) -> Option<usize> {
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
        return compatible;
    }

    /// Creates a compatible device handle id.
    async fn new_device(&mut self, compatible_surface: Option<&Surface>) -> Option<usize> {
        let adapter = self
            .instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface,
                ..Default::default()
            })
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
pub struct RenderSurface {
    pub surface: Surface,
    pub config: SurfaceConfiguration,
    pub dev_id: usize,
}
