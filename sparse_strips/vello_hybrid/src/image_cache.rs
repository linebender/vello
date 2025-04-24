extern crate std;
extern crate wgpu;

use std::collections::HashMap;
use wgpu::{BindGroup, BindGroupLayout, Device, Queue, Texture, TextureView};

/// Represents an image resource for rendering
#[derive(Debug)]
pub struct ImageResource {
    /// The texture containing the image data
    pub texture: Texture,
    /// The texture view for binding
    pub view: TextureView,
    /// The bind group for this image
    pub bind_group: Option<BindGroup>,
}

/// Manages image resources for the renderer
#[derive(Debug)]
pub struct ImageCache {
    /// Map of image IDs to resources
    pub images: HashMap<u32, ImageResource>,
    /// Bind group layout for image textures
    pub bind_group_layout: Option<BindGroupLayout>,
}

impl ImageCache {
    /// Create a new image cache
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            bind_group_layout: None,
        }
    }
    pub fn create_bind_group(&mut self, device: &Device) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Image Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        self.bind_group_layout = Some(bind_group_layout);
    }

    /// Register an image with the cache
    pub fn upload_image(
        &mut self,
        device: &Device,
        queue: &Queue,
        id: u32,
        width: u32,
        height: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Assuming RGBA format (4 bytes per pixel)
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.images.insert(
            id,
            ImageResource {
                texture,
                view,
                bind_group: None,
            },
        );
    }

    /// Remove an image from the cache
    pub fn remove_image(&mut self, id: u32) {
        self.images.remove(&id);
    }

    /// Get a reference to an image resource
    pub fn get_image(&self, id: u32) -> Option<&ImageResource> {
        self.images.get(&id)
    }

    /// Create bind groups for all images
    pub fn create_bind_groups(&mut self, device: &Device) {
        if let Some(bind_group_layout) = &self.bind_group_layout {
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            for resource in self.images.values_mut() {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Image Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&resource.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });
                resource.bind_group = Some(bind_group);
            }
        }
    }
}
