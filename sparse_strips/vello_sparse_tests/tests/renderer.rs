// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_api::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_api::mask::Mask;
use vello_api::paint::PaintType;
use vello_api::peniko::{BlendMode, Fill, Font};
use vello_api::pixmap::Pixmap;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder};
use vello_cpu::RenderContext;
use vello_hybrid::Scene;
use wgpu::RenderPassDescriptor;

pub(crate) trait Renderer: Sized + GlyphRenderer {
    fn new(width: u16, height: u16) -> Self;
    fn fill_path(&mut self, path: &BezPath);
    fn stroke_path(&mut self, path: &BezPath);
    fn fill_rect(&mut self, rect: &Rect);
    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32);
    fn stroke_rect(&mut self, rect: &Rect);
    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self>;
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<u8>,
        mask: Option<Mask>,
    );
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_opacity_layer(&mut self, opacity: u8);
    fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    fn render_to_pixmap(&self, pixmap: &mut Pixmap);
    fn width(&self) -> u16;
    fn height(&self) -> u16;
}

impl Renderer for RenderContext {
    fn new(width: u16, height: u16) -> Self {
        Self::new(width, height)
    }

    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        Self::stroke_path(self, path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        Self::fill_blurred_rounded_rect(self, rect, radius, std_dev);
    }

    fn stroke_rect(&mut self, rect: &Rect) {
        Self::stroke_rect(self, rect);
    }

    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        Self::glyph_run(self, font)
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<u8>,
        mask: Option<Mask>,
    ) {
        Self::push_layer(self, clip_path, blend_mode, opacity, mask);
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        Self::push_clip_layer(self, path);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        Self::push_blend_layer(self, blend_mode);
    }

    fn push_opacity_layer(&mut self, opacity: u8) {
        Self::push_opacity_layer(self, opacity);
    }

    fn push_mask_layer(&mut self, mask: Mask) {
        Self::push_mask_layer(self, mask);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        Self::set_stroke(self, stroke);
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        Self::set_paint(self, paint);
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        Self::set_paint_transform(self, affine);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        Self::set_fill_rule(self, fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }

    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        Self::render_to_pixmap(self, pixmap);
    }

    fn width(&self) -> u16 {
        Self::width(self)
    }

    fn height(&self) -> u16 {
        Self::height(self)
    }
}

impl Renderer for Scene {
    fn new(width: u16, height: u16) -> Self {
        Self::new(width, height)
    }

    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        Self::stroke_path(self, path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    fn fill_blurred_rounded_rect(&mut self, _: &Rect, _: f32, _: f32) {
        unimplemented!()
    }

    fn stroke_rect(&mut self, rect: &Rect) {
        Self::stroke_rect(self, rect);
    }

    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        Self::glyph_run(self, font)
    }

    fn push_layer(
        &mut self,
        _: Option<&BezPath>,
        _: Option<BlendMode>,
        _: Option<u8>,
        _: Option<Mask>,
    ) {
        unimplemented!()
    }

    fn push_clip_layer(&mut self, _: &BezPath) {
        unimplemented!()
    }

    fn push_blend_layer(&mut self, _: BlendMode) {
        unimplemented!()
    }

    fn push_opacity_layer(&mut self, _: u8) {
        unimplemented!()
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn pop_layer(&mut self) {
        unimplemented!()
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        Self::set_stroke(self, stroke);
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        let paint_type: PaintType = paint.into();
        match paint_type {
            PaintType::Solid(s) => Self::set_paint(self, s.into()),
            PaintType::Gradient(_) => {}
            PaintType::Image(_) => {}
        }
    }

    fn set_paint_transform(&mut self, _: Affine) {
        unimplemented!();
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        Self::set_fill_rule(self, fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }

    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        let width = self.width();
        let height = self.height();

        // Copied from vello_hybrid/examples/`render_to_file.rs`.

        // Initialize wgpu device and queue for GPU rendering
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to find an appropriate adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        // Create a render target texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width: width.into(),
                height: height.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create renderer and render the scene to the texture
        let mut renderer = vello_hybrid::Renderer::new(
            &device,
            &vello_hybrid::RenderTargetConfig {
                format: texture.format(),
                width: width.into(),
                height: height.into(),
            },
        );
        let render_size = vello_hybrid::RenderSize {
            width: width.into(),
            height: height.into(),
        };
        renderer.prepare(&device, &queue, self, &render_size);
        // Copy texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Vello Render To Buffer"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            renderer.render(self, &mut pass);
        }

        // Create a buffer to copy the texture data
        let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
        let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &texture_copy_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: width.into(),
                height: height.into(),
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        // Map the buffer for reading
        texture_copy_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    panic!("Failed to map texture for reading");
                }
            });
        device.poll(wgpu::PollType::Wait).unwrap();

        // Read back the pixel data
        let mut img_data = Vec::with_capacity(usize::from(width) * usize::from(height) * 4);
        for row in texture_copy_buffer
            .slice(..)
            .get_mapped_range()
            .chunks_exact(bytes_per_row as usize)
        {
            img_data.extend_from_slice(&row[0..width as usize * 4]);
        }
        texture_copy_buffer.unmap();

        pixmap.buf = img_data;
    }

    fn width(&self) -> u16 {
        Self::width(self)
    }

    fn height(&self) -> u16 {
        Self::height(self)
    }
}
