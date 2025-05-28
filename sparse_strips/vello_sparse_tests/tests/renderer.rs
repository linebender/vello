// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder};
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::PaintType;
use vello_common::peniko::{BlendMode, Fill, Font};
use vello_common::pixmap::Pixmap;
use vello_cpu::{RenderContext, RenderMode};
use vello_hybrid::Scene;

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
        opacity: Option<f32>,
        mask: Option<Mask>,
    );
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_opacity_layer(&mut self, opacity: f32);
    fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    fn render_to_pixmap(&self, pixmap: &mut Pixmap, render_mode: RenderMode);
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
        opacity: Option<f32>,
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

    fn push_opacity_layer(&mut self, opacity: f32) {
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

    fn render_to_pixmap(&self, pixmap: &mut Pixmap, render_mode: RenderMode) {
        Self::render_to_pixmap(self, pixmap, render_mode);
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
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        Self::push_layer(self, clip, blend_mode, opacity, mask);
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        Self::push_clip_layer(self, path);
    }

    fn push_blend_layer(&mut self, _: BlendMode) {
        unimplemented!()
    }

    fn push_opacity_layer(&mut self, _: f32) {
        unimplemented!()
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
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

    // This method creates device resources every time it is called. This does not matter much for
    // testing, but should not be used as a basis for implementing something real. This would be a
    // very bad example for that.
    #[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
    fn render_to_pixmap(&self, pixmap: &mut Pixmap, _: RenderMode) {
        // On some platforms using `cargo test` triggers segmentation faults in wgpu when the GPU
        // tests are run in parallel (likely related to the number of device resources being
        // requested simultaneously). This is "fixed" by putting a mutex around this method,
        // ensuring only one set of device resources is alive at the same time. This slows down
        // testing when `cargo test` is used.
        //
        // Testing with `cargo nextest` (as on CI) is not meaningfully slowed down. `nextest` runs
        // each test in its own process (<https://nexte.st/docs/design/why-process-per-test/>),
        // meaning there is no contention on this mutex.
        let _guard = {
            use std::sync::Mutex;
            static M: Mutex<()> = Mutex::new(());
            M.lock().unwrap()
        };

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
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
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
        // Copy texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Vello Render To Buffer"),
        });
        renderer
            .render(
                self,
                &device,
                &queue,
                &mut encoder,
                &render_size,
                &texture_view,
            )
            .unwrap();

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
        device.poll(wgpu::Maintain::Wait);

        // Read back the pixel data
        for (row, buf) in texture_copy_buffer
            .slice(..)
            .get_mapped_range()
            .chunks_exact(bytes_per_row as usize)
            .zip(
                pixmap
                    .data_as_u8_slice_mut()
                    .chunks_exact_mut(width as usize * 4),
            )
        {
            buf.copy_from_slice(&row[0..width as usize * 4]);
        }
        texture_copy_buffer.unmap();
    }

    // vello_hybrid WebGL renderer backend.
    #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
    fn render_to_pixmap(&self, pixmap: &mut Pixmap, _: RenderMode) {
        use wasm_bindgen::JsCast;
        use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

        let width = self.width();
        let height = self.height();

        // Create an offscreen HTMLCanvasElement, render the test image to it, and finally read off
        // the pixmap for diff checking.
        let document = web_sys::window().unwrap().document().unwrap();

        let canvas = document
            .create_element("canvas")
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()
            .unwrap();

        canvas.set_width(width.into());
        canvas.set_height(height.into());

        let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);
        let render_size = vello_hybrid::RenderSize {
            width: width.into(),
            height: height.into(),
        };

        renderer.render(self, &render_size).unwrap();

        let gl = canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<WebGl2RenderingContext>()
            .unwrap();
        let mut pixels = vec![0_u8; (width as usize) * (height as usize) * 4];
        gl.read_pixels_with_opt_u8_array(
            0,
            0,
            width.into(),
            height.into(),
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(&mut pixels),
        )
        .unwrap();

        let pixmap_data = pixmap.data_as_u8_slice_mut();
        pixmap_data.copy_from_slice(&pixels);
    }

    fn width(&self) -> u16 {
        Self::width(self)
    }

    fn height(&self) -> u16 {
        Self::height(self)
    }
}
