// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::cell::RefCell;
use std::sync::Arc;

use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder};
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{ImageSource, PaintType};
use vello_common::peniko::{BlendMode, Fill, FontData};
use vello_common::pixmap::Pixmap;
use vello_common::recording::{Recordable, Recorder, Recording};
use vello_cpu::{Level, RenderContext, RenderMode, RenderSettings};
use vello_hybrid::Scene;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
use web_sys::WebGl2RenderingContext;

pub(crate) trait Renderer: Sized {
    type GlyphRenderer: GlyphRenderer;

    fn new(
        width: u16,
        height: u16,
        num_threads: u16,
        level: Level,
        render_mode: RenderMode,
    ) -> Self;
    fn fill_path(&mut self, path: &BezPath);
    fn stroke_path(&mut self, path: &BezPath);
    fn fill_rect(&mut self, rect: &Rect);
    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32);
    fn stroke_rect(&mut self, rect: &Rect);
    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self::GlyphRenderer>;
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    );
    fn flush(&mut self);
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_clip_path(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_opacity_layer(&mut self, opacity: f32);
    fn push_mask_layer(&mut self, mask: Mask);
    fn pop_layer(&mut self);
    fn pop_clip_path(&mut self);
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_mask(&mut self, mask: Option<Mask>);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    fn set_blend_mode(&mut self, blend_mode: BlendMode);
    fn render_to_pixmap(&self, pixmap: &mut Pixmap);
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource;
    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>));
    fn prepare_recording(&mut self, recording: &mut Recording);
    fn execute_recording(&mut self, recording: &Recording);
}

impl Renderer for RenderContext {
    type GlyphRenderer = Self;

    fn new(
        width: u16,
        height: u16,
        num_threads: u16,
        level: Level,
        render_mode: RenderMode,
    ) -> Self {
        let settings = RenderSettings {
            level,
            num_threads,
            render_mode,
        };

        Self::new_with(width, height, settings)
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

    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self> {
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

    fn flush(&mut self) {
        Self::flush(self);
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        Self::push_clip_layer(self, path);
    }

    fn push_clip_path(&mut self, path: &BezPath) {
        Self::push_clip_path(self, path);
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

    fn pop_clip_path(&mut self) {
        Self::pop_clip_path(self);
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        Self::set_stroke(self, stroke);
    }

    fn set_mask(&mut self, mask: Option<Mask>) {
        Self::set_mask(self, mask);
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

    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        Self::set_aliasing_threshold(self, aliasing_threshold);
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        Self::set_blend_mode(self, blend_mode);
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

    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource {
        ImageSource::Pixmap(pixmap)
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        Recordable::record(self, recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        Recordable::prepare_recording(self, recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        Recordable::execute_recording(self, recording);
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
pub(crate) struct HybridRenderer {
    scene: Scene,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    renderer: RefCell<vello_hybrid::Renderer>,
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
impl Renderer for HybridRenderer {
    type GlyphRenderer = Scene;

    fn new(width: u16, height: u16, num_threads: u16, level: Level, _: RenderMode) -> Self {
        if num_threads != 0 {
            panic!("hybrid renderer doesn't support multi-threading");
        }

        if !matches!(level, Level::Fallback(_)) {
            panic!("hybrid renderer doesn't support SIMD");
        }

        let scene = Scene::new(width, height);
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
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create renderer and render the scene to the texture
        let renderer = vello_hybrid::Renderer::new(
            &device,
            &vello_hybrid::RenderTargetConfig {
                format: texture.format(),
                width: width.into(),
                height: height.into(),
            },
        );

        Self {
            scene,
            device,
            queue,
            texture,
            texture_view,
            renderer: RefCell::new(renderer),
        }
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.scene.fill_path(path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.scene.stroke_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    fn fill_blurred_rounded_rect(&mut self, _: &Rect, _: f32, _: f32) {
        unimplemented!()
    }

    fn stroke_rect(&mut self, rect: &Rect) {
        self.scene.stroke_rect(rect);
    }

    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self::GlyphRenderer> {
        self.scene.glyph_run(font)
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.scene.push_layer(clip, blend_mode, opacity, mask);
    }

    fn flush(&mut self) {}

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.scene.push_clip_layer(path);
    }

    fn push_clip_path(&mut self, path: &BezPath) {
        self.scene.push_clip_path(path);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.scene.push_layer(None, Some(blend_mode), None, None);
    }

    fn push_opacity_layer(&mut self, opacity: f32) {
        self.scene.push_layer(None, None, Some(opacity), None);
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.scene.pop_clip_path();
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        self.scene.set_stroke(stroke);
    }

    fn set_mask(&mut self, _: Option<Mask>) {
        unimplemented!()
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        let paint_type: PaintType = paint.into();
        match paint_type {
            PaintType::Solid(s) => self.scene.set_paint(s),
            PaintType::Gradient(g) => self.scene.set_paint(g),
            PaintType::Image(i) => self.scene.set_paint(i),
        }
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        self.scene.set_paint_transform(affine);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.scene.set_fill_rule(fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        self.scene.set_transform(transform);
    }

    fn set_blend_mode(&mut self, _: BlendMode) {
        unimplemented!()
    }

    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.scene.set_aliasing_threshold(aliasing_threshold);
    }

    // This method creates device resources every time it is called. This does not matter much for
    // testing, but should not be used as a basis for implementing something real. This would be a
    // very bad example for that.
    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
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

        let width = self.scene.width();
        let height = self.scene.height();

        // for image in image_cache.images {}

        let render_size = vello_hybrid::RenderSize {
            width: width.into(),
            height: height.into(),
        };
        // Copy texture to buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vello Render To Buffer"),
            });
        self.renderer
            .borrow_mut()
            .render(
                &self.scene,
                &self.device,
                &self.queue,
                &mut encoder,
                &render_size,
                &self.texture_view,
            )
            .unwrap();

        // Create a buffer to copy the texture data
        let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
        let texture_copy_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
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
        self.queue.submit([encoder.finish()]);

        // Map the buffer for reading
        texture_copy_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    panic!("Failed to map texture for reading");
                }
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

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

    fn width(&self) -> u16 {
        self.scene.width()
    }

    fn height(&self) -> u16 {
        self.scene.height()
    }

    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Upload Test Image"),
            });

        // Upload image to cache and atlas in one step!
        let image_id = self.renderer.borrow_mut().upload_image(
            &self.device,
            &self.queue,
            &mut encoder,
            &pixmap,
        );

        self.queue.submit([encoder.finish()]);

        ImageSource::OpaqueId(image_id)
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        Recordable::record(&mut self.scene, recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        Recordable::prepare_recording(&mut self.scene, recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        Recordable::execute_recording(&mut self.scene, recording);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub(crate) struct HybridRenderer {
    scene: Scene,
    renderer: RefCell<vello_hybrid::WebGlRenderer>,
    gl: WebGl2RenderingContext,
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
impl Renderer for HybridRenderer {
    type GlyphRenderer = Scene;

    fn new(width: u16, height: u16, num_threads: u16, level: Level, _: RenderMode) -> Self {
        use wasm_bindgen::JsCast;
        use web_sys::HtmlCanvasElement;

        if num_threads != 0 {
            panic!("hybrid renderer doesn't support multi-threading");
        }

        if !matches!(level, Level::Fallback(_)) {
            panic!("hybrid renderer doesn't support SIMD");
        }

        let scene = Scene::new(width, height);

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

        let renderer = vello_hybrid::WebGlRenderer::new(&canvas);

        let gl = canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<WebGl2RenderingContext>()
            .unwrap();

        Self {
            scene,
            renderer: RefCell::new(renderer),
            gl,
        }
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.scene.fill_path(path);
    }

    fn set_blend_mode(&mut self, _: BlendMode) {
        unimplemented!()
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.scene.stroke_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    fn fill_blurred_rounded_rect(&mut self, _: &Rect, _: f32, _: f32) {
        unimplemented!()
    }

    fn stroke_rect(&mut self, rect: &Rect) {
        self.scene.stroke_rect(rect);
    }

    fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self::GlyphRenderer> {
        self.scene.glyph_run(font)
    }

    fn push_clip_path(&mut self, path: &BezPath) {
        self.scene.push_clip_path(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
    ) {
        self.scene.push_layer(clip, blend_mode, opacity, mask);
    }

    fn flush(&mut self) {}

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.scene.push_clip_layer(path);
    }

    fn push_blend_layer(&mut self, mode: BlendMode) {
        self.scene.push_layer(None, Some(mode), None, None);
    }

    fn push_opacity_layer(&mut self, opacity: f32) {
        self.scene.push_layer(None, None, Some(opacity), None);
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.scene.pop_clip_path();
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        self.scene.set_stroke(stroke);
    }

    fn set_mask(&mut self, _: Option<Mask>) {
        unimplemented!()
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        let paint_type: PaintType = paint.into();
        match paint_type {
            PaintType::Solid(s) => self.scene.set_paint(s),
            PaintType::Gradient(g) => self.scene.set_paint(g),
            PaintType::Image(i) => self.scene.set_paint(i),
        }
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        self.scene.set_paint_transform(affine);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.scene.set_fill_rule(fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        self.scene.set_transform(transform);
    }

    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.scene.set_aliasing_threshold(aliasing_threshold);
    }

    // vello_hybrid WebGL renderer backend.
    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        use web_sys::WebGl2RenderingContext;

        let width = self.scene.width();
        let height = self.scene.height();

        let render_size = vello_hybrid::RenderSize {
            width: width.into(),
            height: height.into(),
        };
        self.renderer
            .borrow_mut()
            .render(&self.scene, &render_size)
            .unwrap();

        let mut pixels = vec![0_u8; (width as usize) * (height as usize) * 4];
        self.gl
            .read_pixels_with_opt_u8_array(
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
        self.scene.width()
    }

    fn height(&self) -> u16 {
        self.scene.height()
    }

    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource {
        let image_id = self.renderer.borrow_mut().upload_image(&pixmap);
        ImageSource::OpaqueId(image_id)
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        self.scene.record(recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        self.scene.prepare_recording(recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        self.scene.execute_recording(recording);
    }
}
