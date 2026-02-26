// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::cell::RefCell;
use std::sync::Arc;

use parley_draw::atlas::ImageCache;
use parley_draw::{CpuGlyphCaches, Glyph, GlyphRunBuilder};
use vello_common::filter_effects::Filter;
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{ImageId, ImageSource, PaintType, Tint};
use vello_common::peniko::{BlendMode, Fill, FontData};
use vello_common::pixmap::Pixmap;
use vello_common::recording::{Recordable, Recorder, Recording};
use vello_cpu::{Level, RenderContext, RenderMode, RenderSettings};
use vello_hybrid::Scene;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
use web_sys::WebGl2RenderingContext;

pub(crate) struct GlyphConfig {
    pub font_size: f32,
    pub hint: bool,
    pub glyph_transform: Option<Affine>,
}

pub(crate) struct TestGlyphRun<'a, T: Renderer + ?Sized> {
    renderer: &'a mut T,
    font: FontData,
    config: GlyphConfig,
}

impl<'a, T: Renderer> TestGlyphRun<'a, T> {
    pub(crate) fn font_size(mut self, size: f32) -> Self {
        self.config.font_size = size;
        self
    }

    pub(crate) fn hint(mut self, hint: bool) -> Self {
        self.config.hint = hint;
        self
    }

    pub(crate) fn glyph_transform(mut self, transform: Affine) -> Self {
        self.config.glyph_transform = Some(transform);
        self
    }

    pub(crate) fn fill_glyphs(self, glyphs: impl Iterator<Item = Glyph>) {
        let glyphs: Vec<Glyph> = glyphs.collect();
        self.renderer
            .fill_glyphs_impl(&self.font, &self.config, &glyphs);
    }

    pub(crate) fn stroke_glyphs(self, glyphs: impl Iterator<Item = Glyph>) {
        let glyphs: Vec<Glyph> = glyphs.collect();
        self.renderer
            .stroke_glyphs_impl(&self.font, &self.config, &glyphs);
    }
}

pub(crate) trait Renderer: Sized {
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
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    );
    fn flush(&mut self);
    fn push_clip_layer(&mut self, path: &BezPath);
    fn push_clip_path(&mut self, path: &BezPath);
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    fn push_opacity_layer(&mut self, opacity: f32);
    fn push_mask_layer(&mut self, mask: Mask);
    fn push_filter_layer(&mut self, filter: Filter);
    fn pop_layer(&mut self);
    fn pop_clip_path(&mut self);
    fn set_stroke(&mut self, stroke: Stroke);
    fn set_mask(&mut self, mask: Mask);
    fn set_paint(&mut self, paint: impl Into<PaintType>);
    fn set_tint(&mut self, tint: Option<Tint>);
    fn set_paint_transform(&mut self, affine: Affine);
    fn set_fill_rule(&mut self, fill_rule: Fill);
    fn set_transform(&mut self, transform: Affine);
    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>);
    fn set_blend_mode(&mut self, blend_mode: BlendMode);
    fn set_filter_effect(&mut self, filter: Filter);
    fn reset_filter_effect(&mut self);
    fn render_to_pixmap(&self, pixmap: &mut Pixmap);
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource;
    fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId;
    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>));
    fn prepare_recording(&mut self, recording: &mut Recording);
    fn execute_recording(&mut self, recording: &Recording);

    fn fill_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]);
    fn stroke_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]);

    fn glyph_run<'a>(&'a mut self, font: &FontData) -> TestGlyphRun<'a, Self> {
        TestGlyphRun {
            renderer: self,
            font: font.clone(),
            config: GlyphConfig {
                font_size: 16.0,
                hint: true,
                glyph_transform: None,
            },
        }
    }
}

fn default_image_cache() -> ImageCache {
    ImageCache::new_with_config(AtlasConfig {
        initial_atlas_count: 1,
        max_atlases: 4,
        atlas_size: (512, 512),
        auto_grow: true,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// CPU renderer
// ---------------------------------------------------------------------------

pub(crate) struct CpuRenderer {
    ctx: RenderContext,
    glyph_caches: CpuGlyphCaches,
    image_cache: ImageCache,
}

impl Renderer for CpuRenderer {
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
        Self {
            ctx: RenderContext::new_with(width, height, settings),
            glyph_caches: CpuGlyphCaches::new(512, 512),
            image_cache: default_image_cache(),
        }
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.ctx.fill_path(path);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.ctx.stroke_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.ctx.fill_rect(rect);
    }

    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32) {
        self.ctx.fill_blurred_rounded_rect(rect, radius, std_dev);
    }

    fn stroke_rect(&mut self, rect: &Rect) {
        self.ctx.stroke_rect(rect);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.ctx
            .push_layer(clip_path, blend_mode, opacity, mask, filter);
    }

    fn flush(&mut self) {
        self.ctx.flush();
    }

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.ctx.push_clip_layer(path);
    }

    fn push_clip_path(&mut self, path: &BezPath) {
        self.ctx.push_clip_path(path);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.ctx.push_blend_layer(blend_mode);
    }

    fn push_opacity_layer(&mut self, opacity: f32) {
        self.ctx.push_opacity_layer(opacity);
    }

    fn push_mask_layer(&mut self, mask: Mask) {
        self.ctx.push_mask_layer(mask);
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.ctx.push_filter_layer(filter);
    }

    fn pop_layer(&mut self) {
        self.ctx.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.ctx.pop_clip_path();
    }

    fn set_stroke(&mut self, stroke: Stroke) {
        self.ctx.set_stroke(stroke);
    }

    fn set_mask(&mut self, mask: Mask) {
        self.ctx.set_mask(mask);
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.ctx.set_paint(paint);
    }

    fn set_tint(&mut self, tint: Option<Tint>) {
        self.ctx.set_tint(tint);
    }

    fn set_paint_transform(&mut self, affine: Affine) {
        self.ctx.set_paint_transform(affine);
    }

    fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.ctx.set_fill_rule(fill_rule);
    }

    fn set_transform(&mut self, transform: Affine) {
        self.ctx.set_transform(transform);
    }

    fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.ctx.set_aliasing_threshold(aliasing_threshold);
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.ctx.set_blend_mode(blend_mode);
    }

    fn set_filter_effect(&mut self, filter: Filter) {
        self.ctx.set_filter_effect(filter);
    }

    fn reset_filter_effect(&mut self) {
        self.ctx.reset_filter_effect();
    }

    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        self.ctx.render_to_pixmap(pixmap);
    }

    fn width(&self) -> u16 {
        self.ctx.width()
    }

    fn height(&self) -> u16 {
        self.ctx.height()
    }

    fn get_image_source(&mut self, pixmap: Arc<Pixmap>) -> ImageSource {
        ImageSource::Pixmap(pixmap)
    }

    fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        self.ctx.register_image(pixmap)
    }

    fn record(&mut self, recording: &mut Recording, f: impl FnOnce(&mut Recorder<'_>)) {
        Recordable::record(&mut self.ctx, recording, f);
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        Recordable::prepare_recording(&mut self.ctx, recording);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        Recordable::execute_recording(&mut self.ctx, recording);
    }

    fn fill_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.ctx.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.ctx)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.fill_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.image_cache,
        );
    }

    fn stroke_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.ctx.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.ctx)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.stroke_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.image_cache,
        );
    }
}

// ---------------------------------------------------------------------------
// Hybrid renderer (native / non-WebGL)
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
pub(crate) struct HybridRenderer {
    scene: Scene,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    renderer: RefCell<vello_hybrid::Renderer>,
    glyph_caches: parley_draw::GpuGlyphCaches,
}

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
impl Renderer for HybridRenderer {
    fn new(width: u16, height: u16, num_threads: u16, level: Level, _: RenderMode) -> Self {
        if num_threads != 0 {
            panic!("hybrid renderer doesn't support multi-threading");
        }

        if !level.is_fallback() {
            panic!("hybrid renderer doesn't support SIMD");
        }

        let scene = Scene::new(width, height);
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
            glyph_caches: parley_draw::GpuGlyphCaches::with_config(
                parley_draw::GlyphCacheConfig::default(),
            ),
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

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.scene
            .push_layer(clip, blend_mode, opacity, mask, filter);
    }

    fn flush(&mut self) {}

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.scene.push_clip_layer(path);
    }

    fn push_clip_path(&mut self, path: &BezPath) {
        self.scene.push_clip_path(path);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.scene
            .push_layer(None, Some(blend_mode), None, None, None);
    }

    fn push_opacity_layer(&mut self, opacity: f32) {
        self.scene.push_layer(None, None, Some(opacity), None, None);
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.scene.push_filter_layer(filter);
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

    fn set_mask(&mut self, _: Mask) {
        unimplemented!()
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.scene.set_paint(paint);
    }

    fn set_tint(&mut self, tint: Option<Tint>) {
        self.scene.set_tint(tint);
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

    fn set_filter_effect(&mut self, filter: Filter) {
        self.scene.set_filter_effect(filter);
    }

    fn reset_filter_effect(&mut self) {
        self.scene.reset_filter_effect();
    }

    fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        let _guard = {
            use std::sync::Mutex;
            static M: Mutex<()> = Mutex::new(());
            M.lock().unwrap()
        };

        let width = self.scene.width();
        let height = self.scene.height();

        let render_size = vello_hybrid::RenderSize {
            width: width.into(),
            height: height.into(),
        };
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

        let image_id = self.renderer.borrow_mut().upload_image(
            &self.device,
            &self.queue,
            &mut encoder,
            &pixmap,
        );

        self.queue.submit([encoder.finish()]);

        ImageSource::opaque_id_with_opacity_hint(image_id, pixmap.may_have_opacities())
    }

    fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Register Test Image"),
            });

        let image_id = self.renderer.borrow_mut().upload_image(
            &self.device,
            &self.queue,
            &mut encoder,
            &pixmap,
        );

        self.queue.submit([encoder.finish()]);

        image_id
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

    fn fill_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.scene.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.scene)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.fill_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.renderer.borrow_mut().image_cache,
        );
    }

    fn stroke_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.scene.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.scene)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.stroke_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.renderer.borrow_mut().image_cache,
        );
    }
}

// ---------------------------------------------------------------------------
// Hybrid renderer (WebGL / wasm32)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub(crate) struct HybridRenderer {
    scene: Scene,
    renderer: RefCell<vello_hybrid::WebGlRenderer>,
    gl: WebGl2RenderingContext,
    glyph_caches: parley_draw::GpuGlyphCaches,
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
impl Renderer for HybridRenderer {
    fn new(width: u16, height: u16, num_threads: u16, level: Level, _: RenderMode) -> Self {
        use wasm_bindgen::JsCast;
        use web_sys::HtmlCanvasElement;

        if num_threads != 0 {
            panic!("hybrid renderer doesn't support multi-threading");
        }

        if !level.is_fallback() {
            panic!("hybrid renderer doesn't support SIMD");
        }

        let scene = Scene::new(width, height);

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
            glyph_caches: parley_draw::GpuGlyphCaches::with_config(
                parley_draw::GlyphCacheConfig::default(),
            ),
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

    fn push_clip_path(&mut self, path: &BezPath) {
        self.scene.push_clip_path(path);
    }

    fn push_layer(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.scene
            .push_layer(clip, blend_mode, opacity, mask, filter);
    }

    fn flush(&mut self) {}

    fn push_clip_layer(&mut self, path: &BezPath) {
        self.scene.push_clip_layer(path);
    }

    fn push_blend_layer(&mut self, mode: BlendMode) {
        self.scene.push_layer(None, Some(mode), None, None, None);
    }

    fn push_opacity_layer(&mut self, opacity: f32) {
        self.scene.push_layer(None, None, Some(opacity), None, None);
    }

    fn push_mask_layer(&mut self, _: Mask) {
        unimplemented!()
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.scene.push_filter_layer(filter);
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

    fn set_mask(&mut self, _: Mask) {
        unimplemented!()
    }

    fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.scene.set_paint(paint);
    }

    fn set_tint(&mut self, tint: Option<Tint>) {
        self.scene.set_tint(tint);
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

    fn set_filter_effect(&mut self, filter: Filter) {
        self.scene.set_filter_effect(filter);
    }

    fn reset_filter_effect(&mut self) {
        self.scene.reset_filter_effect();
    }

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
        ImageSource::opaque_id_with_opacity_hint(image_id, pixmap.may_have_opacities())
    }

    fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        self.renderer.borrow_mut().upload_image(&pixmap)
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

    fn fill_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.scene.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.scene)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.fill_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.renderer.borrow_mut().image_cache,
        );
    }

    fn stroke_glyphs_impl(&mut self, font: &FontData, config: &GlyphConfig, glyphs: &[Glyph]) {
        let transform = *self.scene.transform();
        let mut builder = GlyphRunBuilder::new(font.clone(), transform, &mut self.scene)
            .font_size(config.font_size)
            .hint(config.hint);
        if let Some(gt) = config.glyph_transform {
            builder = builder.glyph_transform(gt);
        }
        builder.stroke_glyphs(
            glyphs.iter().copied(),
            &mut self.glyph_caches,
            &mut self.renderer.borrow_mut().image_cache,
        );
    }
}
