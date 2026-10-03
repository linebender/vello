// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the lifecycle of image atlas regions in the wgpu renderer.

use std::sync::MutexGuard;

use glifo::GlyphCacheConfig;
use vello_common::color::palette::css::{BLUE, GREEN, RED};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Rect};
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::paint::{Image, ImageId, ImageSource};
use vello_common::peniko::ImageSampler;
use vello_common::pixmap::Pixmap;
use vello_gpu::{
    ClearSettings, MemorySettings, RenderSettings, RenderSize, RenderTargetConfig, Renderer,
    Resources, Scene, TargetInit, TextureBindings,
};
use vello_tests::renderer::{lock_wgpu_tests, read_rgba8_texture, wgpu_device_queue};

use crate::util::layout_glyphs_roboto;

const SIZE: u16 = 16;
const TRANSPARENT: [u8; 4] = [0; 4];

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    resources: Resources,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    _guard: MutexGuard<'static, ()>,
}

impl Ctx {
    fn new(atlas_size: u16) -> Self {
        let _guard = lock_wgpu_tests();
        let (device, queue) = wgpu_device_queue();
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Test Target"),
            size: wgpu::Extent3d {
                width: SIZE.into(),
                height: SIZE.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let settings = RenderSettings {
            memory_settings: MemorySettings {
                image_atlas_config: AtlasConfig {
                    atlas_size: (atlas_size, atlas_size),
                    ..AtlasConfig::default()
                },
                ..MemorySettings::default()
            },
            ..RenderSettings::default()
        };
        let (renderer, resources) = Renderer::new_with(
            &device,
            &RenderTargetConfig {
                format: target.format(),
                width: SIZE,
                height: SIZE,
            },
            settings,
        );
        Self {
            device,
            queue,
            renderer,
            resources,
            target,
            view,
            _guard,
        }
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    fn upload(&mut self, width: u16, height: u16, color: AlphaColor<Srgb>) -> ImageId {
        let mut pixmap = Pixmap::new(width, height);
        pixmap.data_mut().fill(color.premultiply().to_rgba8());
        let mut encoder = self.encoder();
        let id = self.renderer.upload_image(
            &mut self.resources,
            &self.device,
            &self.queue,
            &mut encoder,
            &pixmap,
        );
        self.queue.submit([encoder.finish()]);
        id
    }

    fn slot(&self, id: ImageId) -> (AtlasId, [u16; 2]) {
        let image = self.resources.image_cache().get(id).unwrap();
        (image.atlas_id, image.offset)
    }

    fn encode_render(&mut self, scene: &Scene, encoder: &mut wgpu::CommandEncoder) {
        self.renderer
            .render(
                scene,
                &mut self.resources,
                &self.device,
                &self.queue,
                encoder,
                &RenderSize {
                    width: SIZE,
                    height: SIZE,
                },
                &self.view,
                None,
                &TextureBindings::new(),
                TargetInit::Clear(ClearSettings::Viewport {
                    color: AlphaColor::TRANSPARENT,
                }),
            )
            .unwrap();
    }

    fn render(&mut self, scene: &Scene) {
        let mut encoder = self.encoder();
        self.encode_render(scene, &mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    /// Reads back an `Rgba8Unorm` texture, row by row.
    fn read(&self, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
        let pixmap = read_rgba8_texture(&self.device, &self.queue, texture);
        pixmap
            .data()
            .iter()
            .map(|pixel| pixel.to_u8_array())
            .collect()
    }

    fn target_center(&self) -> [u8; 4] {
        let half = usize::from(SIZE / 2);
        self.read(&self.target)[half * usize::from(SIZE) + half]
    }
}

fn premul(color: AlphaColor<Srgb>) -> [u8; 4] {
    color.premultiply().to_rgba8().to_u8_array()
}

fn image_scene(id: ImageId) -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(Image {
        image: ImageSource::opaque_id(id),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));
    scene
}

/// A render encoded before `destroy_image` still draws the image when submitted afterwards.
#[test]
fn gpu_atlas_destroyed_image_survives_encoded_render() {
    let mut ctx = Ctx::new(64);
    let red = ctx.upload(4, 4, RED);

    let mut encoder = ctx.encoder();
    ctx.encode_render(&image_scene(red), &mut encoder);
    ctx.resources.destroy_image(red);
    ctx.queue.submit([encoder.finish()]);

    assert_eq!(
        ctx.target_center(),
        premul(RED),
        "the encoded render must draw the image"
    );
}

/// An image uploaded into the region of a destroyed one is not wiped by that region's clear,
/// even when uploaded before the render that frees the region is submitted.
#[test]
fn gpu_atlas_reused_slot_shows_new_image() {
    let mut ctx = Ctx::new(64);
    let red = ctx.upload(4, 4, RED);
    let red_slot = ctx.slot(red);

    ctx.resources.destroy_image(red);
    let mut encoder = ctx.encoder();
    ctx.encode_render(&Scene::new(SIZE, SIZE), &mut encoder);
    let green = ctx.upload(4, 4, GREEN);
    assert_eq!(ctx.slot(green), red_slot, "green must reuse red's region");
    ctx.queue.submit([encoder.finish()]);

    ctx.render(&image_scene(green));
    assert_eq!(
        ctx.target_center(),
        premul(GREEN),
        "the new image must survive the clear"
    );
}

/// A region allocated for `render_to_atlas` starts out transparent, even where a destroyed
/// image was, so source-over compositing leaves no stale pixels where the scene doesn't paint.
#[test]
fn gpu_atlas_render_to_reused_region_starts_transparent() {
    let mut ctx = Ctx::new(64);
    let red = ctx.upload(8, 8, RED);
    let red_slot = ctx.slot(red);
    ctx.resources.destroy_image(red);
    ctx.render(&Scene::new(SIZE, SIZE));

    let id = ctx.resources.image_cache_mut().allocate(8, 8, 0).unwrap();
    let (atlas_id, [x, y]) = ctx.slot(id);
    assert_eq!((atlas_id, [x, y]), red_slot, "must reuse red's region");
    let image_cache = ctx.resources.image_cache();
    let atlas_config = *image_cache.atlas_manager().config();
    let atlas_count = u32::try_from(image_cache.atlas_count()).unwrap();
    let (atlas_width, atlas_height) = atlas_config.atlas_size;
    let mut scene = Scene::new(atlas_width, atlas_height);
    scene.set_paint(GREEN);
    // Only the left half of the region.
    let (left, top) = (f64::from(x), f64::from(y));
    scene.fill_rect(&Rect::new(left, top, left + 4.0, top + 8.0));
    ctx.renderer
        .render_to_atlas(
            &scene,
            atlas_count,
            atlas_config,
            &ctx.device,
            &ctx.queue,
            atlas_id,
            &TextureBindings::new(),
        )
        .unwrap();

    let atlas = ctx.read(ctx.renderer.atlas_texture(atlas_id));
    let (x, y) = (usize::from(x), usize::from(y));
    for py in y..y + 8 {
        for px in x..x + 8 {
            let expected = if px < x + 4 {
                premul(GREEN)
            } else {
                TRANSPARENT
            };
            let pixel = atlas[py * usize::from(atlas_width) + px];
            assert_eq!(pixel, expected, "atlas pixel ({px}, {py})");
        }
    }
}

/// Destroying an image clears exactly its region, including regions larger than one clear chunk.
#[test]
fn gpu_atlas_destroyed_image_region_is_cleared() {
    const ATLAS_SIZE: u16 = 1024;

    let mut ctx = Ctx::new(ATLAS_SIZE);
    // Larger than one clear chunk.
    let red = ctx.upload(600, 500, RED);
    let blue = ctx.upload(8, 8, BLUE);
    let (atlas_id, [bx, by]) = ctx.slot(blue);
    assert_eq!(ctx.slot(red).0, atlas_id, "both images must share a page");

    ctx.resources.destroy_image(red);
    ctx.render(&Scene::new(SIZE, SIZE));

    let atlas = ctx.read(ctx.renderer.atlas_texture(atlas_id));
    let blue_columns = usize::from(bx)..usize::from(bx) + 8;
    let blue_rows = usize::from(by)..usize::from(by) + 8;
    for (i, pixel) in atlas.iter().enumerate() {
        let (px, py) = (i % usize::from(ATLAS_SIZE), i / usize::from(ATLAS_SIZE));
        let expected = if blue_columns.contains(&px) && blue_rows.contains(&py) {
            premul(BLUE)
        } else {
            TRANSPARENT
        };
        assert_eq!(*pixel, expected, "atlas pixel ({px}, {py})");
    }
}

/// An allocation on an atlas page whose texture was never created can be destroyed.
#[test]
fn gpu_atlas_destroy_allocation_without_texture() {
    let mut ctx = Ctx::new(64);
    let id = ctx.resources.image_cache_mut().allocate(8, 8, 0).unwrap();

    assert!(ctx.resources.destroy_image(id), "the allocation must exist");
    assert!(
        !ctx.resources.destroy_image(id),
        "a queued allocation can't be destroyed again"
    );
    ctx.render(&Scene::new(SIZE, SIZE));

    assert!(
        ctx.resources.image_cache().get(id).is_none(),
        "the allocation must be freed"
    );
    assert!(
        !ctx.resources.destroy_image(id),
        "a freed allocation can't be destroyed again"
    );
}

/// Evicting cached glyphs clears their atlas regions.
#[test]
fn gpu_atlas_evicted_glyphs_are_cleared() {
    let mut ctx = Ctx::new(256);
    let (font, glyphs) = layout_glyphs_roboto("Hello", 12.0);
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_transform(Affine::translate((0.0, 12.0)));
    scene
        .glyph_run(&mut ctx.resources, &font)
        .font_size(12.0)
        .atlas_cache(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    ctx.render(&scene);
    let page = AtlasId::new(0);
    assert!(
        ctx.read(ctx.renderer.atlas_texture(page))
            .iter()
            .any(|pixel| *pixel != TRANSPARENT),
        "the glyphs must be cached in the atlas"
    );

    // Unused entries are evicted by the first eviction pass after they exceed the maximum age.
    let config = GlyphCacheConfig::default();
    for _ in 0..config.max_entry_age + config.eviction_frequency {
        ctx.render(&Scene::new(SIZE, SIZE));
    }

    assert!(
        ctx.read(ctx.renderer.atlas_texture(page))
            .iter()
            .all(|pixel| *pixel == TRANSPARENT),
        "evicted glyphs must be cleared"
    );
}
