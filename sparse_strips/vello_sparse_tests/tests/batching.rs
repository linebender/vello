// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests that verify blit rect batching produces identical output to unbatched rendering.
//!
//! These tests exercise the dirty bounding box tracking in `Scene` by rendering the same
//! scene twice -- once with blit batching enabled and once with it disabled -- and asserting
//! that the resulting pixmaps are identical. Any difference indicates that the dirty rect
//! was too small, causing a blit rect to be incorrectly batched into a previous flush point.

#[cfg(not(target_arch = "wasm32"))]
mod gpu_tests {
    use std::sync::Mutex;

    use vello_common::color::palette::css::{BLUE, RED, WHITE};
    use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
    use vello_common::paint::{Image, ImageSource};
    use vello_common::peniko::ImageSampler;
    use vello_common::pixmap::Pixmap;
    use vello_hybrid::Scene;

    const WIDTH: u16 = 200;
    const HEIGHT: u16 = 200;

    struct GpuContext {
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: vello_hybrid::Renderer,
    }

    impl GpuContext {
        fn new(width: u16, height: u16) -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))
                .expect("Failed to find an appropriate adapter");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Batching Test Device"),
                    required_features: wgpu::Features::empty(),
                    ..Default::default()
                }))
                .expect("Failed to create device");

            let renderer = vello_hybrid::Renderer::new(
                &device,
                &vello_hybrid::RenderTargetConfig {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    width: width.into(),
                    height: height.into(),
                },
            );

            Self {
                device,
                queue,
                renderer,
            }
        }

        fn upload_image(&mut self, pixmap: &Pixmap) -> vello_common::paint::ImageId {
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Upload Image"),
                    });
            let id =
                self.renderer
                    .upload_image(&self.device, &self.queue, &mut encoder, pixmap);
            self.queue.submit([encoder.finish()]);
            id
        }

        fn render_scene(&mut self, scene: &Scene) -> Pixmap {
            let width = scene.width();
            let height = scene.height();

            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Batching Test Render Target"),
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

            let render_size = vello_hybrid::RenderSize {
                width: width.into(),
                height: height.into(),
            };

            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Batching Test Render"),
                    });
            self.renderer
                .render(
                    scene,
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &render_size,
                    &texture_view,
                )
                .unwrap();

            let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
            let texture_copy_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batching Test Output Buffer"),
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

            let mut pixmap = Pixmap::new(width, height);
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

            pixmap
        }
    }

    /// Create a small solid-colour test image for use as a blit source.
    fn make_test_image() -> Pixmap {
        let mut pixmap = Pixmap::new(10, 10);
        for pixel in pixmap.data_mut() {
            *pixel = vello_common::peniko::color::PremulRgba8 {
                r: 0,
                g: 128,
                b: 255,
                a: 255,
            };
        }
        pixmap
    }

    /// Assert that rendering a scene with blit batching enabled produces identical
    /// output to rendering the same scene with blit batching disabled.
    fn assert_batching_equivalent(build_scene: impl Fn(&mut Scene, ImageSource)) {
        let test_image = make_test_image();
        let mut gpu = GpuContext::new(WIDTH, HEIGHT);
        let image_id = gpu.upload_image(&test_image);
        let img_src = ImageSource::OpaqueId(image_id);

        // Render with batching ON (default).
        let pixmap_on = {
            let mut scene = Scene::new(WIDTH, HEIGHT);
            build_scene(&mut scene, img_src.clone());
            gpu.render_scene(&scene)
        };

        // Render with batching OFF.
        let pixmap_off = {
            let mut scene = Scene::new(WIDTH, HEIGHT);
            scene.set_blit_batching(false);
            build_scene(&mut scene, img_src);
            gpu.render_scene(&scene)
        };

        let data_on = pixmap_on.data_as_u8_slice();
        let data_off = pixmap_off.data_as_u8_slice();
        if data_on != data_off {
            let w = WIDTH as usize;
            let mut diff_count = 0;
            let mut first_diffs = Vec::new();
            for (i, (a, b)) in data_on
                .chunks_exact(4)
                .zip(data_off.chunks_exact(4))
                .enumerate()
            {
                if a != b {
                    diff_count += 1;
                    if first_diffs.len() < 5 {
                        let x = i % w;
                        let y = i / w;
                        first_diffs.push(format!(
                            "  ({x},{y}): on={:?} off={:?}",
                            a, b
                        ));
                    }
                }
            }
            panic!(
                "Rendering with blit batching enabled differs from rendering with blit batching \
                 disabled: {diff_count} pixels differ out of {}\nFirst diffs:\n{}",
                data_on.len() / 4,
                first_diffs.join("\n")
            );
        }
    }

    /// Serialise concurrent GPU tests to avoid wgpu segfaults.
    static GPU_MUTEX: Mutex<()> = Mutex::new(());

    /// Test that a blit survives when there are 2 flush points
    /// (batching disabled, 2 blits with strips in between).
    #[test]
    fn blit_survives_two_flush_points() {
        let _guard = GPU_MUTEX.lock().unwrap();
        let test_image = make_test_image();
        let mut gpu = GpuContext::new(WIDTH, HEIGHT);
        let image_id = gpu.upload_image(&test_image);

        let mut scene = Scene::new(WIDTH, HEIGHT);
        scene.set_blit_batching(false);

        // White background.
        scene.set_paint(WHITE);
        scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));
        // Blit image at top-left. -> creates FP0
        scene.set_paint(Image {
            image: ImageSource::OpaqueId(image_id),
            sampler: ImageSampler::default(),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        // Red fill that doesn't overlap the blit.
        scene.set_paint(RED);
        scene.fill_path(&Rect::new(50.0, 50.0, 150.0, 150.0).to_path(0.1));
        // Second blit far away -> creates FP1 because batching is disabled.
        scene.set_paint(Image {
            image: ImageSource::OpaqueId(image_id),
            sampler: ImageSampler::default(),
        });
        scene.fill_rect(&Rect::new(160.0, 160.0, 170.0, 170.0));

        let pixmap = gpu.render_scene(&scene);
        let px = pixmap.data()[0]; // pixel at (0,0)
        assert_eq!(
            [px.r, px.g, px.b, px.a],
            [0, 128, 255, 255],
            "First blit at (0,0) should survive with 2 flush points, got {:?}",
            px
        );
    }

    /// Minimal sanity test: a single blit with no strips should produce
    /// identical output regardless of the batching flag.
    #[test]
    fn batching_equiv_single_blit() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        });
    }

    /// Test that a blit survives a subsequent strip pass (no second blit, no batching involved).
    /// If this fails, the strip render pass is overwriting blit content.
    #[test]
    fn blit_survives_strip_pass() {
        let _guard = GPU_MUTEX.lock().unwrap();
        let test_image = make_test_image();
        let mut gpu = GpuContext::new(WIDTH, HEIGHT);
        let image_id = gpu.upload_image(&test_image);

        let mut scene = Scene::new(WIDTH, HEIGHT);
        // White background.
        scene.set_paint(WHITE);
        scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));
        // Blit image at top-left.
        scene.set_paint(Image {
            image: ImageSource::OpaqueId(image_id),
            sampler: ImageSampler::default(),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        // Red fill that doesn't overlap the blit.
        scene.set_paint(RED);
        scene.fill_path(&Rect::new(50.0, 50.0, 150.0, 150.0).to_path(0.1));

        let pixmap = gpu.render_scene(&scene);
        // The blit at (0,0) should be blue, not white.
        let px = pixmap.data()[0]; // pixel at (0,0)
        assert_eq!(
            [px.r, px.g, px.b, px.a],
            [0, 128, 255, 255],
            "Blit at (0,0) should survive subsequent strip pass, got {:?}",
            px
        );
    }

    #[test]
    fn batching_equiv_miter_join() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(WHITE);
            scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));

            // First blit: image at top-left.
            scene.set_paint(Image {
                image: img_src.clone(),
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));

            // V-shaped path with a sharp angle producing a large miter spike.
            let mut path = BezPath::new();
            path.move_to((50.0, 150.0));
            path.line_to((100.0, 30.0));
            path.line_to((150.0, 150.0));

            scene.set_paint(RED);
            scene.set_stroke(Stroke {
                width: 10.0,
                join: Join::Miter,
                miter_limit: 10.0,
                ..Default::default()
            });
            scene.stroke_path(&path);

            // Second blit: image near the miter spike tip at the top of the V.
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(95.0, 20.0, 105.0, 30.0));
        });
    }

    #[test]
    fn batching_equiv_square_cap() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(WHITE);
            scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));

            // First blit at bottom.
            scene.set_paint(Image {
                image: img_src.clone(),
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 180.0, 10.0, 190.0));

            // Horizontal line with square caps.
            let mut path = BezPath::new();
            path.move_to((50.0, 100.0));
            path.line_to((150.0, 100.0));

            scene.set_paint(BLUE);
            scene.set_stroke(Stroke {
                width: 10.0,
                start_cap: Cap::Square,
                end_cap: Cap::Square,
                join: Join::Bevel,
                ..Default::default()
            });
            scene.stroke_path(&path);

            // Blit just past the right endpoint, in the square cap extension zone.
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(150.0, 95.0, 160.0, 105.0));
        });
    }

    #[test]
    fn batching_equiv_rotated_stroke() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(WHITE);
            scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));

            // First blit in corner.
            scene.set_paint(Image {
                image: img_src.clone(),
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));

            // Miter-joined V-shape under a 45-degree rotation.
            let mut path = BezPath::new();
            path.move_to((-30.0, -30.0));
            path.line_to((0.0, 30.0));
            path.line_to((30.0, -30.0));

            scene.set_transform(
                Affine::translate((100.0, 100.0))
                    * Affine::rotate(std::f64::consts::FRAC_PI_4),
            );
            scene.set_paint(RED);
            scene.set_stroke(Stroke {
                width: 8.0,
                join: Join::Miter,
                miter_limit: 10.0,
                ..Default::default()
            });
            scene.stroke_path(&path);
            scene.set_transform(Affine::IDENTITY);

            // Blit near the rotated miter spike.
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(90.0, 50.0, 110.0, 70.0));
        });
    }

    #[test]
    fn batching_equiv_round_join() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(WHITE);
            scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));

            // First blit.
            scene.set_paint(Image {
                image: img_src.clone(),
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));

            // Round-joined stroke -- half-width inflation is correct for this case.
            let mut path = BezPath::new();
            path.move_to((50.0, 150.0));
            path.line_to((100.0, 50.0));
            path.line_to((150.0, 150.0));

            scene.set_paint(BLUE);
            scene.set_stroke(Stroke {
                width: 10.0,
                join: Join::Round,
                ..Default::default()
            });
            scene.stroke_path(&path);

            // Blit in a non-overlapping area.
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(170.0, 170.0, 180.0, 180.0));
        });
    }

    #[test]
    fn batching_equiv_fill_path() {
        let _guard = GPU_MUTEX.lock().unwrap();
        assert_batching_equivalent(|scene, img_src| {
            scene.set_paint(WHITE);
            scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));

            // First blit.
            scene.set_paint(Image {
                image: img_src.clone(),
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));

            // Filled rectangle (no stroke).
            scene.set_paint(RED);
            scene.fill_path(&Rect::new(50.0, 50.0, 150.0, 150.0).to_path(0.1));

            // Blit in a non-overlapping area.
            scene.set_paint(Image {
                image: img_src,
                sampler: ImageSampler::default(),
            });
            scene.fill_rect(&Rect::new(160.0, 160.0, 170.0, 170.0));
        });
    }
}
