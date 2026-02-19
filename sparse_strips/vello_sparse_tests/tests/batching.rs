// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests that verify blit rect batching produces identical output to unbatched rendering.
//!
//! These tests exercise the dirty bounding box tracking in `Scene` by rendering the same
//! scene twice -- once with blit batching enabled and once with it disabled -- and asserting
//! that the resulting pixmaps are identical. Any difference indicates that the dirty rect
//! was too small, causing a blit rect to be incorrectly folded into a previous blit batch.

#[cfg(not(target_arch = "wasm32"))]
mod gpu_tests {
    use std::sync::Mutex;

    use vello_common::color::palette::css::{BLUE, RED, WHITE};
    use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
    use vello_common::paint::{Image, ImageSource};
    use vello_common::peniko::ImageSampler;
    use vello_common::pixmap::Pixmap;
    use vello_hybrid::Scene;

    use crate::renderer::HybridContext;

    const WIDTH: u16 = 200;
    const HEIGHT: u16 = 200;

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
        let ctx = HybridContext::new(WIDTH, HEIGHT);
        let image_id = ctx.upload_image(&test_image);
        let img_src = ImageSource::OpaqueId(image_id);

        // Render with batching ON (default).
        let pixmap_on = {
            let mut scene = Scene::new(WIDTH, HEIGHT);
            build_scene(&mut scene, img_src.clone());
            ctx.render_scene(&scene)
        };

        // Render with batching OFF.
        let pixmap_off = {
            let mut scene = Scene::new(WIDTH, HEIGHT);
            scene.set_blit_batching(false);
            build_scene(&mut scene, img_src);
            ctx.render_scene(&scene)
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
                        first_diffs.push(format!("  ({x},{y}): on={:?} off={:?}", a, b));
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

    /// Test that a blit survives when there are 2 batches
    /// (batching disabled, 2 blits with strips in between).
    #[test]
    fn blit_survives_two_batches() {
        let _guard = GPU_MUTEX.lock().unwrap();
        let test_image = make_test_image();
        let gpu = HybridContext::new(WIDTH, HEIGHT);
        let image_id = gpu.upload_image(&test_image);

        let mut scene = Scene::new(WIDTH, HEIGHT);
        scene.set_blit_batching(false);

        // White background.
        scene.set_paint(WHITE);
        scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));
        // Blit image at top-left. -> creates batch 0
        scene.set_paint(Image {
            image: ImageSource::OpaqueId(image_id),
            sampler: ImageSampler::default(),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        // Red fill that doesn't overlap the blit.
        scene.set_paint(RED);
        scene.fill_path(&Rect::new(50.0, 50.0, 150.0, 150.0).to_path(0.1));
        // Second blit far away -> creates batch 1 because batching is disabled.
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
            "First blit at (0,0) should survive with 2 batches, got {:?}",
            px
        );
    }

    /// Minimal sanity test: a single blit with no strip batches should produce
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

    /// Test that a blit survives a subsequent strip batch (no second blit, no batching involved).
    /// If this fails, the strip batch is overwriting blit content.
    #[test]
    fn blit_survives_strip_batch() {
        let _guard = GPU_MUTEX.lock().unwrap();
        let test_image = make_test_image();
        let gpu = HybridContext::new(WIDTH, HEIGHT);
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
            "Blit at (0,0) should survive subsequent strip batch, got {:?}",
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
                Affine::translate((100.0, 100.0)) * Affine::rotate(std::f64::consts::FRAC_PI_4),
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
