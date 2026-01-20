// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Infrastructure used to implement the tests in the Vello API tests.

use vello_api::{
    PaintScene,
    peniko::{Color, Fill},
};
use vello_cpu::{
    Level, Pixmap, RenderSettings,
    api::CPUScenePainter,
    kurbo::{Affine, Rect},
};
use vello_hybrid::api::HybridScenePainter;

use crate::util::check_pixmap_ref;

#[derive(Copy, Clone)]
pub(crate) struct TestParams {
    pub(crate) name: &'static str,
    /// The tolerances for a pixel to count as different for different scenarios.
    ///
    /// See the descriptions in [`vello_dev_macros`] for more information (e.g. `DEFAULT_CPU_U8_TOLERANCE`).
    ///
    /// Note that these are *not* actually used by the functions in this module; instead, the specific field is chosen
    /// for each test in the macro.
    /// This pattern makes the expanded macro for the tests easier to read.
    pub(crate) cpu_u8_scalar_tolerance: u8,
    pub(crate) cpu_f32_scalar_tolerance: u8,
    pub(crate) cpu_u8_simd_tolerance: u8,
    pub(crate) cpu_f32_simd_tolerance: u8,
    pub(crate) hybrid_tolerance: u8,
    /// Number of pixels allowed to exceed the threshold of difference.
    pub(crate) diff_pixels: u16,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) reference_image_bytes: Option<&'static [u8]>,
    pub(crate) transparent: bool,
    pub(crate) no_ref: bool,
}

pub(crate) fn run_test_cpu<M>(
    scene_func: impl SceneFunction<CPUScenePainter, M>,
    params: TestParams,
    render_settings: RenderSettings,
    specific_name: &str,
    threshold: u8,
    is_reference: bool,
) {
    let renderer = vello_cpu::RenderContext::new_with(params.width, params.height, render_settings);
    let mut scene = CPUScenePainter {
        render_context: renderer,
    };
    if !params.transparent {
        scene.set_solid_brush(Color::WHITE);
        scene.fill_path(
            Affine::IDENTITY,
            Fill::EvenOdd,
            Rect::new(0., 0., params.width as f64, params.height as f64),
        );
    }

    scene_func.run(&mut scene);
    let mut render_context = scene.render_context;
    render_context.flush();
    let mut target = Pixmap::new(params.width, params.height);
    render_context.render_to_pixmap(&mut target);

    if !params.no_ref {
        check_pixmap_ref(
            params.name,
            specific_name,
            threshold,
            params.diff_pixels,
            is_reference,
            params.reference_image_bytes,
            target,
        );
    }
}

// There are two reasons we don't run hybrid tests on web:
// - WebGPU support isn't quite there yet in browsers
// - This code assumes that it can block (the `device.poll(wgpu::PollType::wait_indefinitely())`), but that isn't valid on the web.
//
// Tweaking this code to be async as required on the web would be hard.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_test_hybrid_wgpu<M>(
    scene_func: impl SceneFunction<HybridScenePainter, M>,
    params: TestParams,
    render_settings: vello_hybrid::RenderSettings,
    specific_name: &str,
    threshold: u8,
    is_reference: bool,
) {
    // On some platforms using `cargo test` triggers segmentation faults in wgpu when the GPU
    // tests are run in parallel (likely related to the number of device resources being
    // requested simultaneously). This is "fixed" by putting a mutex around this method,
    // ensuring only one set of device resources is alive at the same time. This slows down
    // testing when `cargo test` is used.
    //
    // Testing with `cargo nextest` (as on CI) is not meaningfully slowed down. `nextest` runs
    // each test in its own process (<https://nexte.st/docs/design/why-process-per-test/>),
    // meaning there is no contention on this mutex.
    // (We think it's reasonable for one "new style" and one "legacy style" test to be
    // running at the same time)
    let _guard = {
        use std::sync::Mutex;
        static M: Mutex<()> = Mutex::new(());
        M.lock().unwrap()
    };

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

    let format = wgpu::TextureFormat::Rgba8Unorm;

    let mut scene = HybridScenePainter {
        scene: vello_hybrid::Scene::new_with(params.width, params.height, render_settings),
    };
    if !params.transparent {
        scene.set_solid_brush(Color::WHITE);
        scene.fill_path(
            Affine::IDENTITY,
            Fill::EvenOdd,
            Rect::new(0., 0., params.width as f64, params.height as f64),
        );
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Vello Hybrid Render"),
    });
    scene_func.run(&mut scene);

    // Create renderer and render the scene to the texture.
    let mut renderer = vello_hybrid::Renderer::new_with(
        &device,
        &vello_hybrid::RenderTargetConfig {
            format,
            // TODO: I don't think these width and height are ever actually used?
            width: params.width.into(),
            height: params.height.into(),
        },
        render_settings,
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(params.name),
        size: wgpu::Extent3d {
            width: params.width.into(),
            height: params.height.into(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor {
        label: Some(params.name),
        ..Default::default()
    });
    renderer
        .render(
            &scene.scene,
            &device,
            &queue,
            &mut encoder,
            &vello_hybrid::RenderSize {
                width: params.width.into(),
                height: params.height.into(),
            },
            &view,
        )
        .expect("Better error handling.");

    if params.no_ref {
        queue.submit([encoder.finish()]);
        // Make sure the GPU work completes.
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    } else {
        let mut pixmap = Pixmap::new(params.width, params.height);
        // Create a buffer to copy the texture data
        let bytes_per_row = (u32::from(params.width) * 4).next_multiple_of(256);
        let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: u64::from(bytes_per_row) * u64::from(params.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
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
                width: params.width.into(),
                height: params.height.into(),
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
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        // Read back the pixel data
        for (row, buf) in texture_copy_buffer
            .slice(..)
            .get_mapped_range()
            .chunks_exact(bytes_per_row as usize)
            .zip(
                pixmap
                    .data_as_u8_slice_mut()
                    .chunks_exact_mut(params.width as usize * 4),
            )
        {
            buf.copy_from_slice(&row[0..params.width as usize * 4]);
        }
        texture_copy_buffer.unmap();
        check_pixmap_ref(
            params.name,
            specific_name,
            threshold,
            params.diff_pixels,
            is_reference,
            params.reference_image_bytes,
            pixmap,
        );
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn run_test_hybrid_webgl<M>(
    scene_func: impl SceneFunction<HybridScenePainter, M>,
    params: TestParams,
    render_settings: vello_hybrid::RenderSettings,
    specific_name: &str,
    threshold: u8,
) {
    use wasm_bindgen::JsCast;
    use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

    // Create an offscreen HTMLCanvasElement, render the test image to it, and finally read off
    // the pixmap for diff checking.
    let document = web_sys::window().unwrap().document().unwrap();

    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    canvas.set_width(params.width.into());
    canvas.set_height(params.height.into());

    let renderer = vello_hybrid::api::VelloHybridWebgl::new(&canvas, render_settings);

    let mut scene = renderer
        .create_scene(
            &vello_hybrid::api::VelloHybridWebgl::CANVAS_TEXTURE_ID,
            SceneOptions {
                target: None,
                clear_color: (!params.transparent).then_some(Color::WHITE),
            },
        )
        .unwrap();
    scene_func.run(&mut scene, &*renderer);
    renderer.queue_render(scene);

    if !params.no_ref {
        let gl = renderer.gl_context();

        let width = params.width;
        let height = params.height;

        let mut pixmap = Pixmap::new(width, height);

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

        check_pixmap_ref(
            params.name,
            specific_name,
            threshold,
            params.diff_pixels,
            // Can never be a reference image
            false,
            params.reference_image_bytes,
            pixmap,
        );
    }
}

/// Convert a level string into a Fearless SIMD [`Level`].
///
/// Returns `None` if the level is known, but happens to not be supported on this machine.
///
/// If that is the case, your best option is to pass the test as "succeeding".
#[track_caller]
pub(crate) fn parse_level(level: &str) -> Option<Level> {
    match level {
        #[cfg(target_arch = "aarch64")]
        "neon" => Some(Level::Neon(
            Level::try_detect().unwrap_or(Level::fallback()).as_neon()?,
        )),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        "wasm_simd128" => Some(Level::WasmSimd128(
            Level::try_detect()
                .unwrap_or(Level::fallback())
                .as_wasm_simd128()?,
        )),
        // TODO: It's a known Fearless SIMD bug that we can't use `as_sse4` here
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse42" => {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                Some(Level::Sse4_2(unsafe {
                    vello_common::fearless_simd::Sse4_2::new_unchecked()
                }))
            } else {
                None
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx2" => {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Some(Level::Avx2(unsafe {
                    vello_common::fearless_simd::Avx2::new_unchecked()
                }))
            } else {
                None
            }
        }
        "fallback" => Some(Level::fallback()),
        _ => panic!("unknown level on this architecture: {level}"),
    }
}

pub(crate) trait SceneFunction<S: PaintScene, M> {
    fn run(self, scene: &mut S /* renderer: &dyn Renderer */);
}

/// Marker type used to disambiguate implementations of `SceneFunction`.
pub(crate) struct Single;
impl<S: PaintScene, F> SceneFunction<S, Single> for F
where
    F: Fn(&mut S),
{
    fn run(self, scene: &mut S) {
        self(scene);
    }
}

// /// Marker type used to disambiguate implementations of `SceneFunction`.
// pub(crate) struct Double;
// impl<S: PaintScene, F> SceneFunction<S, Double> for F
// where
//     F: Fn(&mut S, &dyn Renderer),
// {
//     fn run(self, scene: &mut S, renderer: &dyn Renderer) {
//         self(scene, renderer);
//     }
// }
