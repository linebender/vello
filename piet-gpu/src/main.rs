use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use rand::{Rng, RngCore};

use piet::kurbo::{BezPath, Circle, Line, Point, Vec2};
use piet::{Color, RenderContext};

use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, Error, ImageLayout, MemFlags};

mod pico_svg;
mod render_ctx;

use render_ctx::PietGpuRenderContext;
use pico_svg::PicoSvg;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WIDTH: usize = TILE_W * WIDTH_IN_TILES;
const HEIGHT: usize = TILE_H * HEIGHT_IN_TILES;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const WIDTH_IN_TILEGROUPS: usize = 4;
const HEIGHT_IN_TILEGROUPS: usize = 96;
const TILEGROUP_STRIDE: usize = 2048;

const WIDTH_IN_TILES: usize = 128;
const HEIGHT_IN_TILES: usize = 96;
const PTCL_INITIAL_ALLOC: usize = 1024;

const K2_PER_TILE_SIZE: usize = 8;

const N_CIRCLES: usize = 1;

const NUM_FRAMES: usize = 2;

fn render_scene(rc: &mut impl RenderContext) {
    let mut rng = rand::thread_rng();
    for _ in 0..N_CIRCLES {
        let color = Color::from_rgba32_u32(rng.next_u32());
        let center = Point::new(
            rng.gen_range(0.0, WIDTH as f64),
            rng.gen_range(0.0, HEIGHT as f64),
        );
        let radius = rng.gen_range(0.0, 50.0);
        let circle = Circle::new(center, radius);
        rc.fill(circle, &color);
    }
    let mut path = BezPath::new();
    path.move_to((100.0, 1150.0));
    path.line_to((200.0, 1200.0));
    path.line_to((150.0, 1250.0));
    path.close_path();
    rc.fill(path, &Color::rgb8(128, 0, 128));
    rc.stroke(
        Line::new((100.0, 100.0), (200.0, 150.0)),
        &Color::WHITE,
        5.0,
    );
    //render_cardioid(rc);
    render_tiger(rc);
}

#[allow(unused)]
fn render_cardioid(rc: &mut impl RenderContext) {
    let n = 91;
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = BezPath::new();
    for i in 1..n {
        let p0 = center + Vec2::from_angle(i as f64 * dth) * r;
        let p1 = center + Vec2::from_angle(((i * 2) % n) as f64 * dth) * r;
        rc.fill(&Circle::new(p0, 8.0), &Color::WHITE);
        path.move_to(p0);
        path.line_to(p1);
        //rc.stroke(Line::new(p0, p1), &Color::BLACK, 2.0);
    }
    rc.stroke(&path, &Color::BLACK, 2.0);
}

fn render_tiger(rc: &mut impl RenderContext) {
    let xml_str = std::str::from_utf8(include_bytes!("../Ghostscript_Tiger.svg")).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(xml_str, 8.0).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

#[allow(unused)]
fn dump_scene(buf: &[u8]) {
    for i in 0..(buf.len() / 4) {
        let mut buf_u32 = [0u8; 4];
        buf_u32.copy_from_slice(&buf[i * 4..i * 4 + 4]);
        println!("{:4x}: {:8x}", i * 4, u32::from_le_bytes(buf_u32));
    }
}

#[allow(unused)]
fn dump_k1_data(k1_buf: &[u32]) {
    for i in 0..k1_buf.len() {
        if k1_buf[i] != 0 {
            println!("{:4x}: {:8x}", i, k1_buf[i]);
        }
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: (WIDTH_IN_TILES * 8) as f64,
            height: (HEIGHT_IN_TILES * 8) as f64,
        })
        .with_resizable(false) // currently not supported
        .build(&event_loop)?;

    let (instance, surface) = VkInstance::new(Some(&window))?;
    unsafe {
        let device = instance.device(surface.as_ref())?;
        let mut swapchain = instance.swapchain(&device, surface.as_ref().unwrap())?;

        let mut current_frame = 0;
        let present_semaphores = (0..NUM_FRAMES)
            .map(|_| device.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;
        let frame_fences = (0..NUM_FRAMES)
            .map(|_| device.create_fence(false))
            .collect::<Result<Vec<_>, Error>>()?;
        let mut cmd_buffers = (0..NUM_FRAMES)
            .map(|_| device.create_cmd_buf())
            .collect::<Result<Vec<_>, Error>>()?;
        let query_pools = (0..NUM_FRAMES)
            .map(|_| device.create_query_pool(6))
            .collect::<Result<Vec<_>, Error>>()?;

        let host = MemFlags::host_coherent();
        let dev = MemFlags::device_local();
        let mut ctx = PietGpuRenderContext::new();
        render_scene(&mut ctx);
        let scene = ctx.get_scene_buf();
        //dump_scene(&scene);
        let scene_buf = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, host)
            .unwrap();
        let scene_dev = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, dev)
            .unwrap();
        device.write_buffer(&scene_buf, &scene)?;
        let tilegroup_buf = device.create_buffer(4 * 1024 * 1024, dev)?;
        let ptcl_buf = device.create_buffer(48 * 1024 * 1024, dev)?;
        let segment_buf = device.create_buffer(64 * 1024 * 1024, dev)?;
        let fill_seg_buf = device.create_buffer(64 * 1024 * 1024, dev).unwrap();
        let image_buf = device.create_buffer((WIDTH * HEIGHT * 4) as u64, host)?;
        let image_dev = device.create_image2d(WIDTH as u32, HEIGHT as u32, dev)?;

        let k1_alloc_buf_host = device.create_buffer(4, host)?;
        let k1_alloc_buf_dev = device.create_buffer(4, dev)?;
        let k1_alloc_start = WIDTH_IN_TILEGROUPS * HEIGHT_IN_TILEGROUPS * TILEGROUP_STRIDE;
        device.write_buffer(&k1_alloc_buf_host, &[k1_alloc_start as u32])?;
        let k1_code = include_bytes!("../shader/kernel1.spv");
        let k1_pipeline = device
            .create_simple_compute_pipeline(k1_code, 3, 0)
            .unwrap();
        let k1_ds = device
            .create_descriptor_set(
                &k1_pipeline,
                &[&scene_dev, &tilegroup_buf, &k1_alloc_buf_dev],
                &[],
            )
            .unwrap();

        let k2s_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k2s_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k2s_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * K2_PER_TILE_SIZE;
        device
            .write_buffer(&k2s_alloc_buf_host, &[k2s_alloc_start as u32])
            .unwrap();
        let k2s_code = include_bytes!("../shader/kernel2s.spv");
        let k2s_pipeline = device
            .create_simple_compute_pipeline(k2s_code, 4, 0)
            .unwrap();
        let k2s_ds = device
            .create_descriptor_set(
                &k2s_pipeline,
                &[&scene_dev, &tilegroup_buf, &segment_buf, &k2s_alloc_buf_dev],
                &[],
            )
            .unwrap();

        let k2f_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k2f_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k2f_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * K2_PER_TILE_SIZE;
        device
            .write_buffer(&k2f_alloc_buf_host, &[k2f_alloc_start as u32])
            .unwrap();
        let k2f_code = include_bytes!("../shader/kernel2f.spv");
        let k2f_pipeline = device.create_simple_compute_pipeline(k2f_code, 4).unwrap();
        let k2f_ds = device
            .create_descriptor_set(
                &k2f_pipeline,
                &[
                    &scene_dev,
                    &tilegroup_buf,
                    &fill_seg_buf,
                    &k2f_alloc_buf_dev,
                ],
            )
            .unwrap();

        let k3_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k3_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k3_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * PTCL_INITIAL_ALLOC;
        device
            .write_buffer(&k3_alloc_buf_host, &[k3_alloc_start as u32])
            .unwrap();
        let k3_code = include_bytes!("../shader/kernel3.spv");
        let k3_pipeline = device.create_simple_compute_pipeline(k3_code, 6, 0).unwrap();
        let k3_ds = device
            .create_descriptor_set(
                &k3_pipeline,
                &[
                    &scene_dev,
                    &tilegroup_buf,
                    &segment_buf,
                    &fill_seg_buf,
                    &ptcl_buf,
                    &k3_alloc_buf_dev,
                ],
                &[],
            )
            .unwrap();

        let k4_code = include_bytes!("../shader/kernel4.spv");
        let k4_pipeline = device.create_simple_compute_pipeline(k4_code, 3, 1).unwrap();
        let k4_ds = device
            .create_descriptor_set(&k4_pipeline, &[&ptcl_buf, &segment_buf, &fill_seg_buf], &[&image_dev])
            .unwrap();
        let query_pool = &query_pools[0];
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.copy_buffer(&scene_buf, &scene_dev);
        // Note: we could use one alloc buf and reuse it. But we'll stick with
        // multiple ones for clarity.
        cmd_buf.copy_buffer(&k1_alloc_buf_host, &k1_alloc_buf_dev);
        cmd_buf.copy_buffer(&k2s_alloc_buf_host, &k2s_alloc_buf_dev);
        cmd_buf.copy_buffer(&k2f_alloc_buf_host, &k2f_alloc_buf_dev);
        cmd_buf.copy_buffer(&k3_alloc_buf_host, &k3_alloc_buf_dev);
        // Note: these clears aren't necessary, and are here to make inspection
        // of the buffers cleaner. Can likely be removed.
        cmd_buf.clear_buffer(&tilegroup_buf);
        cmd_buf.clear_buffer(&ptcl_buf);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(&image_dev, ImageLayout::Undefined, ImageLayout::General);
        cmd_buf.reset_query_pool(&query_pool);
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(
            &k1_pipeline,
            &k1_ds,
            ((WIDTH / 512) as u32, (HEIGHT / 512) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k2s_pipeline,
            &k2s_ds,
            ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 2);
        // Note: this barrier is not necessary (k2f does not depend on
        // k2s output), but I'm keeping it here to increase transparency
        // of performance.
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k2f_pipeline,
            &k2f_ds,
            ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 2),
        );
        cmd_buf.write_timestamp(&query_pool, 3);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k3_pipeline,
            &k3_ds,
            ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 3),
        );
        cmd_buf.write_timestamp(&query_pool, 4);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k4_pipeline,
            &k4_ds,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 5);
        cmd_buf.memory_barrier();
        cmd_buf.image_barrier(&image_dev, ImageLayout::General, ImageLayout::BlitSrc);
        cmd_buf.copy_image_to_buffer(&image_dev, &image_buf);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf, &[], &[], Some(&frame_fences[0]))?;
        device.wait_and_reset(&[frame_fences[0]])?;
        let timestamps = device.reap_query_pool(&query_pool).unwrap();
        println!("Kernel 1 time: {:.3}ms", timestamps[0] * 1e3);
        println!(
            "Kernel 2s time: {:.3}ms",
            (timestamps[1] - timestamps[0]) * 1e3
        );
        println!(
            "Kernel 2f time: {:.3}ms",
            (timestamps[2] - timestamps[1]) * 1e3
        );
        println!(
            "Kernel 3 time: {:.3}ms",
            (timestamps[3] - timestamps[2]) * 1e3
        );
        println!(
            "Render time: {:.3}ms",
            (timestamps[4] - timestamps[3]) * 1e3
        );

        /*
        let mut k1_data: Vec<u32> = Default::default();
        device.read_buffer(&segment_buf, &mut k1_data).unwrap();
        dump_k1_data(&k1_data);
        */

        if false {
            let mut img_data: Vec<u8> = Default::default();
            // Note: because png can use a `&[u8]` slice, we could avoid an extra copy
            // (probably passing a slice into a closure). But for now: keep it simple.
            device.read_buffer(&image_buf, &mut img_data).unwrap();

            // Write image as PNG file.
            let path = Path::new("image.png");
            let file = File::create(path).unwrap();
            let ref mut w = BufWriter::new(file);

            let mut encoder = png::Encoder::new(w, WIDTH as u32, HEIGHT as u32);
            encoder.set_color(png::ColorType::RGBA);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();

            writer.write_image_data(&img_data).unwrap();
        }

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(window_id) if window_id == window.id() => {
                    let frame_idx = current_frame % NUM_FRAMES;
                    let query_pool = &query_pools[frame_idx];

                    if current_frame >= NUM_FRAMES {
                        device.wait_and_reset(&[frame_fences[frame_idx]]).unwrap();

                        let timestamps = device.reap_query_pool(query_pool).unwrap();
                        window.set_title(&format!("k1: {:.3}ms, k2: {:.3}ms, k3: {:.3}ms, k4: {:.3}ms",
                            timestamps[0] * 1e3,
                            (timestamps[1] - timestamps[0]) * 1e3,
                            (timestamps[2] - timestamps[1]) * 1e3,
                            (timestamps[3] - timestamps[2]) * 1e3,
                        ));
                    }

                    let (image_idx, acquisition_semaphore) = swapchain.next().unwrap();
                    let swap_image = swapchain.image(image_idx);
                    let cmd_buf = &mut cmd_buffers[frame_idx];
                    cmd_buf.begin();
                    cmd_buf.reset_query_pool(&query_pool);
                    cmd_buf.copy_buffer(&scene_buf, &scene_dev);
                    cmd_buf.copy_buffer(&k1_alloc_buf_host, &k1_alloc_buf_dev);
                    cmd_buf.copy_buffer(&k2s_alloc_buf_host, &k2s_alloc_buf_dev);
                    cmd_buf.copy_buffer(&k3_alloc_buf_host, &k3_alloc_buf_dev);
                    cmd_buf.clear_buffer(&tilegroup_buf);
                    cmd_buf.clear_buffer(&ptcl_buf);
                    cmd_buf.memory_barrier();
                    cmd_buf.write_timestamp(&query_pool, 0);
                    cmd_buf.dispatch(
                        &k1_pipeline,
                        &k1_ds,
                        ((WIDTH / 512) as u32, (HEIGHT / 512) as u32, 1),
                    );
                    cmd_buf.write_timestamp(&query_pool, 1);
                    cmd_buf.memory_barrier();
                    cmd_buf.dispatch(
                        &k2s_pipeline,
                        &k2s_ds,
                        ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 1),
                    );
                    cmd_buf.write_timestamp(&query_pool, 2);
                    cmd_buf.memory_barrier();
                    cmd_buf.dispatch(
                        &k3_pipeline,
                        &k3_ds,
                        ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 1),
                    );
                    cmd_buf.write_timestamp(&query_pool, 3);
                    cmd_buf.memory_barrier();
                    cmd_buf.image_barrier(&image_dev, ImageLayout::BlitSrc, ImageLayout::General);
                    cmd_buf.dispatch(
                        &k4_pipeline,
                        &k4_ds,
                        ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
                    );
                    cmd_buf.write_timestamp(&query_pool, 4);
                    cmd_buf.memory_barrier();
                    cmd_buf.image_barrier(
                        &swap_image,
                        ImageLayout::Undefined,
                        ImageLayout::BlitDst,
                    );
                    cmd_buf.image_barrier(&image_dev, ImageLayout::General, ImageLayout::BlitSrc);
                    cmd_buf.blit_image(&image_dev, &swap_image);
                    cmd_buf.image_barrier(
                        &swap_image,
                        ImageLayout::BlitDst,
                        ImageLayout::Present,
                    );
                    cmd_buf.finish();

                    device
                        .run_cmd_buf(
                            &cmd_buf,
                            &[acquisition_semaphore],
                            &[present_semaphores[frame_idx]],
                            Some(&frame_fences[frame_idx]),
                        )
                        .unwrap();

                    swapchain
                        .present(image_idx, &[present_semaphores[frame_idx]])
                        .unwrap();

                    current_frame += 1;
                }
                _ => (),
            }
        })
    }
}
