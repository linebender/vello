use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use rand::{Rng, RngCore};

use piet::kurbo::{BezPath, Circle, Line, Point, Vec2};
use piet::{Color, RenderContext};

use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

mod render_ctx;

use render_ctx::PietGpuRenderContext;

const WIDTH: usize = 2048;
const HEIGHT: usize = 1536;

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
    rc.stroke(
        Line::new((100.0, 100.0), (200.0, 150.0)),
        &Color::WHITE,
        5.0,
    );
    render_cardioid(rc);
}

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

fn main() {
    let instance = VkInstance::new().unwrap();
    unsafe {
        let device = instance.device().unwrap();
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
        device.write_buffer(&scene_buf, &scene).unwrap();
        let tilegroup_buf = device.create_buffer(4 * 1024 * 1024, dev).unwrap();
        let ptcl_buf = device.create_buffer(48 * 1024 * 1024, dev).unwrap();
        let segment_buf = device.create_buffer(64 * 1024 * 1024, dev).unwrap();
        let image_buf = device
            .create_buffer((WIDTH * HEIGHT * 4) as u64, host)
            .unwrap();
        let image_dev = device
            .create_buffer((WIDTH * HEIGHT * 4) as u64, dev)
            .unwrap();

        let k1_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k1_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k1_alloc_start = WIDTH_IN_TILEGROUPS * HEIGHT_IN_TILEGROUPS * TILEGROUP_STRIDE;
        device
            .write_buffer(&k1_alloc_buf_host, &[k1_alloc_start as u32])
            .unwrap();
        let k1_code = include_bytes!("../shader/kernel1.spv");
        let k1_pipeline = device.create_simple_compute_pipeline(k1_code, 3).unwrap();
        let k1_ds = device
            .create_descriptor_set(
                &k1_pipeline,
                &[&scene_dev, &tilegroup_buf, &k1_alloc_buf_dev],
            )
            .unwrap();

        let k2s_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k2s_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k2s_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * K2_PER_TILE_SIZE;
        device
            .write_buffer(&k2s_alloc_buf_host, &[k2s_alloc_start as u32])
            .unwrap();
        let k2s_code = include_bytes!("../shader/kernel2s.spv");
        let k2s_pipeline = device.create_simple_compute_pipeline(k2s_code, 4).unwrap();
        let k2s_ds = device
            .create_descriptor_set(
                &k2s_pipeline,
                &[&scene_dev, &tilegroup_buf, &segment_buf, &k2s_alloc_buf_dev],
            )
            .unwrap();

        let k3_alloc_buf_host = device.create_buffer(4, host).unwrap();
        let k3_alloc_buf_dev = device.create_buffer(4, dev).unwrap();
        let k3_alloc_start = WIDTH_IN_TILES * HEIGHT_IN_TILES * PTCL_INITIAL_ALLOC;
        device
            .write_buffer(&k3_alloc_buf_host, &[k3_alloc_start as u32])
            .unwrap();
        let k3_code = include_bytes!("../shader/kernel3.spv");
        let k3_pipeline = device.create_simple_compute_pipeline(k3_code, 5).unwrap();
        let k3_ds = device
            .create_descriptor_set(
                &k3_pipeline,
                &[
                    &scene_dev,
                    &tilegroup_buf,
                    &segment_buf,
                    &ptcl_buf,
                    &k3_alloc_buf_dev,
                ],
            )
            .unwrap();

        let k4_code = include_bytes!("../shader/kernel4.spv");
        let k4_pipeline = device.create_simple_compute_pipeline(k4_code, 3).unwrap();
        let k4_ds = device
            .create_descriptor_set(&k4_pipeline, &[&ptcl_buf, &segment_buf, &image_dev])
            .unwrap();

        let query_pool = device.create_query_pool(5).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.copy_buffer(&scene_buf, &scene_dev);
        cmd_buf.copy_buffer(&k1_alloc_buf_host, &k1_alloc_buf_dev);
        cmd_buf.copy_buffer(&k2s_alloc_buf_host, &k2s_alloc_buf_dev);
        cmd_buf.copy_buffer(&k3_alloc_buf_host, &k3_alloc_buf_dev);
        cmd_buf.clear_buffer(&tilegroup_buf);
        cmd_buf.clear_buffer(&ptcl_buf);
        cmd_buf.memory_barrier();
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
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k3_pipeline,
            &k3_ds,
            ((WIDTH / 512) as u32, (HEIGHT / 16) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 3);
        cmd_buf.memory_barrier();
        cmd_buf.dispatch(
            &k4_pipeline,
            &k4_ds,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 4);
        cmd_buf.memory_barrier();
        cmd_buf.copy_buffer(&image_dev, &image_buf);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let timestamps = device.reap_query_pool(query_pool).unwrap();
        println!("Kernel 1 time: {:.3}ms", timestamps[0] * 1e3);
        println!(
            "Kernel 2 time: {:.3}ms",
            (timestamps[1] - timestamps[0]) * 1e3
        );
        println!(
            "Kernel 3 time: {:.3}ms",
            (timestamps[2] - timestamps[1]) * 1e3
        );
        println!(
            "Render time: {:.3}ms",
            (timestamps[3] - timestamps[2]) * 1e3
        );

        /*
        let mut k1_data: Vec<u32> = Default::default();
        device.read_buffer(&segment_buf, &mut k1_data).unwrap();
        dump_k1_data(&k1_data);
        */

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
}
