use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use rand::{Rng, RngCore};

use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

use piet_gpu_types::encoder::{Encode, Encoder};
use piet_gpu_types::scene::{Bbox, PietCircle, PietItem, Point, SimpleGroup};

const WIDTH: usize = 2048;
const HEIGHT: usize = 1536;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

const N_CIRCLES: usize = 100;

fn make_scene() -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut encoder = Encoder::new();
    let _reserve_root = encoder.alloc_chunk(PietItem::fixed_size() as u32);

    let mut items = Vec::new();
    let mut bboxes = Vec::new();
    for _ in 0..N_CIRCLES {
        let circle = PietCircle {
            rgba_color: rng.next_u32(),
            center: Point {
                xy: [
                    rng.gen_range(0.0, WIDTH as f32),
                    rng.gen_range(0.0, HEIGHT as f32),
                ],
            },
            radius: rng.gen_range(0.0, 50.0),
        };
        let bbox = Bbox {
            bbox: [
                (circle.center.xy[0] - circle.radius).floor() as i16,
                (circle.center.xy[1] - circle.radius).floor() as i16,
                (circle.center.xy[0] + circle.radius).ceil() as i16,
                (circle.center.xy[1] + circle.radius).ceil() as i16,
            ],
        };
        items.push(PietItem::Circle(circle));
        bboxes.push(bbox);
    }

    let n_items = bboxes.len() as u32;
    let bboxes = bboxes.encode(&mut encoder).transmute();
    let items = items.encode(&mut encoder).transmute();
    let offset = Point { xy: [0.0, 0.0] };
    let simple_group = SimpleGroup {
        n_items,
        bboxes,
        items,
        offset,
    };
    let root_item = PietItem::Group(simple_group);
    root_item.encode_to(&mut encoder.buf_mut()[0..PietItem::fixed_size()]);
    // We should avoid this clone.
    encoder.buf().to_owned()
}

#[allow(unused)]
fn dump_scene(buf: &[u8]) {
    for i in 0..(buf.len() / 4) {
        let mut buf_u32 = [0u8; 4];
        buf_u32.copy_from_slice(&buf[i * 4..i * 4 + 4]);
        println!("{:4x}: {:8x}", i * 4, u32::from_le_bytes(buf_u32));
    }
}

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
        let scene = make_scene();
        //dump_scene(&scene);
        let scene_buf = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, host)
            .unwrap();
        let scene_dev = device
            .create_buffer(std::mem::size_of_val(&scene[..]) as u64, dev)
            .unwrap();
        device.write_buffer(&scene_buf, &scene).unwrap();
        let tilegroup_buf = device.create_buffer(384 * 1024, host).unwrap();
        let image_buf = device
            .create_buffer((WIDTH * HEIGHT * 4) as u64, host)
            .unwrap();
        let image_dev = device
            .create_buffer((WIDTH * HEIGHT * 4) as u64, dev)
            .unwrap();

        let k1_code = include_bytes!("../shader/kernel1.spv");
        let k1_pipeline = device.create_simple_compute_pipeline(k1_code, 2).unwrap();
        let k1_ds = device
            .create_descriptor_set(&k1_pipeline, &[&scene_dev, &tilegroup_buf])
            .unwrap();

        let code = include_bytes!("../shader/image.spv");
        let pipeline = device.create_simple_compute_pipeline(code, 2).unwrap();
        let descriptor_set = device
            .create_descriptor_set(&pipeline, &[&scene_dev, &image_dev])
            .unwrap();
        let query_pool = device.create_query_pool(3).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.copy_buffer(&scene_buf, &scene_dev);
        cmd_buf.clear_buffer(&tilegroup_buf);
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
            &pipeline,
            &descriptor_set,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 2);
        cmd_buf.memory_barrier();
        cmd_buf.copy_buffer(&image_dev, &image_buf);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let timestamps = device.reap_query_pool(query_pool).unwrap();
        println!("Kernel 1 time: {:.3}ms", timestamps[0] * 1e3);
        println!("Render time: {:.3}ms", (timestamps[1] - timestamps[0]) * 1e3);

        let mut k1_data: Vec<u32> = Default::default();
        device.read_buffer(&tilegroup_buf, &mut k1_data).unwrap();
        dump_k1_data(&k1_data);

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
