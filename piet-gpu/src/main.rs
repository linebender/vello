use std::path::Path;
use std::fs::File;
use std::io::BufWriter;

use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, MemFlags};

const WIDTH: usize = 2048;
const HEIGHT: usize = 1536;

const TILE_W: usize = 16;
const TILE_H: usize = 16;

fn main() {
    let instance = VkInstance::new().unwrap();
    unsafe {
        let device = instance.device().unwrap();
        let mem_flags = MemFlags::host_coherent();
        let src = (0..256).map(|x| x + 1).collect::<Vec<u32>>();
        let scene_buf = device
            .create_buffer(std::mem::size_of_val(&src[..]) as u64, mem_flags)
            .unwrap();
        device.write_buffer(&scene_buf, &src).unwrap();
        let image_buf = device
            .create_buffer((WIDTH * HEIGHT * 4) as u64, mem_flags)
            .unwrap();
        let code = include_bytes!("../shader/image.spv");
        let pipeline = device.create_simple_compute_pipeline(code, 2).unwrap();
        let descriptor_set = device
            .create_descriptor_set(&pipeline, &[&scene_buf, &image_buf])
            .unwrap();
        let query_pool = device.create_query_pool(2).unwrap();
        let mut cmd_buf = device.create_cmd_buf().unwrap();
        cmd_buf.begin();
        cmd_buf.write_timestamp(&query_pool, 0);
        cmd_buf.dispatch(
            &pipeline,
            &descriptor_set,
            ((WIDTH / TILE_W) as u32, (HEIGHT / TILE_H) as u32, 1),
        );
        cmd_buf.write_timestamp(&query_pool, 1);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf).unwrap();
        let timestamps = device.reap_query_pool(query_pool).unwrap();
        println!("Render time: {:.3}ms", timestamps[0] * 1e3);
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
