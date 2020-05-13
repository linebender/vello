use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Device, Error, MemFlags};

use piet_gpu::{render_scene, PietGpuRenderContext, Renderer, HEIGHT, WIDTH};

#[allow(unused)]
fn dump_scene(buf: &[u8]) {
    for i in 0..(buf.len() / 4) {
        let mut buf_u32 = [0u8; 4];
        buf_u32.copy_from_slice(&buf[i * 4..i * 4 + 4]);
        println!("{:4x}: {:8x}", i * 4, u32::from_le_bytes(buf_u32));
    }
}

#[allow(unused)]
fn dump_state(buf: &[u8]) {
    for i in 0..(buf.len() / 48) {
        let j = i * 48;
        let floats = (0..11).map(|k| {
            let mut buf_f32 = [0u8; 4];
            buf_f32.copy_from_slice(&buf[j + k * 4..j + k * 4 + 4]);
            f32::from_le_bytes(buf_f32)
        }).collect::<Vec<_>>();
        println!("{}: [{} {} {} {} {} {}] ({}, {})-({} {}) {} {}",
            i,
            floats[0], floats[1], floats[2], floats[3], floats[4], floats[5],
            floats[6], floats[7], floats[8], floats[9],
            floats[10], buf[j + 44]);
    }

}

fn main() -> Result<(), Error> {
    let (instance, _) = VkInstance::new(None)?;
    unsafe {
        let device = instance.device(None)?;

        let fence = device.create_fence(false)?;
        let mut cmd_buf = device.create_cmd_buf()?;
        let query_pool = device.create_query_pool(4)?;

        let mut ctx = PietGpuRenderContext::new();
        render_scene(&mut ctx);
        let scene = ctx.get_scene_buf();
        //dump_scene(&scene);

        let renderer = Renderer::new(&device, scene)?;
        let image_buf =
            device.create_buffer((WIDTH * HEIGHT * 4) as u64, MemFlags::host_coherent())?;

        cmd_buf.begin();
        renderer.record(&mut cmd_buf, &query_pool);
        cmd_buf.copy_image_to_buffer(&renderer.image_dev, &image_buf);
        cmd_buf.finish();
        device.run_cmd_buf(&cmd_buf, &[], &[], Some(&fence))?;
        device.wait_and_reset(&[fence])?;
        let ts = device.reap_query_pool(&query_pool).unwrap();
        println!("Element kernel time: {:.3}ms", ts[0] * 1e3);
        println!("Binning kernel time: {:.3}ms", (ts[1] - ts[0]) * 1e3);
        println!("Coarse kernel time: {:.3}ms", (ts[2] - ts[1]) * 1e3);

        /*
        let mut data: Vec<u32> = Default::default();
        device.read_buffer(&renderer.bin_buf, &mut data).unwrap();
        piet_gpu::dump_k1_data(&data);

        let mut data: Vec<u32> = Default::default();
        device.read_buffer(&renderer.ptcl_buf, &mut data).unwrap();
        piet_gpu::dump_k1_data(&data);
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

    Ok(())
}
