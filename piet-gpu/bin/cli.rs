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

fn main() -> Result<(), Error> {
    let (instance, _) = VkInstance::new(None)?;
    unsafe {
        let device = instance.device(None)?;

        let fence = device.create_fence(false)?;
        let mut cmd_buf = device.create_cmd_buf()?;
        let query_pool = device.create_query_pool(6)?;

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
