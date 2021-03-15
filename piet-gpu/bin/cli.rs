use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use clap::{App, Arg};

use piet_gpu_hal::hub;
use piet_gpu_hal::vulkan::VkInstance;
use piet_gpu_hal::{CmdBuf, Error, MemFlags};

use piet_gpu::{render_scene, render_svg, PietGpuRenderContext, Renderer, HEIGHT, WIDTH};

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
        let floats = (0..11)
            .map(|k| {
                let mut buf_f32 = [0u8; 4];
                buf_f32.copy_from_slice(&buf[j + k * 4..j + k * 4 + 4]);
                f32::from_le_bytes(buf_f32)
            })
            .collect::<Vec<_>>();
        println!(
            "{}: [{} {} {} {} {} {}] ({}, {})-({} {}) {} {}",
            i,
            floats[0],
            floats[1],
            floats[2],
            floats[3],
            floats[4],
            floats[5],
            floats[6],
            floats[7],
            floats[8],
            floats[9],
            floats[10],
            buf[j + 44]
        );
    }
}

/// Interpret the output of the binning stage, for diagnostic purposes.
#[allow(unused)]
fn trace_merge(buf: &[u32]) {
    for bin in 0..256 {
        println!("bin {}:", bin);
        let mut starts = (0..16)
            .map(|i| Some((bin * 16 + i) * 64))
            .collect::<Vec<Option<usize>>>();
        loop {
            let min_start = starts
                .iter()
                .map(|st| {
                    st.map(|st| {
                        if buf[st / 4] == 0 {
                            !0
                        } else {
                            buf[st / 4 + 2]
                        }
                    })
                    .unwrap_or(!0)
                })
                .min()
                .unwrap();
            if min_start == !0 {
                break;
            }
            let mut selected = !0;
            for i in 0..16 {
                if let Some(st) = starts[i] {
                    if buf[st / 4] != 0 && buf[st / 4 + 2] == min_start {
                        selected = i;
                        break;
                    }
                }
            }
            let st = starts[selected].unwrap();
            println!("selected {}, start {:x}", selected, st);
            for j in 0..buf[st / 4] {
                println!("{:x}", buf[st / 4 + 2 + j as usize])
            }
            if buf[st / 4 + 1] == 0 {
                starts[selected] = None;
            } else {
                starts[selected] = Some(buf[st / 4 + 1] as usize);
            }
        }
    }
}

/// Interpret the output of the coarse raster stage, for diagnostic purposes.
#[allow(unused)]
fn trace_ptcl(buf: &[u32]) {
    for y in 0..96 {
        for x in 0..128 {
            let tile_ix = y * 128 + x;
            println!("tile {} @({}, {})", tile_ix, x, y);
            let mut tile_offset = tile_ix * 1024;
            loop {
                let tag = buf[tile_offset / 4];
                match tag {
                    0 => break,
                    3 => {
                        let backdrop = buf[tile_offset / 4 + 2];
                        let rgba_color = buf[tile_offset / 4 + 3];
                        println!("  {:x}: fill {:x} {}", tile_offset, rgba_color, backdrop);
                        let mut seg_chunk = buf[tile_offset / 4 + 1] as usize;
                        let n = buf[seg_chunk / 4] as usize;
                        let segs = buf[seg_chunk / 4 + 2] as usize;
                        println!("    chunk @{:x}: n={}, segs @{:x}", seg_chunk, n, segs);
                        for i in 0..n {
                            let x0 = f32::from_bits(buf[segs / 4 + i * 5]);
                            let y0 = f32::from_bits(buf[segs / 4 + i * 5 + 1]);
                            let x1 = f32::from_bits(buf[segs / 4 + i * 5 + 2]);
                            let y1 = f32::from_bits(buf[segs / 4 + i * 5 + 3]);
                            let y_edge = f32::from_bits(buf[segs / 4 + i * 5 + 4]);
                            println!(
                                "      ({:.3}, {:.3}) - ({:.3}, {:.3}) | {:.3}",
                                x0, y0, x1, y1, y_edge
                            );
                        }
                        loop {
                            seg_chunk = buf[seg_chunk / 4 + 1] as usize;
                            if seg_chunk == 0 {
                                break;
                            }
                        }
                    }
                    4 => {
                        let line_width = f32::from_bits(buf[tile_offset / 4 + 2]);
                        let rgba_color = buf[tile_offset / 4 + 3];
                        println!(
                            "  {:x}: stroke {:x} {}",
                            tile_offset, rgba_color, line_width
                        );
                        let mut seg_chunk = buf[tile_offset / 4 + 1] as usize;
                        let n = buf[seg_chunk / 4] as usize;
                        let segs = buf[seg_chunk / 4 + 2] as usize;
                        println!("    chunk @{:x}: n={}, segs @{:x}", seg_chunk, n, segs);
                        for i in 0..n {
                            let x0 = f32::from_bits(buf[segs / 4 + i * 5]);
                            let y0 = f32::from_bits(buf[segs / 4 + i * 5 + 1]);
                            let x1 = f32::from_bits(buf[segs / 4 + i * 5 + 2]);
                            let y1 = f32::from_bits(buf[segs / 4 + i * 5 + 3]);
                            let y_edge = f32::from_bits(buf[segs / 4 + i * 5 + 4]);
                            println!(
                                "      ({:.3}, {:.3}) - ({:.3}, {:.3}) | {:.3}",
                                x0, y0, x1, y1, y_edge
                            );
                        }
                        loop {
                            seg_chunk = buf[seg_chunk / 4 + 1] as usize;
                            if seg_chunk == 0 {
                                break;
                            }
                        }
                    }
                    6 => {
                        let backdrop = buf[tile_offset / 4 + 2];
                        println!("  {:x}: begin_clip {}", tile_offset, backdrop);
                        let mut seg_chunk = buf[tile_offset / 4 + 1] as usize;
                        let n = buf[seg_chunk / 4] as usize;
                        let segs = buf[seg_chunk / 4 + 2] as usize;
                        println!("    chunk @{:x}: n={}, segs @{:x}", seg_chunk, n, segs);
                        for i in 0..n {
                            let x0 = f32::from_bits(buf[segs / 4 + i * 5]);
                            let y0 = f32::from_bits(buf[segs / 4 + i * 5 + 1]);
                            let x1 = f32::from_bits(buf[segs / 4 + i * 5 + 2]);
                            let y1 = f32::from_bits(buf[segs / 4 + i * 5 + 3]);
                            let y_edge = f32::from_bits(buf[segs / 4 + i * 5 + 4]);
                            println!(
                                "      ({:.3}, {:.3}) - ({:.3}, {:.3}) | {:.3}",
                                x0, y0, x1, y1, y_edge
                            );
                        }
                        loop {
                            seg_chunk = buf[seg_chunk / 4 + 1] as usize;
                            if seg_chunk == 0 {
                                break;
                            }
                        }
                    }
                    7 => {
                        let backdrop = buf[tile_offset / 4 + 1];
                        println!("{:x}: solid_clip {:x}", tile_offset, backdrop);
                    }
                    8 => {
                        println!("{:x}: end_clip", tile_offset);
                    }
                    _ => {
                        println!("{:x}: {}", tile_offset, tag);
                    }
                }
                if tag == 0 {
                    break;
                }
                if tag == 8 {
                    tile_offset = buf[tile_offset / 4 + 1] as usize;
                } else {
                    tile_offset += 20;
                }
            }
        }
    }
}

fn main() -> Result<(), Error> {
    let matches = App::new("piet-gpu test")
        .arg(Arg::with_name("INPUT").index(1))
        .arg(Arg::with_name("flip").short("f").long("flip"))
        .arg(
            Arg::with_name("scale")
                .short("s")
                .long("scale")
                .takes_value(true),
        )
        .get_matches();
    let (instance, _) = VkInstance::new(None)?;
    unsafe {
        let device = instance.device(None)?;
        let session = hub::Session::new(device);

        let mut cmd_buf = session.cmd_buf()?;
        let query_pool = session.create_query_pool(8)?;

        let mut ctx = PietGpuRenderContext::new();
        if let Some(input) = matches.value_of("INPUT") {
            let mut scale = matches
                .value_of("scale")
                .map(|scale| scale.parse().unwrap())
                .unwrap_or(8.0);
            if matches.is_present("flip") {
                scale = -scale;
            }
            render_svg(&mut ctx, input, scale);
        } else {
            render_scene(&mut ctx);
        }
        let n_paths = ctx.path_count();
        let n_pathseg = ctx.pathseg_count();
        let n_trans = ctx.trans_count();
        let scene = ctx.get_scene_buf();
        //dump_scene(&scene);

        let renderer = Renderer::new(&session, scene, n_paths, n_pathseg, n_trans)?;
        let image_buf =
            session.create_buffer((WIDTH * HEIGHT * 4) as u64, MemFlags::host_coherent())?;

        cmd_buf.begin();
        renderer.record(&mut cmd_buf, &query_pool);
        cmd_buf.copy_image_to_buffer(renderer.image_dev.vk_image(), image_buf.vk_buffer());
        cmd_buf.host_barrier();
        cmd_buf.finish();
        let start = std::time::Instant::now();
        let submitted = session.run_cmd_buf(cmd_buf, &[], &[])?;
        submitted.wait()?;
        println!("elapsed = {:?}", start.elapsed());
        let ts = session.fetch_query_pool(&query_pool).unwrap();
        println!("Element kernel time: {:.3}ms", ts[0] * 1e3);
        println!(
            "Tile allocation kernel time: {:.3}ms",
            (ts[1] - ts[0]) * 1e3
        );
        println!("Coarse path kernel time: {:.3}ms", (ts[2] - ts[1]) * 1e3);
        println!("Backdrop kernel time: {:.3}ms", (ts[3] - ts[2]) * 1e3);
        println!("Binning kernel time: {:.3}ms", (ts[4] - ts[3]) * 1e3);
        println!("Coarse raster kernel time: {:.3}ms", (ts[5] - ts[4]) * 1e3);
        println!("Render kernel time: {:.3}ms", (ts[6] - ts[5]) * 1e3);

        /*
        let mut data: Vec<u32> = Default::default();
        renderer.tile_buf.read(&mut data).unwrap();
        piet_gpu::dump_k1_data(&data);
        trace_ptcl(&data);
        */

        let mut img_data: Vec<u8> = Default::default();
        // Note: because png can use a `&[u8]` slice, we could avoid an extra copy
        // (probably passing a slice into a closure). But for now: keep it simple.
        image_buf.read(&mut img_data).unwrap();

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
