// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! A simple application to run a compute shader.

use std::{fs::File, io::BufWriter};

use engine::Engine;

use wgpu::{Device, Limits, Queue};

mod engine;
mod pico_svg;
mod render;
mod shaders;
mod test_scene;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let mut limits = Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & wgpu::Features::TIMESTAMP_QUERY,
                limits,
            },
            None,
        )
        .await?;
    let mut engine = Engine::new();
    do_render(&device, &queue, &mut engine).await?;

    Ok(())
}

fn dump_buf(buf: &[u32]) {
    for (i, val) in buf.iter().enumerate() {
        if *val != 0 {
            let lo = val & 0x7fff_ffff;
            if lo >= 0x3000_0000 && lo < 0x5000_0000 {
                println!("{}: {:x} {}", i, val, f32::from_bits(*val));
            } else {
                println!("{}: {:x}", i, val);

            }
        }
    }
}

async fn do_render(
    device: &Device,
    queue: &Queue,
    engine: &mut Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    #[allow(unused)]
    let shaders = shaders::init_shaders(device, engine)?;
    let full_shaders = shaders::full_shaders(device, engine)?;
    let scene = test_scene::gen_test_scene();
    //test_scene::dump_scene_info(&scene);
    //let (recording, buf) = render::render(&scene, &shaders);
    let (recording, buf) = render::render_full(&scene, &full_shaders);
    let downloads = engine.run_recording(&device, &queue, &recording)?;
    let mapped = downloads.map();
    device.poll(wgpu::Maintain::Wait);
    let buf = mapped.get_mapped(buf).await?;

    if false {
        dump_buf(bytemuck::cast_slice(&buf));
    } else {
        let file = File::create("image.png")?;
        let w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, 1024, 1024);
        encoder.set_color(png::ColorType::Rgba);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&buf)?;
    }
    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
