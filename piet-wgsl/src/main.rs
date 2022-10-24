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

use bytemuck;
use render::render;
use test_scene::dump_scene_info;
use wgpu::{Device, Queue};

mod engine;
mod render;
mod shaders;
mod template;
mod test_scene;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & wgpu::Features::TIMESTAMP_QUERY,
                limits: Default::default(),
            },
            None,
        )
        .await?;
    let mut engine = Engine::new();
    do_render(&device, &queue, &mut engine).await?;

    Ok(())
}

async fn do_render(
    device: &Device,
    queue: &Queue,
    engine: &mut Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    let shaders = shaders::init_shaders(device, engine)?;
    let scene = test_scene::gen_test_scene();
    dump_scene_info(&scene);
    let (recording, buf) = render(&scene, &shaders);
    let downloads = engine.run_recording(&device, &queue, &recording)?;
    let mapped = downloads.map();
    device.poll(wgpu::Maintain::Wait);
    let buf = mapped.get_mapped(buf).await?;

    let file = File::create("image.png")?;
    let w = BufWriter::new(file);
    let encoder = png::Encoder::new(w, 1024, 1024);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&buf)?;
    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
