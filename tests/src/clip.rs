// Copyright 2022 The piet-gpu authors.
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

//! Tests for the piet-gpu clip processing stage.

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use piet_gpu::stages::{self, ClipBinding, ClipCode, DrawMonoid};
use piet_gpu_hal::{BufWrite, BufferUsage};

use crate::{Config, Runner, TestResult};

struct ClipData {
    clip_stream: Vec<u32>,
    // In the atomic-int friendly encoding
    path_bbox_stream: Vec<PathBbox>,
}

#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
#[repr(C)]
struct PathBbox {
    bbox: [u32; 4],
    linewidth: f32,
    trans_ix: u32,
}

pub unsafe fn clip_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("clip");
    let n_clip: u64 = config.size.choose(1 << 8, 1 << 12, 1 << 16);
    let data = ClipData::new(n_clip);
    let stage_config = data.get_config();
    let config_buf = runner
        .session
        .create_buffer_init(std::slice::from_ref(&stage_config), BufferUsage::STORAGE)
        .unwrap();
    // Need to actually get data uploaded
    let mut memory = runner.buf_down(data.memory_size(), BufferUsage::STORAGE);
    {
        let mut buf_write = memory.map_write(..);
        data.fill_memory(&mut buf_write);
    }

    let code = ClipCode::new(&runner.session);
    let binding = ClipBinding::new(&runner.session, &code, &config_buf, &memory.dev_buf);

    let mut commands = runner.commands();
    commands.upload(&memory);
    let mut pass = commands.compute_pass(0, 1);
    binding.record(&mut pass, &code, n_clip as u32);
    pass.end();
    commands.download(&memory);
    runner.submit(commands);
    let dst = memory.map_read(..);
    if let Some(failure) = data.verify(&dst) {
        result.fail(failure);
    }
    result
}

fn rand_bbox() -> [u32; 4] {
    let mut rng = rand::thread_rng();
    const Y_MIN: u32 = 32768;
    const Y_MAX: u32 = Y_MIN + 1000;
    let mut x0 = rng.gen_range(Y_MIN, Y_MAX);
    let mut y0 = rng.gen_range(Y_MIN, Y_MAX);
    let mut x1 = rng.gen_range(Y_MIN, Y_MAX);
    let mut y1 = rng.gen_range(Y_MIN, Y_MAX);
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
    }
    if y0 > y1 {
        std::mem::swap(&mut y0, &mut y1);
    }
    [x0, y0, x1, y1]
}

/// Convert from atomic-friendly to normal float bbox.
fn decode_bbox(raw: [u32; 4]) -> [f32; 4] {
    fn decode(x: u32) -> f32 {
        x as f32 - 32768.0
    }
    [
        decode(raw[0]),
        decode(raw[1]),
        decode(raw[2]),
        decode(raw[3]),
    ]
}

fn intersect_bbox(b0: [f32; 4], b1: [f32; 4]) -> [f32; 4] {
    [
        b0[0].max(b1[0]),
        b0[1].max(b1[1]),
        b0[2].min(b1[2]),
        b0[3].min(b1[3]),
    ]
}

const INFTY_BBOX: [f32; 4] = [-1e9, -1e9, 1e9, 1e9];

impl ClipData {
    /// Generate a random clip sequence
    fn new(n: u64) -> ClipData {
        // Simple LCG random generator, for deterministic results
        let mut z = 20170705u64;
        let mut depth = 0;
        let mut path_bbox_stream = Vec::new();
        let clip_stream = (0..n)
            .map(|i| {
                let is_push = if depth == 0 {
                    true
                } else if depth >= 255 {
                    false
                } else {
                    z = z.wrapping_mul(742938285) % ((1 << 31) - 1);
                    (z % 2) != 0
                };
                if is_push {
                    depth += 1;
                    let path_ix = path_bbox_stream.len() as u32;
                    let bbox = rand_bbox();
                    let path_bbox = PathBbox {
                        bbox,
                        ..Default::default()
                    };
                    path_bbox_stream.push(path_bbox);
                    path_ix
                } else {
                    depth -= 1;
                    !(i as u32)
                }
            })
            .collect();
        ClipData {
            clip_stream,
            path_bbox_stream,
        }
    }

    fn get_config(&self) -> stages::Config {
        let n_clip = self.clip_stream.len();
        let n_path = self.path_bbox_stream.len();
        let clip_alloc = 0;
        let path_bbox_alloc = clip_alloc + 4 * n_clip;
        let drawmonoid_alloc = path_bbox_alloc + 24 * n_path;
        let clip_bic_alloc = drawmonoid_alloc + 8 * n_clip;
        // TODO: this is over-allocated, we only need one bic per wg
        let clip_stack_alloc = clip_bic_alloc + 8 * n_clip;
        let clip_bbox_alloc = clip_stack_alloc + 20 * n_clip;
        stages::Config {
            clip_alloc: clip_alloc as u32,
            path_bbox_alloc: path_bbox_alloc as u32,
            drawmonoid_alloc: drawmonoid_alloc as u32,
            clip_bic_alloc: clip_bic_alloc as u32,
            clip_stack_alloc: clip_stack_alloc as u32,
            clip_bbox_alloc: clip_bbox_alloc as u32,
            n_clip: n_clip as u32,
            ..Default::default()
        }
    }

    fn memory_size(&self) -> u64 {
        (8 + self.clip_stream.len() * (4 + 8 + 8 + 20 + 16) + self.path_bbox_stream.len() * 24)
            as u64
    }

    fn fill_memory(&self, buf: &mut BufWrite) {
        // offset / header; no dynamic allocation
        buf.fill_zero(8);
        buf.extend_slice(&self.clip_stream);
        buf.extend_slice(&self.path_bbox_stream);
        // drawmonoid is left uninitialized
    }

    fn verify(&self, buf: &[u8]) -> Option<String> {
        let n_clip = self.clip_stream.len();
        let n_path = self.path_bbox_stream.len();
        let clip_bbox_start = 8 + n_clip * (4 + 8 + 8 + 20) + n_path * 24;
        let clip_range = clip_bbox_start..(clip_bbox_start + n_clip * 16);
        let clip_result = bytemuck::cast_slice::<u8, [f32; 4]>(&buf[clip_range]);
        let draw_start = 8 + n_clip * 4 + n_path * 24;
        let draw_range = draw_start..(draw_start + n_clip * 16);
        let draw_result = bytemuck::cast_slice::<u8, DrawMonoid>(&buf[draw_range]);
        let mut bbox_stack = Vec::new();
        let mut parent_stack = Vec::new();
        for (i, path_ix) in self.clip_stream.iter().enumerate() {
            let mut expected_path = None;
            if *path_ix >= 0x8000_0000 {
                let parent = parent_stack.pop().unwrap();
                expected_path = Some(self.clip_stream[parent as usize]);
                bbox_stack.pop().unwrap();
            } else {
                parent_stack.push(i);
                let path_bbox_stream = self.path_bbox_stream[*path_ix as usize];
                let bbox = decode_bbox(path_bbox_stream.bbox);
                let new = match bbox_stack.last() {
                    None => bbox,
                    Some(old) => intersect_bbox(*old, bbox),
                };
                bbox_stack.push(new);
            };
            let expected = bbox_stack.last().copied().unwrap_or(INFTY_BBOX);
            let clip_bbox = clip_result[i];
            if clip_bbox != expected {
                return Some(format!(
                    "{}: path_ix={}, expected bbox={:?}, clip_bbox={:?}",
                    i, path_ix, expected, clip_bbox
                ));
            }
            if let Some(expected_path) = expected_path {
                let actual_path = draw_result[i].path_ix;
                if expected_path != actual_path {
                    return Some(format!(
                        "{}: expected path {}, actual {}",
                        i, expected_path, actual_path
                    ));
                }
            }
        }
        None
    }
}
