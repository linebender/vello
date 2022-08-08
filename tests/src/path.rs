// Copyright 2021 The piet-gpu authors.
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

//! Tests for the piet-gpu path stage.

use crate::{Config, Runner, TestResult};

use bytemuck::{Pod, Zeroable};
use piet_gpu::stages::{self, PathCode, PathEncoder, PathStage, Transform};
use piet_gpu_hal::{BufWrite, BufferUsage};
use rand::{prelude::ThreadRng, Rng};

struct PathData {
    n_trans: u32,
    n_linewidth: u32,
    n_path: u32,
    n_pathseg: u32,
    tags: Vec<u8>,
    pathsegs: Vec<u8>,
    bbox: Vec<(f32, f32, f32, f32)>,
    lines: Vec<([f32; 2], [f32; 2])>,
}

// This is designed to match pathseg.h
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct PathSeg {
    tag: u32,
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    path_ix: u32,
    trans_ix: u32,
    stroke: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Zeroable, Pod)]
struct Bbox {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    linewidth: f32,
    trans_ix: u32,
}

pub unsafe fn path_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("path");

    // TODO: implement large scans and raise limit
    let n_path: u64 = config.size.choose(1 << 12, 1 << 16, 209_000);
    let path_data = PathData::new(n_path as u32);
    let stage_config = path_data.get_config();
    let config_buf = runner
        .session
        .create_buffer_init(std::slice::from_ref(&stage_config), BufferUsage::STORAGE)
        .unwrap();
    let scene_size = n_path * 256;
    let scene_buf = runner
        .session
        .create_buffer_with(
            scene_size,
            |b| path_data.fill_scene(b),
            BufferUsage::STORAGE,
        )
        .unwrap();
    let memory_init = runner
        .session
        .create_buffer_with(
            path_data.memory_init_size(),
            |b| path_data.fill_memory(b),
            BufferUsage::COPY_SRC,
        )
        .unwrap();
    let memory = runner.buf_down(path_data.memory_full_size(), BufferUsage::empty());

    let code = PathCode::new(&runner.session);
    let stage = PathStage::new(&runner.session, &code);
    let binding = stage.bind(
        &runner.session,
        &code,
        &config_buf,
        &scene_buf,
        &memory.dev_buf,
    );

    let mut total_elapsed = 0.0;
    let n_iter = config.n_iter;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        commands.cmd_buf.copy_buffer(&memory_init, &memory.dev_buf);
        commands.cmd_buf.memory_barrier();
        let mut pass = commands.compute_pass(0, 1);
        stage.record(
            &mut pass,
            &code,
            &binding,
            path_data.n_path,
            path_data.tags.len() as u32,
        );
        pass.end();
        if i == 0 || config.verify_all {
            commands.cmd_buf.memory_barrier();
            commands.download(&memory);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 || config.verify_all {
            let dst = memory.map_read(..);
            if let Some(failure) = path_data.verify(&dst) {
                result.fail(failure);
            }
        }
    }
    let n_elements = path_data.n_pathseg as u64;
    result.timing(total_elapsed, n_elements * n_iter);

    result
}

fn rand_point(rng: &mut ThreadRng) -> (f32, f32) {
    let x = rng.gen_range(0.0, 100.0);
    let y = rng.gen_range(0.0, 100.0);
    (x, y)
}

// Must match shader/pathseg.h
const PATHSEG_SIZE: u32 = 52;

impl PathData {
    fn new(n_path: u32) -> PathData {
        let mut rng = rand::thread_rng();
        let n_trans = 1;
        let n_linewidth = 1;
        let segments_per_path = 8;
        let mut tags = Vec::new();
        let mut pathsegs = Vec::new();
        let mut bbox = Vec::new();
        let mut lines = Vec::new();
        let mut encoder = PathEncoder::new(&mut tags, &mut pathsegs);
        for _ in 0..n_path {
            let (x, y) = rand_point(&mut rng);
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;
            let first_pt = [x, y];
            let mut last_pt = [x, y];
            encoder.move_to(x, y);
            for _ in 0..segments_per_path {
                let (x, y) = rand_point(&mut rng);
                lines.push((last_pt, [x, y]));
                last_pt = [x, y];
                encoder.line_to(x, y);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            bbox.push((min_x, min_y, max_x, max_y));
            encoder.close_path();
            // With very low probability last_pt and first_pt might be equal, which
            // would cause a test failure - might want to seed RNG.
            lines.push((last_pt, first_pt));
            encoder.path();
        }
        let n_pathseg = encoder.n_pathseg();
        //println!("tags: {:x?}", &tags[0..8]);
        //println!("path: {:?}", bytemuck::cast_slice::<u8, f32>(&pathsegs[0..64]));
        PathData {
            n_trans,
            n_linewidth,
            n_path,
            n_pathseg,
            tags,
            pathsegs,
            bbox,
            lines,
        }
    }

    fn get_config(&self) -> stages::Config {
        let n_trans = self.n_trans;

        // Layout of scene buffer
        let linewidth_offset = 0;
        let pathtag_offset = linewidth_offset + self.n_linewidth * 4;
        let n_tagbytes = self.tags.len() as u32;
        // Depends on workgroup size, maybe get from stages?
        let padded_n_tagbytes = (n_tagbytes + 2047) & !2047;
        let pathseg_offset = pathtag_offset + padded_n_tagbytes;

        // Layout of memory
        let trans_alloc = 0;
        let pathseg_alloc = trans_alloc + n_trans * 24;
        let path_bbox_alloc = pathseg_alloc + self.n_pathseg * PATHSEG_SIZE;
        let stage_config = stages::Config {
            pathseg_alloc,
            path_bbox_alloc,
            n_trans,
            n_path: self.n_path,
            pathtag_offset,
            linewidth_offset,
            pathseg_offset,
            ..Default::default()
        };
        stage_config
    }

    fn fill_scene(&self, buf: &mut BufWrite) {
        let linewidth = -1.0f32;
        buf.push(linewidth);
        buf.extend_slice(&self.tags);
        buf.fill_zero(self.tags.len().wrapping_neg() & 2047);
        buf.extend_slice(&self.pathsegs);
    }

    fn memory_init_size(&self) -> u64 {
        let mut size = 8; // offset and error
        size += self.n_trans * 24;
        size as u64
    }

    fn memory_full_size(&self) -> u64 {
        let mut size = self.memory_init_size();
        size += (self.n_pathseg * PATHSEG_SIZE) as u64;
        size += (self.n_path * 24) as u64;
        size
    }

    fn fill_memory(&self, buf: &mut BufWrite) {
        // This stage is not dynamically allocating memory
        let mem_offset = 0u32;
        let mem_error = 0u32;
        let mem_init = [mem_offset, mem_error];
        buf.push(mem_init);
        let trans = Transform::IDENTITY;
        buf.push(trans);
    }

    fn verify(&self, memory: &[u8]) -> Option<String> {
        fn round_down(x: f32) -> u32 {
            (x.floor() + 32768.0) as u32
        }
        fn round_up(x: f32) -> u32 {
            (x.ceil() + 32768.0) as u32
        }
        let begin_pathseg = 32;
        for i in 0..self.n_pathseg {
            let offset = (begin_pathseg + PATHSEG_SIZE * i) as usize;
            let actual =
                bytemuck::from_bytes::<PathSeg>(&memory[offset..offset + PATHSEG_SIZE as usize]);
            let expected = self.lines[i as usize];
            const EPSILON: f32 = 1e-9;
            if (expected.0[0] - actual.p0[0]).abs() > EPSILON
                || (expected.0[1] - actual.p0[1]).abs() > EPSILON
                || (expected.1[0] - actual.p3[0]).abs() > EPSILON
                || (expected.1[1] - actual.p3[1]).abs() > EPSILON
            {
                println!("{}: {:.1?} {:.1?}", i, actual, expected);
            }
        }
        let begin_bbox = 32 + PATHSEG_SIZE * self.n_pathseg;
        for i in 0..self.n_path {
            let offset = (begin_bbox + 24 * i) as usize;
            let actual = bytemuck::from_bytes::<Bbox>(&memory[offset..offset + 24]);
            let expected_f32 = self.bbox[i as usize];
            if round_down(expected_f32.0) != actual.left
                || round_down(expected_f32.1) != actual.top
                || round_up(expected_f32.2) != actual.right
                || round_up(expected_f32.3) != actual.bottom
            {
                println!("{}: {:?} {:?}", i, actual, expected_f32);
                return Some(format!("bbox mismatch at {}", i));
            }
        }
        None
    }
}
