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

//! Tests for the piet-gpu draw object stage.

use piet_gpu_hal::{BufWrite, BufferUsage};
use rand::{seq::SliceRandom, Rng};

use crate::{Config, Runner, TestResult};

use piet_gpu::stages::{self, DrawCode, DrawMonoid, DrawStage};

const DRAWTAG_SIZE: usize = 4;
const ANNOTATED_SIZE: usize = 40;

// Tags for draw objects. See shader/drawtag.h for the authoritative source.
const DRAWTAG_FILLCOLOR: u32 = 4;
const DRAWTAG_FILLLINGRADIENT: u32 = 20;
const DRAWTAG_FILLIMAGE: u32 = 8;
const DRAWTAG_BEGINCLIP: u32 = 5;
const DRAWTAG_ENDCLIP: u32 = 37;

const TAGS: &[u32] = &[
    DRAWTAG_FILLCOLOR,
    DRAWTAG_FILLLINGRADIENT,
    DRAWTAG_FILLIMAGE,
    DRAWTAG_BEGINCLIP,
    DRAWTAG_ENDCLIP,
];

struct DrawTestData {
    tags: Vec<u32>,
}

pub unsafe fn draw_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("draw");
    // TODO: implement large scan and set large to 1 << 24
    let n_tag: u64 = config.size.choose(1 << 12, 1 << 20, 1 << 22);
    let data = DrawTestData::new(n_tag);
    let stage_config = data.get_config();

    let config_buf = runner
        .session
        .create_buffer_init(std::slice::from_ref(&stage_config), BufferUsage::STORAGE)
        .unwrap();
    let scene_size = n_tag * DRAWTAG_SIZE as u64;
    let scene_buf = runner
        .session
        .create_buffer_with(scene_size, |b| data.fill_scene(b), BufferUsage::STORAGE)
        .unwrap();
    let memory = runner.buf_down(data.memory_size(), BufferUsage::STORAGE);

    let code = DrawCode::new(&runner.session);
    let stage = DrawStage::new(&runner.session, &code);
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
        commands.write_timestamp(0);
        stage.record(&mut commands.cmd_buf, &code, &binding, n_tag);
        commands.write_timestamp(1);
        if i == 0 || config.verify_all {
            commands.cmd_buf.memory_barrier();
            commands.download(&memory);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 || config.verify_all {
            let dst = memory.map_read(..);
            if let Some(failure) = data.verify(&dst) {
                result.fail(failure);
            }
        }
    }
    let n_elements = n_tag;
    result.timing(total_elapsed, n_elements * n_iter);

    result
}

impl DrawTestData {
    fn new(n: u64) -> DrawTestData {
        let mut rng = rand::thread_rng();
        let tags = (0..n).map(|_| *TAGS.choose(&mut rng).unwrap()).collect();
        DrawTestData { tags }
    }

    fn get_config(&self) -> stages::Config {
        let n_tags = self.tags.len();

        // Layout of memory
        let drawmonoid_alloc = 0;
        let anno_alloc = drawmonoid_alloc + 16 * n_tags;
        let clip_alloc = anno_alloc + ANNOTATED_SIZE * n_tags;
        let stage_config = stages::Config {
            n_elements: n_tags as u32,
            anno_alloc: anno_alloc as u32,
            drawmonoid_alloc: drawmonoid_alloc as u32,
            clip_alloc: clip_alloc as u32,
            drawtag_offset: 0,
            ..Default::default()
        };
        stage_config
    }

    fn memory_size(&self) -> u64 {
        // Note: this overallocates the clip buf a bit - only needed for the
        // total number of begin_clip and end_clip tags.
        (8 + self.tags.len() * (16 + 4 + ANNOTATED_SIZE)) as u64
    }

    fn fill_scene(&self, buf: &mut BufWrite) {
        buf.extend_slice(&self.tags);
    }

    fn verify(&self, buf: &[u8]) -> Option<String> {
        let size = self.tags.len() * 16;
        let actual = bytemuck::cast_slice::<u8, DrawMonoid>(&buf[8..8 + size]);
        let mut expected = DrawMonoid::default();
        for (i, (tag, actual)) in self.tags.iter().zip(actual).enumerate() {
            // Verify exclusive prefix sum.
            let (path_ix, clip_ix) = Self::reduce_tag(*tag);
            if *actual != expected {
                println!("{:?} {:?}", actual, expected);
                return Some(format!("draw mismatch at {}", i));
            }
            expected.path_ix += path_ix;
            expected.clip_ix += clip_ix;
            expected.scene_offset += tag & 28;
        }
        None
    }

    fn reduce_tag(tag: u32) -> (u32, u32) {
        match tag {
            DRAWTAG_FILLCOLOR | DRAWTAG_FILLLINGRADIENT | DRAWTAG_FILLIMAGE => (1, 0),
            DRAWTAG_BEGINCLIP | DRAWTAG_ENDCLIP => (1, 1),
            // TODO: ENDCLIP will become (0, 1)
            _ => (0, 0),
        }
    }
}
