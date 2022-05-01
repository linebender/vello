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

use rand::Rng;

use bytemuck::{Pod, Zeroable};
use piet_gpu_hal::{include_shader, BindType, BufferUsage, ComputePass, DescriptorSet};
use piet_gpu_hal::{Buffer, Pipeline};

use crate::config::Config;
use crate::runner::Runner;
use crate::test_result::TestResult;

const WG_SIZE: u64 = 256;
const N_ROWS: u64 = 1;
const ELEMENTS_PER_WG: u64 = WG_SIZE * N_ROWS;

struct UnionCode {
    reduce_pipeline: Pipeline,
    leaf_pipeline: Pipeline,
}

struct UnionStage {
    bic_buf: Buffer,
    union_buf: Buffer,
}

struct UnionBinding {
    reduce_ds: DescriptorSet,
    leaf_ds: DescriptorSet,
}

const OPEN_PAREN: u32 = 0;
const CLOSE_PAREN: u32 = 1;
const LEAF: u32 = 2;

#[derive(Clone, Copy, Zeroable, Pod, Default, Debug)]
#[repr(C)]
struct Node {
    node_type: u32,
    // Note: the padding here is annoying, but it leaves the code in a somewhat
    // more readable state than hand-marshaling.
    pad1: u32,
    pad2: u32,
    pad3: u32,
    bbox: [f32; 4],
}

struct UnionData {
    nodes: Vec<Node>,
}

pub unsafe fn run_union_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("Bounding box union");
    println!("# bounding box union");
    for exp in 10..=16 {
        let n_elements: u64 = 1 << exp;
        let data = UnionData::new(n_elements);
        let data_buf = runner
            .session
            .create_buffer_init(&data.nodes, BufferUsage::STORAGE)
            .unwrap();
        let out_buf = runner.buf_down(data_buf.size(), BufferUsage::empty());

        let code = UnionCode::new(runner);
        let stage = UnionStage::new(runner, n_elements);
        let binding = stage.bind(runner, &code, &data_buf, &out_buf.dev_buf);

        let mut total_elapsed = 0.0;
        let n_iter = config.n_iter;
        for i in 0..n_iter {
            let mut commands = runner.commands();
            let mut pass = commands.compute_pass(0, 1);
            stage.record(&mut pass, &code, &binding, n_elements);
            pass.end();
            if i == 0 || config.verify_all {
                commands.cmd_buf.memory_barrier();
                commands.download(&out_buf);
            }
            total_elapsed += runner.submit(commands);
            if i == 0 || config.verify_all {
                let dst = out_buf.map_read(..);
                if let Some(failure) = data.verify(dst.cast_slice()) {
                    result.fail(failure);
                }
            }
        }
        let throughput = (n_elements * n_iter) as f64 / total_elapsed;
        println!("{} {}", n_elements, throughput);
    }
    println!("e");

    //result.timing(total_elapsed, n_elements * n_iter);
    result
}

impl UnionCode {
    unsafe fn new(runner: &mut Runner) -> UnionCode {
        let reduce_code = include_shader!(&runner.session, "../shader/gen/union_reduce");
        let reduce_pipeline = runner
            .session
            .create_compute_pipeline(
                reduce_code,
                &[BindType::BufReadOnly, BindType::Buffer, BindType::Buffer],
            )
            .unwrap();
        let leaf_code = include_shader!(&runner.session, "../shader/gen/union_leaf");
        let leaf_pipeline = runner
            .session
            .create_compute_pipeline(
                leaf_code,
                &[
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::Buffer,
                ],
            )
            .unwrap();
        UnionCode {
            reduce_pipeline,
            leaf_pipeline,
        }
    }
}

impl UnionStage {
    unsafe fn new(runner: &mut Runner, n_elements: u64) -> UnionStage {
        assert!(n_elements <= ELEMENTS_PER_WG.pow(2));
        let union_buf = runner
            .session
            .create_buffer(16 * n_elements, BufferUsage::STORAGE)
            .unwrap();
        let bic_size = ELEMENTS_PER_WG * 32;
        let bic_buf = runner
            .session
            .create_buffer(bic_size, BufferUsage::STORAGE)
            .unwrap();
        UnionStage { bic_buf, union_buf }
    }

    unsafe fn bind(
        &self,
        runner: &mut Runner,
        code: &UnionCode,
        in_buf: &Buffer,
        out_buf: &Buffer,
    ) -> UnionBinding {
        let reduce_ds = runner
            .session
            .create_simple_descriptor_set(
                &code.reduce_pipeline,
                &[in_buf, &self.bic_buf, &self.union_buf],
            )
            .unwrap();
        let leaf_ds = runner
            .session
            .create_simple_descriptor_set(
                &code.leaf_pipeline,
                &[in_buf, &self.bic_buf, &self.union_buf, out_buf],
            )
            .unwrap();
        UnionBinding { reduce_ds, leaf_ds }
    }

    unsafe fn record(
        &self,
        pass: &mut ComputePass,
        code: &UnionCode,
        binding: &UnionBinding,
        size: u64,
    ) {
        let n_workgroups = (size + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
        pass.dispatch(
            &code.reduce_pipeline,
            &binding.reduce_ds,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
        pass.memory_barrier();
        pass.dispatch(
            &code.leaf_pipeline,
            &binding.leaf_ds,
            (n_workgroups as u32, 1, 1),
            (WG_SIZE as u32, 1, 1),
        );
    }
}

fn rand_bbox() -> [f32; 4] {
    let mut rng = rand::thread_rng();
    const Y_MIN: f32 = -1000.0;
    const Y_MAX: f32 = 1000.0;
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

fn bbox_union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

const EMPTY_BBOX: [f32; 4] = [1e9, 1e9, -1e9, -1e9];

impl UnionData {
    /// Generate a random scene
    fn new(n: u64) -> UnionData {
        let mut rng = rand::thread_rng();
        let mut depth = 0;
        let nodes = (0..n)
            .map(|_| {
                let node_type = if depth < 1 {
                    OPEN_PAREN
                } else {
                    rng.gen_range(0, 3)
                };
                if node_type == OPEN_PAREN {
                    depth += 1;
                } else if node_type == CLOSE_PAREN {
                    depth -= 1;
                }
                let bbox = if node_type == LEAF {
                    // Could have bboxes on paren nodes, but keep things simple
                    rand_bbox()
                } else {
                    EMPTY_BBOX
                };
                Node { node_type, bbox, .. Default::default() }
            })
            .collect();
        UnionData { nodes }
    }

    fn verify(&self, data: &[[f32; 4]]) -> Option<String> {
        let mut stack = Vec::new();
        for (i, (inp, outp)) in self.nodes.iter().zip(data).enumerate() {
            //println!("{}: {:.1?} {:.1?}", i, inp, outp);
            let mut expected = inp.bbox;
            match inp.node_type {
                OPEN_PAREN => stack.push(EMPTY_BBOX),
                CLOSE_PAREN => {
                    let tos = stack.pop().unwrap();
                    expected = tos;
                    if let Some(nos) = stack.last_mut() {
                        *nos = bbox_union(*nos, tos);
                    }
                }
                LEAF => {
                    let tos = stack.last_mut().unwrap();
                    *tos = bbox_union(*tos, inp.bbox);
                }
                _ => unreachable!(),
            }
            if expected != *outp {
                println!("{}: {:.1?} {:.1?}", i, expected, outp);
            }
        }
        None
    }
}
