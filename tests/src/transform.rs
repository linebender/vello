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

//! Tests for the piet-gpu transform stage.

use crate::{Config, Runner, TestResult};

use kurbo::Affine;
use piet_gpu::stages::{self, Transform, TransformCode, TransformStage};
use piet_gpu_hal::BufferUsage;
use rand::Rng;

struct AffineTestData {
    input_data: Vec<Transform>,
    expected: Vec<Affine>,
}

pub unsafe fn transform_test(runner: &mut Runner, config: &Config) -> TestResult {
    let mut result = TestResult::new("transform");
    // TODO: implement large scan and set large to 1 << 24
    let n_elements: u64 = config.size.choose(1 << 12, 1 << 18, 1 << 22);
    // Validate with real transform data.
    let data = AffineTestData::new(n_elements as usize);
    let data_buf = runner
        .session
        .create_buffer_init(&data.input_data, BufferUsage::STORAGE)
        .unwrap();
    let memory = runner.buf_down(data_buf.size() + 8, BufferUsage::empty());
    let stage_config = stages::Config {
        n_trans: n_elements as u32,
        ..Default::default()
    };
    let config_buf = runner
        .session
        .create_buffer_init(std::slice::from_ref(&stage_config), BufferUsage::STORAGE)
        .unwrap();

    let code = TransformCode::new(&runner.session);
    let stage = TransformStage::new(&runner.session, &code);
    let binding = stage.bind(
        &runner.session,
        &code,
        &config_buf,
        &data_buf,
        &memory.dev_buf,
    );
    let mut total_elapsed = 0.0;
    let n_iter = config.n_iter;
    for i in 0..n_iter {
        let mut commands = runner.commands();
        commands.write_timestamp(0);
        stage.record(&mut commands.cmd_buf, &code, &binding, n_elements);
        commands.write_timestamp(1);
        if i == 0 || config.verify_all {
            commands.cmd_buf.memory_barrier();
            commands.download(&memory);
        }
        total_elapsed += runner.submit(commands);
        if i == 0 || config.verify_all {
            let dst = memory.map_read(8..);
            if let Some(failure) = data.verify(dst.cast_slice()) {
                result.fail(failure);
            }
        }
    }
    result.timing(total_elapsed, n_elements * n_iter);
    result
}

impl AffineTestData {
    fn new(n: usize) -> AffineTestData {
        let mut rng = rand::thread_rng();
        let mut a = Affine::default();
        let mut input_data = Vec::with_capacity(n);
        let mut expected = Vec::with_capacity(n);
        for _ in 0..n {
            loop {
                let b = Affine::new([
                    rng.gen_range(-3.0..3.0),
                    rng.gen_range(-3.0..3.0),
                    rng.gen_range(-3.0..3.0),
                    rng.gen_range(-3.0..3.0),
                    rng.gen_range(-3.0..3.0),
                    rng.gen_range(-3.0..3.0),
                ]);
                if b.determinant().abs() >= 1.0 {
                    expected.push(b);
                    let c = a.inverse() * b;
                    input_data.push(Transform::from_kurbo(c));
                    a = b;
                    break;
                }
            }
        }
        AffineTestData {
            input_data,
            expected,
        }
    }

    fn verify(&self, actual: &[Transform]) -> Option<String> {
        for (i, (actual, expected)) in actual.iter().zip(&self.expected).enumerate() {
            let error: f64 = actual
                .to_kurbo()
                .as_coeffs()
                .iter()
                .zip(expected.as_coeffs())
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum();
            // Hopefully this is right; most of the time the error is much
            // smaller, but occasionally we see outliers.
            let tolerance = 1e-9 * (i + 1) as f64;
            if error > tolerance {
                return Some(format!("{}: {} {}", i, error, tolerance));
            }
        }
        None
    }
}
