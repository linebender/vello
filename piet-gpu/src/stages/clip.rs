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

//! The clip processing stage (includes substages).

use piet_gpu_hal::{include_shader, BindType, Buffer, ComputePass, DescriptorSet, Pipeline, Session};

// Note that this isn't the code/stage/binding pattern of most of the other stages
// in the new element processing pipeline. We want to move those temporary buffers
// into common memory and converge on this pattern.
pub struct ClipCode {
    reduce_pipeline: Pipeline,
    leaf_pipeline: Pipeline,
}

pub struct ClipBinding {
    reduce_ds: DescriptorSet,
    leaf_ds: DescriptorSet,
}

pub const CLIP_PART_SIZE: u32 = 256;

impl ClipCode {
    pub unsafe fn new(session: &Session) -> ClipCode {
        let reduce_code = include_shader!(session, "../../shader/gen/clip_reduce");
        let reduce_pipeline = session
            .create_compute_pipeline(reduce_code, &[BindType::Buffer, BindType::BufReadOnly])
            .unwrap();
        let leaf_code = include_shader!(session, "../../shader/gen/clip_leaf");
        let leaf_pipeline = session
            .create_compute_pipeline(leaf_code, &[BindType::Buffer, BindType::BufReadOnly])
            .unwrap();
        ClipCode {
            reduce_pipeline,
            leaf_pipeline,
        }
    }
}

impl ClipBinding {
    pub unsafe fn new(
        session: &Session,
        code: &ClipCode,
        config: &Buffer,
        memory: &Buffer,
    ) -> ClipBinding {
        let reduce_ds = session
            .create_simple_descriptor_set(&code.reduce_pipeline, &[memory, config])
            .unwrap();
        let leaf_ds = session
            .create_simple_descriptor_set(&code.leaf_pipeline, &[memory, config])
            .unwrap();
        ClipBinding { reduce_ds, leaf_ds }
    }

    /// Record the clip dispatches.
    ///
    /// Assumes memory barrier on entry. Provides memory barrier on exit.
    pub unsafe fn record(&self, pass: &mut ComputePass, code: &ClipCode, n_clip: u32) {
        let n_wg_reduce = n_clip.saturating_sub(1) / CLIP_PART_SIZE;
        if n_wg_reduce > 0 {
            pass.dispatch(
                &code.reduce_pipeline,
                &self.reduce_ds,
                (n_wg_reduce, 1, 1),
                (CLIP_PART_SIZE, 1, 1),
            );
            pass.memory_barrier();
        }
        let n_wg = (n_clip + CLIP_PART_SIZE - 1) / CLIP_PART_SIZE;
        if n_wg > 0 {
            pass.dispatch(
                &code.leaf_pipeline,
                &self.leaf_ds,
                (n_wg, 1, 1),
                (CLIP_PART_SIZE, 1, 1),
            );
            pass.memory_barrier();
        }
    }
}
