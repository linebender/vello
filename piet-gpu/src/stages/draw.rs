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

//! The draw object stage of the element processing pipeline.

use bytemuck::{Pod, Zeroable};

use piet_gpu_hal::{
    include_shader, BindType, Buffer, BufferUsage, ComputePass, DescriptorSet, Pipeline, Session,
};

/// The output element of the draw object stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Zeroable, Pod)]
pub struct DrawMonoid {
    pub path_ix: u32,
    pub clip_ix: u32,
    pub scene_offset: u32,
    pub info_offset: u32,
}

const DRAW_WG: u64 = 256;
const DRAW_N_ROWS: u64 = 8;
pub const DRAW_PART_SIZE: u64 = DRAW_WG * DRAW_N_ROWS;

pub struct DrawCode {
    reduce_pipeline: Pipeline,
    root_pipeline: Pipeline,
    leaf_pipeline: Pipeline,
}
pub struct DrawStage {
    // Right now we're limited to partition^2 (~16M) elements. This can be
    // expanded but is tedious.
    root_buf: Buffer,
    root_ds: DescriptorSet,
}

pub struct DrawBinding {
    reduce_ds: DescriptorSet,
    leaf_ds: DescriptorSet,
}

impl DrawCode {
    pub unsafe fn new(session: &Session) -> DrawCode {
        let reduce_code = include_shader!(session, "../../shader/gen/draw_reduce");
        let reduce_pipeline = session
            .create_compute_pipeline(
                reduce_code,
                &[
                    BindType::Buffer,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::Buffer,
                ],
            )
            .unwrap();
        let root_code = include_shader!(session, "../../shader/gen/draw_root");
        let root_pipeline = session
            .create_compute_pipeline(root_code, &[BindType::Buffer])
            .unwrap();
        let leaf_code = include_shader!(session, "../../shader/gen/draw_leaf");
        let leaf_pipeline = session
            .create_compute_pipeline(
                leaf_code,
                &[
                    BindType::Buffer,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                ],
            )
            .unwrap();
        DrawCode {
            reduce_pipeline,
            root_pipeline,
            leaf_pipeline,
        }
    }
}

impl DrawStage {
    pub unsafe fn new(session: &Session, code: &DrawCode) -> DrawStage {
        // We're limited to DRAW_PART_SIZE^2
        // Also note: size here allows padding
        let root_buf_size = DRAW_PART_SIZE * 16;
        let root_buf = session
            .create_buffer(root_buf_size, BufferUsage::STORAGE)
            .unwrap();
        let root_ds = session
            .create_simple_descriptor_set(&code.root_pipeline, &[&root_buf])
            .unwrap();
        DrawStage { root_buf, root_ds }
    }

    pub unsafe fn bind(
        &self,
        session: &Session,
        code: &DrawCode,
        config_buf: &Buffer,
        scene_buf: &Buffer,
        memory_buf: &Buffer,
    ) -> DrawBinding {
        let reduce_ds = session
            .create_simple_descriptor_set(
                &code.reduce_pipeline,
                &[memory_buf, config_buf, scene_buf, &self.root_buf],
            )
            .unwrap();
        let leaf_ds = session
            .create_simple_descriptor_set(
                &code.leaf_pipeline,
                &[memory_buf, config_buf, scene_buf, &self.root_buf],
            )
            .unwrap();
        DrawBinding { reduce_ds, leaf_ds }
    }

    pub unsafe fn record(
        &self,
        pass: &mut ComputePass,
        code: &DrawCode,
        binding: &DrawBinding,
        size: u64,
    ) {
        if size > DRAW_PART_SIZE.pow(2) {
            panic!("very large scan not yet implemented");
        }
        let n_workgroups = (size + DRAW_PART_SIZE - 1) / DRAW_PART_SIZE;
        if n_workgroups > 1 {
            pass.dispatch(
                &code.reduce_pipeline,
                &binding.reduce_ds,
                (n_workgroups as u32, 1, 1),
                (DRAW_WG as u32, 1, 1),
            );
            pass.memory_barrier();
            pass.dispatch(
                &code.root_pipeline,
                &self.root_ds,
                (1, 1, 1),
                (DRAW_WG as u32, 1, 1),
            );
        }
        pass.memory_barrier();
        pass.dispatch(
            &code.leaf_pipeline,
            &binding.leaf_ds,
            (n_workgroups as u32, 1, 1),
            (DRAW_WG as u32, 1, 1),
        );
    }
}
