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

//! The transform stage of the element processing pipeline.

use bytemuck::{Pod, Zeroable};

use piet::kurbo::Affine;
use piet_gpu_hal::{
    include_shader, BindType, Buffer, BufferUsage, ComputePass, DescriptorSet, Pipeline, Session,
};

/// An affine transform.
// This is equivalent to the version in piet-gpu-types, but the bytemuck
// representation will likely be faster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Transform {
    pub mat: [f32; 4],
    pub translate: [f32; 2],
}

const TRANSFORM_WG: u64 = 256;
const TRANSFORM_N_ROWS: u64 = 8;
pub const TRANSFORM_PART_SIZE: u64 = TRANSFORM_WG * TRANSFORM_N_ROWS;

pub struct TransformCode {
    reduce_pipeline: Pipeline,
    root_pipeline: Pipeline,
    leaf_pipeline: Pipeline,
}

pub struct TransformStage {
    // Right now we're limited to partition^2 (~16M) elements. This can be
    // expanded but is tedious.
    root_buf: Buffer,
    root_ds: DescriptorSet,
}

pub struct TransformBinding {
    reduce_ds: DescriptorSet,
    leaf_ds: DescriptorSet,
}

impl TransformCode {
    pub unsafe fn new(session: &Session) -> TransformCode {
        let reduce_code = include_shader!(session, "../../shader/gen/transform_reduce");
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
        let root_code = include_shader!(session, "../../shader/gen/transform_root");
        let root_pipeline = session
            .create_compute_pipeline(root_code, &[BindType::Buffer])
            .unwrap();
        let leaf_code = include_shader!(session, "../../shader/gen/transform_leaf");
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
        TransformCode {
            reduce_pipeline,
            root_pipeline,
            leaf_pipeline,
        }
    }
}

impl TransformStage {
    pub unsafe fn new(session: &Session, code: &TransformCode) -> TransformStage {
        // We're limited to TRANSFORM_PART_SIZE^2
        // Also note: size here allows padding
        let root_buf_size = TRANSFORM_PART_SIZE * 32;
        let root_buf = session
            .create_buffer(root_buf_size, BufferUsage::STORAGE)
            .unwrap();
        let root_ds = session
            .create_simple_descriptor_set(&code.root_pipeline, &[&root_buf])
            .unwrap();
        TransformStage { root_buf, root_ds }
    }

    pub unsafe fn bind(
        &self,
        session: &Session,
        code: &TransformCode,
        config_buf: &Buffer,
        scene_buf: &Buffer,
        memory_buf: &Buffer,
    ) -> TransformBinding {
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
        TransformBinding { reduce_ds, leaf_ds }
    }

    pub unsafe fn record(
        &self,
        pass: &mut ComputePass,
        code: &TransformCode,
        binding: &TransformBinding,
        size: u64,
    ) {
        if size > TRANSFORM_PART_SIZE.pow(2) {
            panic!("very large scan not yet implemented");
        }
        let n_workgroups = (size + TRANSFORM_PART_SIZE - 1) / TRANSFORM_PART_SIZE;
        if n_workgroups > 1 {
            pass.dispatch(
                &code.reduce_pipeline,
                &binding.reduce_ds,
                (n_workgroups as u32, 1, 1),
                (TRANSFORM_WG as u32, 1, 1),
            );
            pass.memory_barrier();
            pass.dispatch(
                &code.root_pipeline,
                &self.root_ds,
                (1, 1, 1),
                (TRANSFORM_WG as u32, 1, 1),
            );
            pass.memory_barrier();
        }
        pass.dispatch(
            &code.leaf_pipeline,
            &binding.leaf_ds,
            (n_workgroups as u32, 1, 1),
            (TRANSFORM_WG as u32, 1, 1),
        );
    }
}

impl TransformBinding {
    pub unsafe fn rebind_memory(&mut self, session: &Session, memory: &Buffer) {
        session.update_buffer_descriptor(&mut self.reduce_ds, 0, memory);
        session.update_buffer_descriptor(&mut self.leaf_ds, 0, memory);
    }

    pub unsafe fn rebind_scene(&mut self, session: &Session, scene: &Buffer) {
        session.update_buffer_descriptor(&mut self.reduce_ds, 2, scene);
        session.update_buffer_descriptor(&mut self.leaf_ds, 2, scene);
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        mat: [1.0, 0.0, 0.0, 1.0],
        translate: [0.0, 0.0],
    };

    pub fn from_kurbo(a: Affine) -> Transform {
        let c = a.as_coeffs();
        Transform {
            mat: [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32],
            translate: [c[4] as f32, c[5] as f32],
        }
    }

    pub fn to_kurbo(self) -> Affine {
        Affine::new([
            self.mat[0] as f64,
            self.mat[1] as f64,
            self.mat[2] as f64,
            self.mat[3] as f64,
            self.translate[0] as f64,
            self.translate[1] as f64,
        ])
    }
}
