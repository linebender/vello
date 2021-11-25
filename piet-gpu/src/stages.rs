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

//! Stages for new element pipeline, exposed for testing.

mod path;

use bytemuck::{Pod, Zeroable};

use piet::kurbo::Affine;
use piet_gpu_hal::{
    include_shader, BindType, Buffer, BufferUsage, CmdBuf, DescriptorSet, Pipeline, Session,
};

pub use path::{PathBinding, PathCode, PathEncoder, PathStage};

/// The configuration block passed to piet-gpu shaders.
///
/// Note: this should be kept in sync with the version in setup.h.
#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
pub struct Config {
    pub n_elements: u32, // paths
    pub n_pathseg: u32,
    pub width_in_tiles: u32,
    pub height_in_tiles: u32,
    pub tile_alloc: u32,
    pub bin_alloc: u32,
    pub ptcl_alloc: u32,
    pub pathseg_alloc: u32,
    pub anno_alloc: u32,
    pub trans_alloc: u32,
    pub bbox_alloc: u32,
    pub n_trans: u32,
    pub trans_offset: u32,
    pub pathtag_offset: u32,
    pub linewidth_offset: u32,
    pub pathseg_offset: u32,
}

// The individual stages will probably be separate files but for now, all in one.

// This is equivalent to the version in piet-gpu-types, but the bytemuck
// representation will likely be faster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Transform {
    pub mat: [f32; 4],
    pub translate: [f32; 2],
}

const TRANSFORM_WG: u64 = 512;
const TRANSFORM_N_ROWS: u64 = 8;
const TRANSFORM_PART_SIZE: u64 = TRANSFORM_WG * TRANSFORM_N_ROWS;

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
        let reduce_code = include_shader!(session, "../shader/gen/transform_reduce");
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
        let root_code = include_shader!(session, "../shader/gen/transform_root");
        let root_pipeline = session
            .create_compute_pipeline(root_code, &[BindType::Buffer])
            .unwrap();
        let leaf_code = include_shader!(session, "../shader/gen/transform_leaf");
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
        cmd_buf: &mut CmdBuf,
        code: &TransformCode,
        binding: &TransformBinding,
        size: u64,
    ) {
        if size > TRANSFORM_PART_SIZE.pow(2) {
            panic!("very large scan not yet implemented");
        }
        let n_workgroups = (size + TRANSFORM_PART_SIZE - 1) / TRANSFORM_PART_SIZE;
        if n_workgroups > 1 {
            cmd_buf.dispatch(
                &code.reduce_pipeline,
                &binding.reduce_ds,
                (n_workgroups as u32, 1, 1),
                (TRANSFORM_WG as u32, 1, 1),
            );
            cmd_buf.memory_barrier();
            cmd_buf.dispatch(
                &code.root_pipeline,
                &self.root_ds,
                (1, 1, 1),
                (TRANSFORM_WG as u32, 1, 1),
            );
            cmd_buf.memory_barrier();
        }
        cmd_buf.dispatch(
            &code.leaf_pipeline,
            &binding.leaf_ds,
            (n_workgroups as u32, 1, 1),
            (TRANSFORM_WG as u32, 1, 1),
        );
    }
}

impl Transform {
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
