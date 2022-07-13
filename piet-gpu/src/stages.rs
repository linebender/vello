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

mod clip;
mod draw;
mod path;
mod transform;

use bytemuck::{Pod, Zeroable};

pub use clip::{ClipBinding, ClipCode, CLIP_PART_SIZE};
pub use draw::{DrawBinding, DrawCode, DrawMonoid, DrawStage, DRAW_PART_SIZE};
pub use path::{PathBinding, PathCode, PathEncoder, PathStage, PATHSEG_PART_SIZE};
use piet_gpu_hal::{Buffer, ComputePass, Session};
pub use transform::{
    Transform, TransformBinding, TransformCode, TransformStage, TRANSFORM_PART_SIZE,
};

/// The configuration block passed to piet-gpu shaders.
///
/// Note: this should be kept in sync with the version in setup.h.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Zeroable, Pod)]
pub struct Config {
    pub mem_size: u32,
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
    pub path_bbox_alloc: u32,
    pub drawmonoid_alloc: u32,
    pub clip_alloc: u32,
    pub clip_bic_alloc: u32,
    pub clip_stack_alloc: u32,
    pub clip_bbox_alloc: u32,
    pub draw_bbox_alloc: u32,
    pub drawinfo_alloc: u32,
    pub n_trans: u32,
    pub n_path: u32,
    pub n_clip: u32,
    pub trans_offset: u32,
    pub linewidth_offset: u32,
    pub pathtag_offset: u32,
    pub pathseg_offset: u32,
    pub drawtag_offset: u32,
    pub drawdata_offset: u32,
}

// The "element" stage combines a number of stages for parts of the pipeline.

pub struct ElementCode {
    transform_code: TransformCode,
    path_code: PathCode,
    draw_code: DrawCode,
}

pub struct ElementStage {
    transform_stage: TransformStage,
    path_stage: PathStage,
    draw_stage: DrawStage,
}

pub struct ElementBinding {
    transform_binding: TransformBinding,
    path_binding: PathBinding,
    draw_binding: DrawBinding,
}

impl ElementCode {
    pub unsafe fn new(session: &Session) -> ElementCode {
        ElementCode {
            transform_code: TransformCode::new(session),
            path_code: PathCode::new(session),
            draw_code: DrawCode::new(session),
        }
    }
}

impl ElementStage {
    pub unsafe fn new(session: &Session, code: &ElementCode) -> ElementStage {
        ElementStage {
            transform_stage: TransformStage::new(session, &code.transform_code),
            path_stage: PathStage::new(session, &code.path_code),
            draw_stage: DrawStage::new(session, &code.draw_code),
        }
    }

    pub unsafe fn bind(
        &self,
        session: &Session,
        code: &ElementCode,
        config_buf: &Buffer,
        scene_buf: &Buffer,
        memory_buf: &Buffer,
    ) -> ElementBinding {
        ElementBinding {
            transform_binding: self.transform_stage.bind(
                session,
                &code.transform_code,
                config_buf,
                scene_buf,
                memory_buf,
            ),
            path_binding: self.path_stage.bind(
                session,
                &code.path_code,
                config_buf,
                scene_buf,
                memory_buf,
            ),
            draw_binding: self.draw_stage.bind(
                session,
                &code.draw_code,
                config_buf,
                scene_buf,
                memory_buf,
            ),
        }
    }

    pub unsafe fn record(
        &self,
        pass: &mut ComputePass,
        code: &ElementCode,
        binding: &ElementBinding,
        n_transform: u64,
        n_paths: u32,
        n_tags: u32,
        n_drawobj: u64,
    ) {
        self.transform_stage.record(
            pass,
            &code.transform_code,
            &binding.transform_binding,
            n_transform,
        );
        // No memory barrier needed here; path has at least one before pathseg
        self.path_stage.record(
            pass,
            &code.path_code,
            &binding.path_binding,
            n_paths,
            n_tags,
        );
        // No memory barrier needed here; draw has at least one before draw_leaf
        self.draw_stage
            .record(pass, &code.draw_code, &binding.draw_binding, n_drawobj);
    }
}

impl ElementBinding {
    pub unsafe fn rebind_memory(&mut self, session: &Session, memory: &Buffer) {
        self.transform_binding.rebind_memory(session, memory);
        self.path_binding.rebind_memory(session, memory);
        self.draw_binding.rebind_memory(session, memory);
    }

    pub unsafe fn rebind_scene(&mut self, session: &Session, scene: &Buffer) {
        self.transform_binding.rebind_scene(session, scene);
        self.path_binding.rebind_scene(session, scene);
        self.draw_binding.rebind_scene(session, scene);
    }
}
