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
mod transform;

use bytemuck::{Pod, Zeroable};

pub use path::{PathBinding, PathCode, PathEncoder, PathStage};
pub use transform::{Transform, TransformBinding, TransformCode, TransformStage};

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
