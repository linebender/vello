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

mod builder;
mod resource;

pub use builder::SceneBuilder;
pub use resource::{ResourceBundle, ResourcePatch};

use super::conv;
use peniko::kurbo::Affine;

/// Raw data streams describing an encoded scene.
#[derive(Default)]
pub struct SceneData {
    pub transform_stream: Vec<[f32; 6]>,
    pub tag_stream: Vec<u8>,
    pub pathseg_stream: Vec<u8>,
    pub linewidth_stream: Vec<f32>,
    pub drawtag_stream: Vec<u32>,
    pub drawdata_stream: Vec<u8>,
    pub n_path: u32,
    pub n_pathseg: u32,
    pub n_clip: u32,
    pub resources: ResourceBundle,
}

impl SceneData {
    fn is_empty(&self) -> bool {
        self.pathseg_stream.is_empty()
    }

    fn reset(&mut self, is_fragment: bool) {
        self.transform_stream.clear();
        self.tag_stream.clear();
        self.pathseg_stream.clear();
        self.linewidth_stream.clear();
        self.drawtag_stream.clear();
        self.drawdata_stream.clear();
        self.n_path = 0;
        self.n_pathseg = 0;
        self.n_clip = 0;
        self.resources.clear();
        if !is_fragment {
            self.transform_stream.push([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            self.linewidth_stream.push(-1.0);
        }
    }

    fn append(&mut self, other: &SceneData, transform: &Option<Affine>) {
        let stops_base = self.resources.stops.len();
        let drawdata_base = self.drawdata_stream.len();
        if let Some(transform) = *transform {
            self.transform_stream.extend(
                other
                    .transform_stream
                    .iter()
                    .map(|x| conv::affine_to_f32(&(transform * conv::affine_from_f32(x)))),
            );
        } else {
            self.transform_stream
                .extend_from_slice(&other.transform_stream);
        }
        self.tag_stream.extend_from_slice(&other.tag_stream);
        self.pathseg_stream.extend_from_slice(&other.pathseg_stream);
        self.linewidth_stream
            .extend_from_slice(&other.linewidth_stream);
        self.drawtag_stream.extend_from_slice(&other.drawtag_stream);
        self.drawdata_stream
            .extend_from_slice(&other.drawdata_stream);
        self.n_path += other.n_path;
        self.n_pathseg += other.n_pathseg;
        self.n_clip += other.n_clip;
        self.resources
            .stops
            .extend_from_slice(&other.resources.stops);
        self.resources
            .patches
            .extend(other.resources.patches.iter().map(|patch| match patch {
                ResourcePatch::Ramp { offset, stops } => {
                    let stops = stops.start + stops_base..stops.end + stops_base;
                    ResourcePatch::Ramp {
                        offset: drawdata_base + offset,
                        stops,
                    }
                }
            }));
    }
}

/// Encoded definition of a scene that is ready for rendering when paired with
/// an associated resource context.
#[derive(Default)]
pub struct Scene {
    data: SceneData,
}

impl Scene {
    /// Returns the raw encoded scene data streams.
    pub fn data(&self) -> &SceneData {
        &self.data
    }
}

/// Encoded definition of a scene fragment and associated resources.
#[derive(Default)]
pub struct SceneFragment {
    data: SceneData,
}

impl SceneFragment {
    /// Returns true if the fragment does not contain any paths.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the the entire sequence of points in the scene fragment.
    pub fn points(&self) -> &[[f32; 2]] {
        if self.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&self.data.pathseg_stream)
        }
    }
}
