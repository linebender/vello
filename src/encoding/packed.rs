// Copyright 2022 Google LLC
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

use bytemuck::{Pod, Zeroable};

use super::{
    resource::{Patch, ResourceCache, Token},
    DrawTag, Encoding, PathTag, Transform,
};
use crate::shaders;

/// Layout of a packed encoding.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct Layout {
    /// Number of draw objects.
    pub n_draw_objects: u32,
    /// Number of paths.
    pub n_paths: u32,
    /// Number of clips.
    pub n_clips: u32,
    /// Start of binning data.
    pub bin_data_start: u32,
    /// Start of path tag stream.
    pub path_tag_base: u32,
    /// Start of path data stream.
    pub path_data_base: u32,
    /// Start of draw tag stream.
    pub draw_tag_base: u32,
    /// Start of draw data stream.
    pub draw_data_base: u32,
    /// Start of transform stream.
    pub transform_base: u32,
    /// Start of linewidth stream.
    pub linewidth_base: u32,
}

/// Scene configuration.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct Config {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
}

/// Packed encoding of scene data.
#[derive(Default)]
pub struct PackedEncoding {
    /// Layout of the packed scene data.
    pub layout: Layout,
    /// Packed scene data.
    pub data: Vec<u8>,
    /// Token for current cached resource state.
    pub resources: Token,
}

impl PackedEncoding {
    /// Creates a new packed encoding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the path tag stream.
    pub fn path_tags(&self) -> &[PathTag] {
        let start = self.layout.path_tag_base as usize * 4;
        let end = self.layout.path_data_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the path tag stream in chunks of 4.
    pub fn path_tags_chunked(&self) -> &[u32] {
        let start = self.layout.path_tag_base as usize * 4;
        let end = self.layout.path_data_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the path data stream.
    pub fn path_data(&self) -> &[[f32; 2]] {
        let start = self.layout.path_data_base as usize * 4;
        let end = self.layout.draw_tag_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the draw tag stream.
    pub fn draw_tags(&self) -> &[DrawTag] {
        let start = self.layout.draw_tag_base as usize * 4;
        let end = self.layout.draw_data_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the draw data stream.
    pub fn draw_data(&self) -> &[u32] {
        let start = self.layout.draw_data_base as usize * 4;
        let end = self.layout.transform_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the transform stream.
    pub fn transforms(&self) -> &[Transform] {
        let start = self.layout.transform_base as usize * 4;
        let end = self.layout.linewidth_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..end])
    }

    /// Returns the linewidth stream.
    pub fn linewidths(&self) -> &[f32] {
        let start = self.layout.linewidth_base as usize * 4;
        bytemuck::cast_slice(&self.data[start..])
    }
}

impl PackedEncoding {
    /// Packs the given encoding into self using the specified cache to handle
    /// late bound resources.
    pub fn pack(&mut self, encoding: &Encoding, resource_cache: &mut ResourceCache) {
        // Advance the resource cache epoch.
        self.resources = resource_cache.advance();
        // Pack encoded data.
        let layout = &mut self.layout;
        *layout = Layout::default();
        layout.n_paths = encoding.n_paths;
        layout.n_draw_objects = encoding.n_paths;
        layout.n_clips = encoding.n_clips;
        let data = &mut self.data;
        data.clear();
        // Path tag stream
        let n_path_tags = encoding.path_tags.len();
        let path_tag_padded = align_up(n_path_tags, 4 * shaders::PATHTAG_REDUCE_WG);
        let capacity = path_tag_padded
            + slice_size_in_bytes(&encoding.path_data)
            + slice_size_in_bytes(&encoding.draw_tags)
            + slice_size_in_bytes(&encoding.draw_data)
            + slice_size_in_bytes(&encoding.transforms)
            + slice_size_in_bytes(&encoding.linewidths);
        data.reserve(capacity);
        layout.path_tag_base = size_to_words(data.len());
        data.extend_from_slice(bytemuck::cast_slice(&encoding.path_tags));
        data.resize(path_tag_padded, 0);
        // Path data stream
        layout.path_data_base = size_to_words(data.len());
        data.extend_from_slice(&encoding.path_data);
        // Draw tag stream
        layout.draw_tag_base = size_to_words(data.len());
        data.extend_from_slice(bytemuck::cast_slice(&encoding.draw_tags));
        // Bin data follows draw info
        layout.bin_data_start = encoding.draw_tags.iter().map(|tag| tag.info_size()).sum();
        // Draw data stream
        layout.draw_data_base = size_to_words(data.len());
        // Handle patches, if any
        if !encoding.patches.is_empty() {
            let stop_data = &encoding.color_stops;
            let mut pos = 0;
            for patch in &encoding.patches {
                let (offset, value) = match patch {
                    Patch::Ramp { offset, stops } => {
                        let ramp_id = resource_cache.add_ramp(&stop_data[stops.clone()]);
                        (*offset, ramp_id)
                    }
                };
                if pos < offset {
                    data.extend_from_slice(&encoding.draw_data[pos..offset]);
                }
                data.extend_from_slice(bytemuck::bytes_of(&value));
                pos = offset + 4;
            }
            if pos < encoding.draw_data.len() {
                data.extend_from_slice(&encoding.draw_data[pos..])
            }
        } else {
            data.extend_from_slice(&encoding.draw_data);
        }
        // Transform stream
        layout.transform_base = size_to_words(data.len());
        data.extend_from_slice(bytemuck::cast_slice(&encoding.transforms));
        // Linewidth stream
        layout.linewidth_base = size_to_words(data.len());
        data.extend_from_slice(bytemuck::cast_slice(&encoding.linewidths));
    }
}

fn slice_size_in_bytes<T: Sized>(slice: &[T]) -> usize {
    slice.len() * std::mem::size_of::<T>()
}

fn size_to_words(byte_size: usize) -> u32 {
    (byte_size / std::mem::size_of::<u32>()) as u32
}

fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & (alignment as usize - 1))
}
