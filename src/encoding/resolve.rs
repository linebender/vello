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

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use moscato::pinot::FontRef;

use super::{
    glyph_cache::{CachedRange, GlyphCache, GlyphKey},
    ramp_cache::{RampCache, Ramps},
    DrawTag, Encoding, PathTag, StreamOffsets, Transform,
};
use crate::glyph::GlyphContext;
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

impl Layout {
    /// Creates a zeroed layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the path tag stream.
    pub fn path_tags<'a>(&self, data: &'a [u8]) -> &'a [PathTag] {
        let start = self.path_tag_base as usize * 4;
        let end = self.path_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the path tag stream in chunks of 4.
    pub fn path_tags_chunked<'a>(&self, data: &'a [u8]) -> &'a [u32] {
        let start = self.path_tag_base as usize * 4;
        let end = self.path_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the path data stream.
    pub fn path_data<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let start = self.path_data_base as usize * 4;
        let end = self.draw_tag_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the draw tag stream.
    pub fn draw_tags<'a>(&self, data: &'a [u8]) -> &'a [DrawTag] {
        let start = self.draw_tag_base as usize * 4;
        let end = self.draw_data_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the draw data stream.
    pub fn draw_data<'a>(&self, data: &'a [u8]) -> &'a [u32] {
        let start = self.draw_data_base as usize * 4;
        let end = self.transform_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the transform stream.
    pub fn transforms<'a>(&self, data: &'a [u8]) -> &'a [Transform] {
        let start = self.transform_base as usize * 4;
        let end = self.linewidth_base as usize * 4;
        bytemuck::cast_slice(&data[start..end])
    }

    /// Returns the linewidth stream.
    pub fn linewidths<'a>(&self, data: &'a [u8]) -> &'a [f32] {
        let start = self.linewidth_base as usize * 4;
        bytemuck::cast_slice(&data[start..])
    }
}

/// Scene configuration.
///
/// This data structure must be kept in sync with the definition in
/// shaders/shared/config.wgsl.
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
    /// The base background color applied to the target before any blends.
    pub base_color: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of binning buffer allocation (in u32s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in Tiles).
    pub tiles_size: u32,
    /// Size of segment buffer allocation (in PathSegments).
    pub segments_size: u32,
    /// Size of per-tile command list buffer allocation (in u32s).
    pub ptcl_size: u32,
}

/// Resolver for late bound resources.
#[derive(Default)]
pub struct Resolver {
    glyph_cache: GlyphCache,
    glyph_ranges: Vec<CachedRange>,
    glyph_cx: GlyphContext,
    ramp_cache: RampCache,
    patches: Vec<ResolvedPatch>,
}

impl Resolver {
    /// Creates a new resource cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves late bound resources and packs an encoding. Returns the packed
    /// layout and computed ramp data.
    pub fn resolve<'a>(
        &'a mut self,
        encoding: &Encoding,
        packed: &mut Vec<u8>,
    ) -> (Layout, Ramps<'a>) {
        let sizes = self.resolve_patches(encoding);
        let data = packed;
        data.clear();
        let mut layout = Layout::default();
        layout.n_paths = encoding.n_paths;
        layout.n_clips = encoding.n_clips;
        // Compute size of data buffer
        let n_path_tags =
            encoding.path_tags.len() + sizes.path_tags + encoding.n_open_clips as usize;
        let path_tag_padded = align_up(n_path_tags, 4 * shaders::PATHTAG_REDUCE_WG);
        let capacity = path_tag_padded
            + slice_size_in_bytes(&encoding.path_data, sizes.path_data)
            + slice_size_in_bytes(
                &encoding.draw_tags,
                sizes.draw_tags + encoding.n_open_clips as usize,
            )
            + slice_size_in_bytes(&encoding.draw_data, sizes.draw_data)
            + slice_size_in_bytes(&encoding.transforms, sizes.transforms)
            + slice_size_in_bytes(&encoding.linewidths, sizes.linewidths);
        data.reserve(capacity);
        // Path tag stream
        layout.path_tag_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.path_tags;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    layout.n_paths += 1;
                    let stream_offset = encoding.glyph_runs[*index].stream_offsets.path_tags;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyph_ranges[glyphs.clone()] {
                        data.extend_from_slice(bytemuck::bytes_of(&PathTag::TRANSFORM));
                        let glyph_data = &self.glyph_cache.encoding.path_tags
                            [glyph.start.path_tags..glyph.end.path_tags];
                        data.extend_from_slice(bytemuck::cast_slice(glyph_data));
                    }
                    data.extend_from_slice(bytemuck::bytes_of(&PathTag::PATH));
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
            for _ in 0..encoding.n_open_clips {
                data.extend_from_slice(bytemuck::bytes_of(&PathTag::PATH));
            }
            data.resize(path_tag_padded, 0);
        }
        // Path data stream
        layout.path_data_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.path_data;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    let stream_offset = encoding.glyph_runs[*index].stream_offsets.path_data;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyph_ranges[glyphs.clone()] {
                        let glyph_data = &self.glyph_cache.encoding.path_data
                            [glyph.start.path_data..glyph.end.path_data];
                        data.extend_from_slice(bytemuck::cast_slice(glyph_data));
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Draw tag stream
        layout.draw_tag_base = size_to_words(data.len());
        // Bin data follows draw info
        layout.bin_data_start = encoding.draw_tags.iter().map(|tag| tag.info_size()).sum();
        {
            data.extend_from_slice(bytemuck::cast_slice(&encoding.draw_tags));
            for _ in 0..encoding.n_open_clips {
                data.extend_from_slice(bytemuck::bytes_of(&DrawTag::END_CLIP));
            }
        }
        // Draw data stream
        layout.draw_data_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.draw_data;
            for patch in &self.patches {
                match patch {
                    ResolvedPatch::Ramp {
                        draw_data_offset,
                        ramp_id,
                    } => {
                        if pos < *draw_data_offset {
                            data.extend_from_slice(&encoding.draw_data[pos..*draw_data_offset]);
                        }
                        data.extend_from_slice(bytemuck::bytes_of(ramp_id));
                        pos = *draw_data_offset + 4;
                    }
                    ResolvedPatch::GlyphRun { .. } => {}
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Transform stream
        layout.transform_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.transforms;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun {
                    index,
                    glyphs: _,
                    transform,
                } = patch
                {
                    let run = &encoding.glyph_runs[*index];
                    let stream_offset = encoding.glyph_runs[*index].stream_offsets.transforms;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    if let Some(glyph_transform) = run.glyph_transform {
                        for glyph in &encoding.glyphs[run.glyphs.clone()] {
                            let xform = *transform
                                * Transform {
                                    matrix: [1.0, 0.0, 0.0, -1.0],
                                    translation: [glyph.x, glyph.y],
                                }
                                * glyph_transform;
                            data.extend_from_slice(bytemuck::bytes_of(&xform));
                        }
                    } else {
                        for glyph in &encoding.glyphs[run.glyphs.clone()] {
                            let xform = *transform
                                * Transform {
                                    matrix: [1.0, 0.0, 0.0, -1.0],
                                    translation: [glyph.x, glyph.y],
                                };
                            data.extend_from_slice(bytemuck::bytes_of(&xform));
                        }
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        // Linewidth stream
        layout.linewidth_base = size_to_words(data.len());
        {
            let mut pos = 0;
            let stream = &encoding.linewidths;
            for patch in &self.patches {
                if let ResolvedPatch::GlyphRun { index, glyphs, .. } = patch {
                    let stream_offset = encoding.glyph_runs[*index].stream_offsets.linewidths;
                    if pos < stream_offset {
                        data.extend_from_slice(bytemuck::cast_slice(&stream[pos..stream_offset]));
                        pos = stream_offset;
                    }
                    for glyph in &self.glyph_ranges[glyphs.clone()] {
                        let glyph_data = &self.glyph_cache.encoding.linewidths
                            [glyph.start.linewidths..glyph.end.linewidths];
                        data.extend_from_slice(bytemuck::cast_slice(glyph_data));
                    }
                }
            }
            if pos < stream.len() {
                data.extend_from_slice(bytemuck::cast_slice(&stream[pos..]));
            }
        }
        layout.n_draw_objects = layout.n_paths;
        assert_eq!(capacity, data.len());
        (layout, self.ramp_cache.ramps())
    }

    fn resolve_patches(&mut self, encoding: &Encoding) -> StreamOffsets {
        self.ramp_cache.advance();
        self.glyph_cache.clear();
        self.glyph_ranges.clear();
        self.patches.clear();
        let mut sizes = StreamOffsets::default();
        for patch in &encoding.patches {
            match patch {
                Patch::Ramp { offset, stops } => {
                    let ramp_id = self.ramp_cache.add(&encoding.color_stops[stops.clone()]);
                    self.patches.push(ResolvedPatch::Ramp {
                        draw_data_offset: *offset + sizes.draw_data,
                        ramp_id,
                    });
                }
                Patch::GlyphRun { index } => {
                    let mut run_sizes = StreamOffsets::default();
                    let run = &encoding.glyph_runs[*index];
                    let font_id = run.font.data.id();
                    let font_size_u32 = run.font_size.to_bits();
                    let Some(font) = FontRef::from_index(run.font.data.as_ref(), run.font.index) else { continue };
                    let glyphs = &encoding.glyphs[run.glyphs.clone()];
                    let _coords = &encoding.normalized_coords[run.normalized_coords.clone()];
                    let vars: [(moscato::pinot::types::Tag, f32); 0] = [];
                    let hint_id = if run.font.index < 0xFF {
                        Some(font_id << 8 | run.font.index as u64)
                    } else {
                        None
                    };
                    let mut hint = run.hint;
                    let mut font_size = run.font_size;
                    let mut transform = run.transform;
                    if hint {
                        // If hinting was requested and our transform matrix is just a uniform
                        // scale, then adjust our font size and cancel out the matrix. Otherwise,
                        // disable hinting entirely.
                        if transform.matrix[0] == transform.matrix[3]
                            && transform.matrix[1] == 0.0
                            && transform.matrix[2] == 0.0
                        {
                            font_size *= transform.matrix[0];
                            transform.matrix = [1.0, 0.0, 0.0, 1.0];
                        } else {
                            hint = false;
                        }
                    }
                    let mut scaler = self
                        .glyph_cx
                        .new_provider(&font, hint_id, font_size, hint, vars);
                    let glyph_start = self.glyph_ranges.len();
                    for glyph in glyphs {
                        let key = GlyphKey {
                            font_id,
                            font_index: run.font.index,
                            font_size: font_size_u32,
                            glyph_id: glyph.id,
                            hint: run.hint,
                        };
                        let encoding_range = self
                            .glyph_cache
                            .get_or_insert(key, &run.style, &mut scaler)
                            .unwrap_or_default();
                        run_sizes.add(&encoding_range.len());
                        self.glyph_ranges.push(encoding_range);
                    }
                    let glyph_end = self.glyph_ranges.len();
                    run_sizes.path_tags += glyphs.len() + 1;
                    run_sizes.transforms += glyphs.len();
                    sizes.add(&run_sizes);
                    self.patches.push(ResolvedPatch::GlyphRun {
                        index: *index,
                        glyphs: glyph_start..glyph_end,
                        transform,
                    });
                }
            }
        }
        sizes
    }
}

#[derive(Clone)]
/// Patch for a late bound resource.
pub enum Patch {
    /// Gradient ramp resource.
    Ramp {
        /// Byte offset to the ramp id in the draw data stream.
        offset: usize,
        /// Range of the gradient stops in the resource set.
        stops: Range<usize>,
    },
    /// Glyph run resource.
    GlyphRun {
        /// Index in the glyph run buffer.
        index: usize,
    },
}

#[derive(Clone, Debug)]
enum ResolvedPatch {
    Ramp {
        /// Offset to the ramp id in draw data stream.
        draw_data_offset: usize,
        /// Resolved ramp index.
        ramp_id: u32,
    },
    GlyphRun {
        /// Index of the original glyph run in the encoding.
        index: usize,
        /// Range into the glyphs encoding range buffer.
        glyphs: Range<usize>,
        /// Global transform.
        transform: Transform,
    },
}

fn slice_size_in_bytes<T: Sized>(slice: &[T], extra: usize) -> usize {
    (slice.len() + extra) * std::mem::size_of::<T>()
}

fn size_to_words(byte_size: usize) -> u32 {
    (byte_size / std::mem::size_of::<u32>()) as u32
}

fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & (alignment as usize - 1))
}
