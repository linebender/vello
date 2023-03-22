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

use crate::encoding::DrawImage;

use super::{
    resolve::Patch, DrawColor, DrawLinearGradient, DrawRadialGradient, DrawTag, Glyph, GlyphRun,
    PathEncoder, PathTag, Transform,
};

use fello::NormalizedCoord;
use peniko::{kurbo::Shape, BlendMode, BrushRef, ColorStop, Extend, GradientKind, Image};

/// Encoded data streams for a scene.
#[derive(Clone, Default)]
pub struct Encoding {
    /// The path tag stream.
    pub path_tags: Vec<PathTag>,
    /// The path data stream.
    pub path_data: Vec<u8>,
    /// The draw tag stream.
    pub draw_tags: Vec<DrawTag>,
    /// The draw data stream.
    pub draw_data: Vec<u8>,
    /// Draw data patches for late bound resources.
    pub patches: Vec<Patch>,
    /// Color stop collection for gradients.
    pub color_stops: Vec<ColorStop>,
    /// The transform stream.
    pub transforms: Vec<Transform>,
    /// The line width stream.
    pub linewidths: Vec<f32>,
    /// Positioned glyph buffer.
    pub glyphs: Vec<Glyph>,
    /// Sequences of glyphs.
    pub glyph_runs: Vec<GlyphRun>,
    /// Normalized coordinate buffer for variable fonts.
    pub normalized_coords: Vec<NormalizedCoord>,
    /// Number of encoded paths.
    pub n_paths: u32,
    /// Number of encoded path segments.
    pub n_path_segments: u32,
    /// Number of encoded clips/layers.
    pub n_clips: u32,
    /// Number of unclosed clips/layers.
    pub n_open_clips: u32,
}

impl Encoding {
    /// Creates a new encoding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.path_tags.is_empty()
    }

    /// Clears the encoding.
    pub fn reset(&mut self, is_fragment: bool) {
        self.transforms.clear();
        self.path_tags.clear();
        self.path_data.clear();
        self.linewidths.clear();
        self.draw_data.clear();
        self.draw_tags.clear();
        self.glyphs.clear();
        self.glyph_runs.clear();
        self.normalized_coords.clear();
        self.n_paths = 0;
        self.n_path_segments = 0;
        self.n_clips = 0;
        self.n_open_clips = 0;
        self.patches.clear();
        self.color_stops.clear();
        if !is_fragment {
            self.transforms.push(Transform::IDENTITY);
            self.linewidths.push(-1.0);
        }
    }

    /// Appends another encoding to this one with an optional transform.
    pub fn append(&mut self, other: &Self, transform: &Option<Transform>) {
        let stops_base = self.color_stops.len();
        let glyph_runs_base = self.glyph_runs.len();
        let glyphs_base = self.glyphs.len();
        let coords_base = self.normalized_coords.len();
        let offsets = self.stream_offsets();
        self.path_tags.extend_from_slice(&other.path_tags);
        self.path_data.extend_from_slice(&other.path_data);
        self.draw_tags.extend_from_slice(&other.draw_tags);
        self.draw_data.extend_from_slice(&other.draw_data);
        self.glyphs.extend_from_slice(&other.glyphs);
        self.normalized_coords
            .extend_from_slice(&other.normalized_coords);
        self.glyph_runs
            .extend(other.glyph_runs.iter().cloned().map(|mut run| {
                run.glyphs.start += glyphs_base;
                run.normalized_coords.start += coords_base;
                run.stream_offsets.path_tags += offsets.path_tags;
                run.stream_offsets.path_data += offsets.path_data;
                run.stream_offsets.draw_tags += offsets.draw_tags;
                run.stream_offsets.draw_data += offsets.draw_data;
                run.stream_offsets.transforms += offsets.transforms;
                run.stream_offsets.linewidths += offsets.linewidths;
                run
            }));
        self.n_paths += other.n_paths;
        self.n_path_segments += other.n_path_segments;
        self.n_clips += other.n_clips;
        self.n_open_clips += other.n_open_clips;
        self.patches
            .extend(other.patches.iter().map(|patch| match patch {
                Patch::Ramp {
                    draw_data_offset: offset,
                    stops,
                } => {
                    let stops = stops.start + stops_base..stops.end + stops_base;
                    Patch::Ramp {
                        draw_data_offset: offset + offsets.draw_data,
                        stops,
                    }
                }
                Patch::GlyphRun { index } => Patch::GlyphRun {
                    index: index + glyph_runs_base,
                },
                Patch::Image {
                    image,
                    draw_data_offset,
                } => Patch::Image {
                    image: image.clone(),
                    draw_data_offset: *draw_data_offset + offsets.draw_data,
                },
            }));
        self.color_stops.extend_from_slice(&other.color_stops);
        if let Some(transform) = *transform {
            self.transforms
                .extend(other.transforms.iter().map(|x| transform * *x));
            for run in &mut self.glyph_runs[glyph_runs_base..] {
                run.transform = transform * run.transform;
            }
        } else {
            self.transforms.extend_from_slice(&other.transforms);
        }
        self.linewidths.extend_from_slice(&other.linewidths);
    }

    /// Returns a snapshot of the current stream offsets.
    pub fn stream_offsets(&self) -> StreamOffsets {
        StreamOffsets {
            path_tags: self.path_tags.len(),
            path_data: self.path_data.len(),
            draw_tags: self.draw_tags.len(),
            draw_data: self.draw_data.len(),
            transforms: self.transforms.len(),
            linewidths: self.linewidths.len(),
        }
    }
}

impl Encoding {
    /// Encodes a linewidth.
    pub fn encode_linewidth(&mut self, linewidth: f32) {
        if self.linewidths.last() != Some(&linewidth) {
            self.path_tags.push(PathTag::LINEWIDTH);
            self.linewidths.push(linewidth);
        }
    }

    /// Encodes a transform.
    ///
    /// If the given transform is different from the current one, encodes it and
    /// returns true. Otherwise, encodes nothing and returns false.
    pub fn encode_transform(&mut self, transform: Transform) -> bool {
        if self.transforms.last() != Some(&transform) {
            self.path_tags.push(PathTag::TRANSFORM);
            self.transforms.push(transform);
            true
        } else {
            false
        }
    }

    /// Returns an encoder for encoding a path. If `is_fill` is true, all subpaths will
    /// be automatically closed.
    pub fn encode_path(&mut self, is_fill: bool) -> PathEncoder {
        PathEncoder::new(
            &mut self.path_tags,
            &mut self.path_data,
            &mut self.n_path_segments,
            &mut self.n_paths,
            is_fill,
        )
    }

    /// Encodes a shape. If `is_fill` is true, all subpaths will be automatically closed.
    /// Returns true if a non-zero number of segments were encoded.
    pub fn encode_shape(&mut self, shape: &impl Shape, is_fill: bool) -> bool {
        let mut encoder = self.encode_path(is_fill);
        encoder.shape(shape);
        encoder.finish(true) != 0
    }

    /// Encodes a brush with an optional alpha modifier.
    pub fn encode_brush<'b>(&mut self, brush: impl Into<BrushRef<'b>>, alpha: f32) {
        use super::math::point_to_f32;
        match brush.into() {
            BrushRef::Solid(color) => {
                let color = if alpha != 1.0 {
                    color.with_alpha_factor(alpha)
                } else {
                    color
                };
                self.encode_color(DrawColor::new(color));
            }
            BrushRef::Gradient(gradient) => match gradient.kind {
                GradientKind::Linear { start, end } => {
                    self.encode_linear_gradient(
                        DrawLinearGradient {
                            index: 0,
                            p0: point_to_f32(start),
                            p1: point_to_f32(end),
                        },
                        gradient.stops.iter().copied(),
                        alpha,
                        gradient.extend,
                    );
                }
                GradientKind::Radial {
                    start_center,
                    start_radius,
                    end_center,
                    end_radius,
                } => {
                    self.encode_radial_gradient(
                        DrawRadialGradient {
                            index: 0,
                            p0: point_to_f32(start_center),
                            p1: point_to_f32(end_center),
                            r0: start_radius,
                            r1: end_radius,
                        },
                        gradient.stops.iter().copied(),
                        alpha,
                        gradient.extend,
                    );
                }
                GradientKind::Sweep { .. } => {
                    todo!("sweep gradients aren't supported yet!")
                }
            },
            BrushRef::Image(image) => {
                self.encode_image(image, alpha);
            }
        }
    }

    /// Encodes a solid color brush.
    pub fn encode_color(&mut self, color: DrawColor) {
        self.draw_tags.push(DrawTag::COLOR);
        self.draw_data.extend_from_slice(bytemuck::bytes_of(&color));
    }

    /// Encodes a linear gradient brush.
    pub fn encode_linear_gradient(
        &mut self,
        gradient: DrawLinearGradient,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        _extend: Extend,
    ) {
        self.add_ramp(color_stops, alpha);
        self.draw_tags.push(DrawTag::LINEAR_GRADIENT);
        self.draw_data
            .extend_from_slice(bytemuck::bytes_of(&gradient));
    }

    /// Encodes a radial gradient brush.
    pub fn encode_radial_gradient(
        &mut self,
        gradient: DrawRadialGradient,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        _extend: Extend,
    ) {
        self.add_ramp(color_stops, alpha);
        self.draw_tags.push(DrawTag::RADIAL_GRADIENT);
        self.draw_data
            .extend_from_slice(bytemuck::bytes_of(&gradient));
    }

    /// Encodes an image brush.
    pub fn encode_image(&mut self, image: &Image, _alpha: f32) {
        // TODO: feed the alpha multiplier through the full pipeline for consistency
        // with other brushes?
        self.patches.push(Patch::Image {
            image: image.clone(),
            draw_data_offset: self.draw_data.len(),
        });
        self.draw_tags.push(DrawTag::IMAGE);
        self.draw_data
            .extend_from_slice(bytemuck::bytes_of(&DrawImage {
                xy: 0,
                width_height: (image.width << 16) | (image.height & 0xFFFF),
            }));
    }

    /// Encodes a begin clip command.
    pub fn encode_begin_clip(&mut self, blend_mode: BlendMode, alpha: f32) {
        use super::DrawBeginClip;
        self.draw_tags.push(DrawTag::BEGIN_CLIP);
        self.draw_data
            .extend_from_slice(bytemuck::bytes_of(&DrawBeginClip::new(blend_mode, alpha)));
        self.n_clips += 1;
        self.n_open_clips += 1;
    }

    /// Encodes an end clip command.
    pub fn encode_end_clip(&mut self) {
        if self.n_open_clips > 0 {
            self.draw_tags.push(DrawTag::END_CLIP);
            // This is a dummy path, and will go away with the new clip impl.
            self.path_tags.push(PathTag::PATH);
            self.n_paths += 1;
            self.n_clips += 1;
            self.n_open_clips -= 1;
        }
    }

    // Swap the last two tags in the path tag stream; used for transformed
    // gradients.
    pub fn swap_last_path_tags(&mut self) {
        let len = self.path_tags.len();
        self.path_tags.swap(len - 1, len - 2);
    }

    fn add_ramp(&mut self, color_stops: impl Iterator<Item = ColorStop>, alpha: f32) {
        let offset = self.draw_data.len();
        let stops_start = self.color_stops.len();
        if alpha != 1.0 {
            self.color_stops
                .extend(color_stops.map(|stop| stop.with_alpha_factor(alpha)));
        } else {
            self.color_stops.extend(color_stops);
        }
        self.patches.push(Patch::Ramp {
            draw_data_offset: offset,
            stops: stops_start..self.color_stops.len(),
        });
    }
}

/// Snapshot of offsets for encoded streams.
#[derive(Copy, Clone, Default, Debug)]
pub struct StreamOffsets {
    /// Current length of path tag stream.
    pub path_tags: usize,
    /// Current length of path data stream.
    pub path_data: usize,
    /// Current length of draw tag stream.
    pub draw_tags: usize,
    /// Current length of draw data stream.
    pub draw_data: usize,
    /// Current length of transform stream.
    pub transforms: usize,
    /// Current length of linewidth stream.
    pub linewidths: usize,
}

impl StreamOffsets {
    pub(crate) fn add(&mut self, other: &Self) {
        self.path_tags += other.path_tags;
        self.path_data += other.path_data;
        self.draw_tags += other.draw_tags;
        self.draw_data += other.draw_data;
        self.transforms += other.transforms;
        self.linewidths += other.linewidths;
    }
}
