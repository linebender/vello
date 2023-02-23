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

use super::resource::Patch;
use super::{
    DrawColor, DrawLinearGradient, DrawRadialGradient, DrawTag, PathEncoder, PathTag, Transform,
};

use peniko::{kurbo::Shape, BlendMode, BrushRef, Color, ColorStop, Extend, GradientKind};

/// Encoded data streams for a scene.
#[derive(Default)]
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
    /// Number of encoded paths.
    pub n_paths: u32,
    /// Number of encoded path segments.
    pub n_path_segments: u32,
    /// Number of encoded clips/layers.
    pub n_clips: u32,
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
        self.n_paths = 0;
        self.n_path_segments = 0;
        self.n_clips = 0;
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
        let draw_data_base = self.draw_data.len();
        self.path_tags.extend_from_slice(&other.path_tags);
        self.path_data.extend_from_slice(&other.path_data);
        self.draw_tags.extend_from_slice(&other.draw_tags);
        self.draw_data.extend_from_slice(&other.draw_data);
        self.n_paths += other.n_paths;
        self.n_path_segments += other.n_path_segments;
        self.n_clips += other.n_clips;
        self.patches
            .extend(other.patches.iter().map(|patch| match patch {
                Patch::Ramp { offset, stops } => {
                    let stops = stops.start + stops_base..stops.end + stops_base;
                    Patch::Ramp {
                        offset: draw_data_base + offset,
                        stops,
                    }
                }
            }));
        self.color_stops.extend_from_slice(&other.color_stops);
        if let Some(transform) = *transform {
            self.transforms
                .extend(other.transforms.iter().map(|x| transform * *x));
        } else {
            self.transforms.extend_from_slice(&other.transforms);
        }
        self.linewidths.extend_from_slice(&other.linewidths);
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
    /// If the given transform is different from the current one, encodes it an
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
                    color_with_alpha(color, alpha)
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
            BrushRef::Image(_) => {
                todo!("images aren't supported yet!")
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

    /// Encodes a begin clip command.
    pub fn encode_begin_clip(&mut self, blend_mode: BlendMode, alpha: f32) {
        use super::DrawBeginClip;
        self.draw_tags.push(DrawTag::BEGIN_CLIP);
        self.draw_data
            .extend_from_slice(bytemuck::bytes_of(&DrawBeginClip::new(blend_mode, alpha)));
        self.n_clips += 1;
    }

    /// Encodes an end clip command.
    pub fn encode_end_clip(&mut self) {
        self.draw_tags.push(DrawTag::END_CLIP);
        // This is a dummy path, and will go away with the new clip impl.
        self.path_tags.push(PathTag::PATH);
        self.n_paths += 1;
        self.n_clips += 1;
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
            self.color_stops.extend(color_stops.map(|s| ColorStop {
                offset: s.offset,
                color: color_with_alpha(s.color, alpha),
            }));
        } else {
            self.color_stops.extend(color_stops);
        }
        self.patches.push(Patch::Ramp {
            offset,
            stops: stops_start..self.color_stops.len(),
        });
    }
}

fn color_with_alpha(mut color: Color, alpha: f32) -> Color {
    color.a = ((color.a as f32) * alpha) as u8;
    color
}
