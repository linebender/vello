// Copyright 2022 The vello authors.
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

use peniko::kurbo::{Affine, Rect, Shape};
use peniko::{BlendMode, BrushRef, Fill, Stroke};

use crate::encoding::{Encoding, Transform};

/// Encoded definition of a scene and associated resources.
#[derive(Default, Clone)]
pub struct Scene {
    data: Encoding,
}

impl Scene {
    /// Creates a new scene.
    pub fn new() -> Self {
        let mut this = Self::default();
        this.data.reset(false);
        this
    }

    /// Returns the raw encoded scene data streams.
    pub fn data(&self) -> &Encoding {
        &self.data
    }
    
    pub fn append(&mut self, fragment: &SceneFragment, transform: Option<Affine>) {
        self.data.append(
            &fragment.data,
            &transform.map(|a| Transform::from_kurbo(&a))
        );
    }
}

/// Encoded definition of a scene fragment and associated resources.
#[derive(Default, Clone)]
pub struct SceneFragment {
    data: Encoding,
}

impl SceneFragment {
    /// Creates a new scene fragment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the fragment does not contain any paths.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the the entire sequence of points in the scene fragment.
    pub fn points(&self) -> &[[f32; 2]] {
        if self.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(&self.data.path_data)
        }
    }
}

/// Builder for constructing a scene or scene fragment.
#[derive(Clone)]
pub struct FragmentBuilder {
    scene: Encoding,
    layer_depth: u32,
}

impl FragmentBuilder {
    /// Creates a new builder for constructing a scene.
    pub fn new() -> Self {
        Self {
            scene: Encoding::new(),
            layer_depth: 0,
        }
    }

    /// Pushes a new layer bound by the specifed shape and composed with
    /// previous layers using the specified blend mode.
    pub fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        shape: &impl Shape,
    ) {
        let blend = blend.into();
        self.scene
            .encode_transform(Transform::from_kurbo(&transform));
        self.scene.encode_linewidth(-1.0);
        if !self.scene.encode_shape(shape, true) {
            // If the layer shape is invalid, encode a valid empty path. This suppresses
            // all drawing until the layer is popped.
            self.scene
                .encode_shape(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        }
        self.scene.encode_begin_clip(blend, alpha.clamp(0.0, 1.0));
        self.layer_depth += 1;
    }

    /// Pops the current layer.
    pub fn pop_layer(&mut self) {
        if self.layer_depth > 0 {
            self.scene.encode_end_clip();
            self.layer_depth -= 1;
        }
    }

    /// Fills a shape using the specified style and brush.
    pub fn fill<'b>(
        &mut self,
        _style: Fill,
        transform: Affine,
        brush: impl Into<BrushRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene
            .encode_transform(Transform::from_kurbo(&transform));
        self.scene.encode_linewidth(-1.0);
        if self.scene.encode_shape(shape, true) {
            if let Some(brush_transform) = brush_transform {
                self.scene
                    .encode_transform(Transform::from_kurbo(&(transform * brush_transform)));
                self.scene.swap_last_path_tags();
                self.scene.encode_brush(brush, 1.0);
            } else {
                self.scene.encode_brush(brush, 1.0);
            }
        }
    }

    /// Strokes a shape using the specified style and brush.
    pub fn stroke<'b>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<BrushRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene
            .encode_transform(Transform::from_kurbo(&transform));
        self.scene.encode_linewidth(style.width);
        if self.scene.encode_shape(shape, false) {
            if let Some(brush_transform) = brush_transform {
                self.scene
                    .encode_transform(Transform::from_kurbo(&(transform * brush_transform)));
                self.scene.swap_last_path_tags();
                self.scene.encode_brush(brush, 1.0);
            } else {
                self.scene.encode_brush(brush, 1.0);
            }
        }
    }

    /// Appends a fragment to the scene.
    pub fn append(&mut self, fragment: &SceneFragment, transform: Option<Affine>) {
        self.scene.append(
            &fragment.data,
            &transform.map(|xform| Transform::from_kurbo(&xform)),
        );
    }

    /// Completes construction and finalizes the underlying scene.
    pub fn finish(mut self) -> SceneFragment {
        for _ in 0..self.layer_depth {
            self.scene.encode_end_clip();
        }
        
        SceneFragment {
            data: self.scene,
        }
    }
}
