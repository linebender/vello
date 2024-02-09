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

use peniko::kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, BrushRef, Color, Fill, Font, Image, StyleRef};
use skrifa::instance::NormalizedCoord;
use vello_encoding::{Encoding, Glyph, GlyphRun, Patch, Transform};

/// Encoded definition of a scene and associated resources.
#[derive(Default)]
pub struct Scene {
    data: Encoding,
}

impl Scene {
    /// Creates a new scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the raw encoded scene data streams.
    pub fn data(&self) -> &Encoding {
        &self.data
    }
}

/// Encoded definition of a scene fragment and associated resources.
#[derive(Default)]
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
pub struct SceneBuilder<'a> {
    scene: &'a mut Encoding,
}

impl<'a> SceneBuilder<'a> {
    /// Creates a new builder for filling a scene. Any current content in the scene
    /// will be cleared.
    pub fn for_scene(scene: &'a mut Scene) -> Self {
        Self::new(&mut scene.data, false)
    }

    /// Creates a new builder for filling a scene fragment. Any current content in
    /// the fragment will be cleared.    
    pub fn for_fragment(fragment: &'a mut SceneFragment) -> Self {
        Self::new(&mut fragment.data, true)
    }

    /// Creates a new builder for constructing a scene.
    fn new(scene: &'a mut Encoding, is_fragment: bool) -> Self {
        scene.reset(is_fragment);
        Self { scene }
    }

    /// Pushes a new layer bound by the specified shape and composed with
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
        self.scene.encode_fill_style(Fill::NonZero);
        if !self.scene.encode_shape(shape, true) {
            // If the layer shape is invalid, encode a valid empty path. This suppresses
            // all drawing until the layer is popped.
            self.scene
                .encode_shape(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        }
        self.scene.encode_begin_clip(blend, alpha.clamp(0.0, 1.0));
    }

    /// Pops the current layer.
    pub fn pop_layer(&mut self) {
        self.scene.encode_end_clip();
    }

    /// Fills a shape using the specified style and brush.
    pub fn fill<'b>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<BrushRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene
            .encode_transform(Transform::from_kurbo(&transform));
        self.scene.encode_fill_style(style);
        if self.scene.encode_shape(shape, true) {
            if let Some(brush_transform) = brush_transform {
                if self
                    .scene
                    .encode_transform(Transform::from_kurbo(&(transform * brush_transform)))
                {
                    self.scene.swap_last_path_tags();
                }
            }
            self.scene.encode_brush(brush, 1.0);
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
        // The setting for tolerance are a compromise. For most applications,
        // shape tolerance doesn't matter, as the input is likely Bézier paths,
        // which is exact. Note that shape tolerance is hard-coded as 0.1 in
        // the encoding crate.
        //
        // Stroke tolerance is a different matter. Generally, the cost scales
        // with inverse O(n^6), so there is moderate rendering cost to setting
        // too fine a value. On the other hand, error scales with the transform
        // applied post-stroking, so may exceed visible threshold. When we do
        // GPU-side stroking, the transform will be known. In the meantime,
        // this is a compromise.
        const SHAPE_TOLERANCE: f64 = 0.01;
        const STROKE_TOLERANCE: f64 = SHAPE_TOLERANCE;

        const GPU_STROKES: bool = false; // Set this to `true` to enable GPU-side stroking
        if GPU_STROKES {
            self.scene
                .encode_transform(Transform::from_kurbo(&transform));
            self.scene.encode_stroke_style(style);

            // We currently don't support dashing on the GPU. If the style has a dash pattern, then
            // we convert it into stroked paths on the CPU and encode those as individual draw
            // objects.
            let encode_result = if style.dash_pattern.is_empty() {
                self.scene.encode_shape(shape, false)
            } else {
                let dashed = peniko::kurbo::dash(
                    shape.path_elements(SHAPE_TOLERANCE),
                    style.dash_offset,
                    &style.dash_pattern,
                );
                self.scene.encode_path_elements(dashed, false)
            };
            if encode_result {
                if let Some(brush_transform) = brush_transform {
                    if self
                        .scene
                        .encode_transform(Transform::from_kurbo(&(transform * brush_transform)))
                    {
                        self.scene.swap_last_path_tags();
                    }
                }
                self.scene.encode_brush(brush, 1.0);
            }
        } else {
            let stroked = peniko::kurbo::stroke(
                shape.path_elements(SHAPE_TOLERANCE),
                style,
                &Default::default(),
                STROKE_TOLERANCE,
            );
            self.fill(Fill::NonZero, transform, brush, brush_transform, &stroked);
        }
    }

    /// Draws an image at its natural size with the given transform.
    pub fn draw_image(&mut self, image: &Image, transform: Affine) {
        self.fill(
            Fill::NonZero,
            transform,
            image,
            None,
            &Rect::new(0.0, 0.0, image.width as f64, image.height as f64),
        );
    }

    /// Returns a builder for encoding a glyph run.
    pub fn draw_glyphs(&mut self, font: &Font) -> DrawGlyphs {
        DrawGlyphs::new(self.scene, font)
    }

    /// Appends a fragment to the scene.
    pub fn append(&mut self, fragment: &SceneFragment, transform: Option<Affine>) {
        self.scene.append(
            &fragment.data,
            &transform.map(|xform| Transform::from_kurbo(&xform)),
        );
    }
}

/// Builder for encoding a glyph run.
pub struct DrawGlyphs<'a> {
    encoding: &'a mut Encoding,
    run: GlyphRun,
    brush: BrushRef<'a>,
    brush_alpha: f32,
}

impl<'a> DrawGlyphs<'a> {
    /// Creates a new builder for encoding a glyph run for the specified
    /// encoding with the given font.
    pub fn new(encoding: &'a mut Encoding, font: &Font) -> Self {
        let coords_start = encoding.resources.normalized_coords.len();
        let glyphs_start = encoding.resources.glyphs.len();
        let stream_offsets = encoding.stream_offsets();
        Self {
            encoding,
            run: GlyphRun {
                font: font.clone(),
                transform: Transform::IDENTITY,
                glyph_transform: None,
                font_size: 16.0,
                hint: false,
                normalized_coords: coords_start..coords_start,
                style: Fill::NonZero.into(),
                glyphs: glyphs_start..glyphs_start,
                stream_offsets,
            },
            brush: Color::BLACK.into(),
            brush_alpha: 1.0,
        }
    }

    /// Sets the global transform. This is applied to all glyphs after the offset
    /// translation.
    ///
    /// The default value is the identity matrix.
    pub fn transform(mut self, transform: Affine) -> Self {
        self.run.transform = Transform::from_kurbo(&transform);
        self
    }

    /// Sets the per-glyph transform. This is applied to all glyphs prior to
    /// offset translation. This is common used for applying a shear to simulate
    /// an oblique font.
    ///
    /// The default value is `None`.
    pub fn glyph_transform(mut self, transform: Option<Affine>) -> Self {
        self.run.glyph_transform = transform.map(|xform| Transform::from_kurbo(&xform));
        self
    }

    /// Sets the font size in pixels per em units.
    ///
    /// The default value is 16.0.
    pub fn font_size(mut self, size: f32) -> Self {
        self.run.font_size = size;
        self
    }

    /// Sets whether to enable hinting.
    ///
    /// The default value is `false`.
    pub fn hint(mut self, hint: bool) -> Self {
        self.run.hint = hint;
        self
    }

    /// Sets the normalized design space coordinates for a variable font instance.
    pub fn normalized_coords(mut self, coords: &[NormalizedCoord]) -> Self {
        self.encoding
            .resources
            .normalized_coords
            .truncate(self.run.normalized_coords.start);
        self.encoding
            .resources
            .normalized_coords
            .extend_from_slice(coords);
        self.run.normalized_coords.end = self.encoding.resources.normalized_coords.len();
        self
    }

    /// Sets the brush.
    ///
    /// The default value is solid black.
    pub fn brush(mut self, brush: impl Into<BrushRef<'a>>) -> Self {
        self.brush = brush.into();
        self
    }

    /// Sets an additional alpha multiplier for the brush.
    ///
    /// The default value is 1.0.
    pub fn brush_alpha(mut self, alpha: f32) -> Self {
        self.brush_alpha = alpha;
        self
    }

    /// Encodes a fill or stroke for for the given sequence of glyphs and consumes
    /// the builder.
    ///
    /// The `style` parameter accepts either `Fill` or `&Stroke` types.
    pub fn draw(mut self, style: impl Into<StyleRef<'a>>, glyphs: impl Iterator<Item = Glyph>) {
        let resources = &mut self.encoding.resources;
        self.run.style = style.into().to_owned();
        resources.glyphs.extend(glyphs);
        self.run.glyphs.end = resources.glyphs.len();
        if self.run.glyphs.is_empty() {
            resources
                .normalized_coords
                .truncate(self.run.normalized_coords.start);
            return;
        }
        let index = resources.glyph_runs.len();
        resources.glyph_runs.push(self.run);
        resources.patches.push(Patch::GlyphRun { index });
        self.encoding.encode_brush(self.brush, self.brush_alpha);
        // Glyph run resolve step affects transform and style state in a way
        // that is opaque to the current encoding.
        // See <https://github.com/linebender/vello/issues/424>
        self.encoding.force_next_transform_and_style();
    }
}
