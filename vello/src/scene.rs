// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::{
    kurbo::{Affine, BezPath, Point, Rect, Shape, Stroke, Vec2},
    BlendMode, Brush, BrushRef, Color, ColorStop, ColorStops, ColorStopsSource, Compose, Extend,
    Fill, Font, Gradient, Image, Mix, StyleRef,
};
use skrifa::{
    color::ColorPainter,
    instance::{LocationRef, NormalizedCoord},
    outline::{DrawSettings, OutlinePen},
    raw::{tables::cpal::Cpal, TableProvider},
    GlyphId, MetadataProvider, OutlineGlyphCollection,
};
#[cfg(feature = "bump_estimate")]
use vello_encoding::BumpAllocatorMemory;
use vello_encoding::{Encoding, Glyph, GlyphRun, Patch, Transform};

// TODO - Document invariants and edge cases (#470)
// - What happens when we pass a transform matrix with NaN values to the Scene?
// - What happens if a push_layer isn't matched by a pop_layer?

/// The main datatype for rendering graphics.
///
/// A Scene stores a sequence of drawing commands, their context, and the
/// associated resources, which can later be rendered.
#[derive(Clone, Default)]
pub struct Scene {
    encoding: Encoding,
    #[cfg(feature = "bump_estimate")]
    estimator: vello_encoding::BumpEstimator,
}
static_assertions::assert_impl_all!(Scene: Send, Sync);

impl Scene {
    /// Creates a new scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes all content from the scene.
    pub fn reset(&mut self) {
        self.encoding.reset();
        #[cfg(feature = "bump_estimate")]
        self.estimator.reset();
    }

    /// Tally up the bump allocator estimate for the current state of the encoding,
    /// taking into account an optional `transform` applied to the entire scene.
    #[cfg(feature = "bump_estimate")]
    pub fn bump_estimate(&self, transform: Option<Affine>) -> BumpAllocatorMemory {
        self.estimator
            .tally(transform.as_ref().map(Transform::from_kurbo).as_ref())
    }

    /// Returns the underlying raw encoding.
    pub fn encoding(&self) -> &Encoding {
        &self.encoding
    }

    /// Pushes a new layer clipped by the specified shape and composed with
    /// previous layers using the specified blend mode.
    ///
    /// Every drawing command after this call will be clipped by the shape
    /// until the layer is popped.
    ///
    /// **However, the transforms are *not* saved or modified by the layer stack.**
    pub fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let blend = blend.into();
        let t = Transform::from_kurbo(&transform);
        self.encoding.encode_transform(t);
        self.encoding.encode_fill_style(Fill::NonZero);
        if !self.encoding.encode_shape(clip, true) {
            // If the layer shape is invalid, encode a valid empty path. This suppresses
            // all drawing until the layer is popped.
            self.encoding
                .encode_shape(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        } else {
            #[cfg(feature = "bump_estimate")]
            self.estimator.count_path(clip.path_elements(0.1), &t, None);
        }
        self.encoding
            .encode_begin_clip(blend, alpha.clamp(0.0, 1.0));
    }

    /// Pops the current layer.
    pub fn pop_layer(&mut self) {
        self.encoding.encode_end_clip();
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
        let t = Transform::from_kurbo(&transform);
        self.encoding.encode_transform(t);
        self.encoding.encode_fill_style(style);
        if self.encoding.encode_shape(shape, true) {
            if let Some(brush_transform) = brush_transform {
                if self
                    .encoding
                    .encode_transform(Transform::from_kurbo(&(transform * brush_transform)))
                {
                    self.encoding.swap_last_path_tags();
                }
            }
            self.encoding.encode_brush(brush, 1.0);
            #[cfg(feature = "bump_estimate")]
            self.estimator
                .count_path(shape.path_elements(0.1), &t, None);
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

        const GPU_STROKES: bool = true; // Set this to `true` to enable GPU-side stroking
        if GPU_STROKES {
            let t = Transform::from_kurbo(&transform);
            self.encoding.encode_transform(t);
            self.encoding.encode_stroke_style(style);

            // We currently don't support dashing on the GPU. If the style has a dash pattern, then
            // we convert it into stroked paths on the CPU and encode those as individual draw
            // objects.
            let encode_result = if style.dash_pattern.is_empty() {
                #[cfg(feature = "bump_estimate")]
                self.estimator
                    .count_path(shape.path_elements(SHAPE_TOLERANCE), &t, Some(style));
                self.encoding.encode_shape(shape, false)
            } else {
                // TODO: We currently collect the output of the dash iterator because
                // `encode_path_elements` wants to consume the iterator. We want to avoid calling
                // `dash` twice when `bump_estimate` is enabled because it internally allocates.
                // Bump estimation will move to resolve time rather than scene construction time,
                // so we can revert this back to not collecting when that happens.
                let dashed = peniko::kurbo::dash(
                    shape.path_elements(SHAPE_TOLERANCE),
                    style.dash_offset,
                    &style.dash_pattern,
                )
                .collect::<Vec<_>>();
                #[cfg(feature = "bump_estimate")]
                self.estimator
                    .count_path(dashed.iter().copied(), &t, Some(style));
                self.encoding
                    .encode_path_elements(dashed.into_iter(), false)
            };
            if encode_result {
                if let Some(brush_transform) = brush_transform {
                    if self
                        .encoding
                        .encode_transform(Transform::from_kurbo(&(transform * brush_transform)))
                    {
                        self.encoding.swap_last_path_tags();
                    }
                }
                self.encoding.encode_brush(brush, 1.0);
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
        // TODO: Integrate `BumpEstimator` with the glyph cache.
        DrawGlyphs::new(self, font)
    }

    /// Appends a child scene.
    ///
    /// The given transform is applied to every transform in the child.
    /// This is an O(N) operation.
    pub fn append(&mut self, other: &Scene, transform: Option<Affine>) {
        let t = transform.as_ref().map(Transform::from_kurbo);
        self.encoding.append(&other.encoding, &t);
        #[cfg(feature = "bump_estimate")]
        self.estimator.append(&other.estimator, t.as_ref());
    }
}

impl From<Encoding> for Scene {
    fn from(encoding: Encoding) -> Self {
        // It's fine to create a default estimator here, and that field will be
        // removed at some point - see https://github.com/linebender/vello/issues/541
        Self {
            encoding,
            #[cfg(feature = "bump_estimate")]
            estimator: vello_encoding::BumpEstimator::default(),
        }
    }
}

/// Builder for encoding a glyph run.
pub struct DrawGlyphs<'a> {
    scene: &'a mut Scene,
    run: GlyphRun,
    brush: BrushRef<'a>,
    brush_alpha: f32,
}

impl<'a> DrawGlyphs<'a> {
    /// Creates a new builder for encoding a glyph run for the specified
    /// encoding with the given font.
    pub fn new(scene: &'a mut Scene, font: &Font) -> Self {
        let coords_start = scene.encoding.resources.normalized_coords.len();
        let glyphs_start = scene.encoding.resources.glyphs.len();
        let stream_offsets = scene.encoding.stream_offsets();
        Self {
            scene,
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
        self.scene
            .encoding
            .resources
            .normalized_coords
            .truncate(self.run.normalized_coords.start);
        self.scene
            .encoding
            .resources
            .normalized_coords
            .extend_from_slice(coords);
        self.run.normalized_coords.end = self.scene.encoding.resources.normalized_coords.len();
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

    /// Encodes a fill or stroke for the given sequence of glyphs and consumes the builder.
    ///
    /// The `style` parameter accepts either `Fill` or `&Stroke` types.
    ///
    /// If the font has COLR support, it will try to draw each glyph using that table first,
    /// falling back to non-COLR rendering. `style` is ignored for COLR glyphs.
    ///
    /// For these glyphs, the given [brush](Self::brush) is used as the "foreground colour", and should
    /// be [`Solid`](Brush::Solid) for maximum compatibility.
    pub fn draw(mut self, style: impl Into<StyleRef<'a>>, glyphs: impl Iterator<Item = Glyph>) {
        let font_index = self.run.font.index;
        let font = skrifa::FontRef::from_index(self.run.font.data.as_ref(), font_index).unwrap();
        if font.colr().is_ok() && font.cpal().is_ok() {
            self.try_draw_colr(style.into(), glyphs);
        } else {
            // Shortcut path - no need to test each glyph for a colr outline
            let outline_count = self.draw_outline_glyphs(style, glyphs);
            if outline_count == 0 {
                self.scene
                    .encoding
                    .resources
                    .normalized_coords
                    .truncate(self.run.normalized_coords.start);
            }
        }
    }

    fn draw_outline_glyphs(
        &mut self,
        style: impl Into<StyleRef<'a>>,
        glyphs: impl Iterator<Item = Glyph>,
    ) -> usize {
        let resources = &mut self.scene.encoding.resources;
        self.run.style = style.into().to_owned();
        resources.glyphs.extend(glyphs);
        self.run.glyphs.end = resources.glyphs.len();
        if self.run.glyphs.is_empty() {
            return 0;
        }
        let index = resources.glyph_runs.len();
        resources.glyph_runs.push(self.run.clone());
        resources.patches.push(Patch::GlyphRun { index });
        self.scene
            .encoding
            .encode_brush(self.brush.clone(), self.brush_alpha);
        // Glyph run resolve step affects transform and style state in a way
        // that is opaque to the current encoding.
        // See <https://github.com/linebender/vello/issues/424>
        self.scene.encoding.force_next_transform_and_style();
        self.run.glyphs.len()
    }

    fn try_draw_colr(&mut self, style: StyleRef<'a>, mut glyphs: impl Iterator<Item = Glyph>) {
        let font_index = self.run.font.index;
        let blob = &self.run.font.data.clone();
        let font = skrifa::FontRef::from_index(blob.as_ref(), font_index).unwrap();
        let upem: f32 = font.head().map(|h| h.units_per_em()).unwrap().into();
        let run_transform = self.run.transform.to_kurbo();
        let scale = Affine::scale_non_uniform(
            (self.run.font_size / upem).into(),
            (-self.run.font_size / upem).into(),
        );
        let colour_collection = font.color_glyphs();
        let mut final_glyph = None;
        let mut outline_count = 0;
        // We copy out of the variable font coords here because we need to call an exclusive self method
        let coords = &self.scene.encoding.resources.normalized_coords
            [self.run.normalized_coords.clone()]
        .to_vec();
        let location = LocationRef::new(coords);
        loop {
            let outline_glyphs = (&mut glyphs).take_while(|glyph| {
                match colour_collection.get(GlyphId::new(glyph.id.try_into().unwrap())) {
                    Some(color) => {
                        final_glyph = Some((color, *glyph));
                        false
                    }
                    None => true,
                }
            });
            self.run.glyphs.start = self.run.glyphs.end;
            self.run.stream_offsets = self.scene.encoding.stream_offsets();
            outline_count += self.draw_outline_glyphs(clone_style_ref(&style), outline_glyphs);

            let Some((color, glyph)) = final_glyph.take() else {
                // All of the remaining glyphs were outline glyphs
                break;
            };

            let transform = run_transform
                * Affine::translate(Vec2::new(glyph.x.into(), glyph.y.into()))
                * scale
                * self
                    .run
                    .glyph_transform
                    .unwrap_or(Transform::IDENTITY)
                    .to_kurbo();

            color
                .paint(
                    location,
                    &mut DrawColorGlyphs {
                        scene: self.scene,
                        cpal: &font.cpal().unwrap(),
                        outlines: &font.outline_glyphs(),
                        transform_stack: vec![Transform::from_kurbo(&transform)],
                        clip_box: DEFAULT_CLIP_RECT,
                        clip_depth: 0,
                        location,
                        foreground_brush: self.brush.clone(),
                    },
                )
                .unwrap();
        }
        if outline_count == 0 {
            // If we didn't draw any outline glyphs, the encoded variable font parameters were never used
            // Therefore, we can safely discard them.
            self.scene
                .encoding
                .resources
                .normalized_coords
                .truncate(self.run.normalized_coords.start);
        }
    }
}
const BOUND: f64 = 100_000.;
// Hack: If we don't have a clip box, we guess a rectangle we hope is big enough
const DEFAULT_CLIP_RECT: Rect = Rect::new(-BOUND, -BOUND, BOUND, BOUND);

/// An adapter from [`Scene`] to [`ColorPainter`].
struct DrawColorGlyphs<'a> {
    scene: &'a mut Scene,
    transform_stack: Vec<Transform>,
    cpal: &'a Cpal<'a>,
    outlines: &'a OutlineGlyphCollection<'a>,
    clip_box: Rect,
    clip_depth: u32,
    location: LocationRef<'a>,
    foreground_brush: BrushRef<'a>,
}

impl ColorPainter for DrawColorGlyphs<'_> {
    fn push_transform(&mut self, transform: skrifa::color::Transform) {
        let transform = conv_skrifa_transform(transform);
        let prior_transform = self.last_transform();
        self.transform_stack.push(prior_transform * transform);
    }

    fn pop_transform(&mut self) {
        self.transform_stack.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let Some(outline) = self.outlines.get(glyph_id) else {
            eprintln!("Didn't get expected outline");
            return;
        };

        let mut path = BezPathOutline(BezPath::new());
        let draw_settings =
            DrawSettings::unhinted(skrifa::instance::Size::unscaled(), self.location);

        let Ok(_) = outline.draw(draw_settings, &mut path) else {
            return;
        };
        self.clip_depth += 1;
        self.scene
            .push_layer(Mix::Clip, 1.0, self.last_transform().to_kurbo(), &path.0);
    }

    fn push_clip_box(&mut self, clip_box: skrifa::raw::types::BoundingBox<f32>) {
        let clip_box = Rect::new(
            clip_box.x_min.into(),
            clip_box.y_min.into(),
            clip_box.x_max.into(),
            clip_box.y_max.into(),
        );
        if self.clip_depth == 0 {
            self.clip_box = clip_box;
        }
        self.clip_depth += 1;
        self.scene
            .push_layer(Mix::Clip, 1.0, self.last_transform().to_kurbo(), &clip_box);
    }

    fn pop_clip(&mut self) {
        self.scene.pop_layer();
        self.clip_depth -= 1;
        if self.clip_depth == 0 {
            self.clip_box = DEFAULT_CLIP_RECT;
        }
    }

    fn fill(&mut self, brush: skrifa::color::Brush<'_>) {
        let brush = conv_brush(brush, self.cpal, &self.foreground_brush);
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            &brush,
            Some(self.last_transform().to_kurbo()),
            &self.clip_box,
        );
    }

    fn push_layer(&mut self, composite: skrifa::color::CompositeMode) {
        let blend = match composite {
            skrifa::color::CompositeMode::Clear => Compose::Clear,
            skrifa::color::CompositeMode::Src => Compose::Copy,
            skrifa::color::CompositeMode::Dest => Compose::Dest,
            skrifa::color::CompositeMode::SrcOver => Compose::SrcOver,
            skrifa::color::CompositeMode::DestOver => Compose::DestOver,
            skrifa::color::CompositeMode::SrcIn => Compose::SrcIn,
            skrifa::color::CompositeMode::DestIn => Compose::DestIn,
            skrifa::color::CompositeMode::SrcOut => Compose::SrcOut,
            skrifa::color::CompositeMode::DestOut => Compose::DestOut,
            skrifa::color::CompositeMode::SrcAtop => Compose::SrcAtop,
            skrifa::color::CompositeMode::DestAtop => Compose::DestAtop,
            skrifa::color::CompositeMode::Xor => Compose::Xor,
            skrifa::color::CompositeMode::Plus => Compose::Plus,
            // TODO:
            _ => Compose::SrcOver,
        };
        self.scene
            .push_layer(blend, 1.0, self.last_transform().to_kurbo(), &self.clip_box);
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn fill_glyph(
        &mut self,
        glyph_id: GlyphId,
        brush_transform: Option<skrifa::color::Transform>,
        brush: skrifa::color::Brush<'_>,
    ) {
        let Some(outline) = self.outlines.get(glyph_id) else {
            eprintln!("Didn't get expected outline");
            return;
        };

        let mut path = BezPathOutline(BezPath::new());
        let draw_settings =
            DrawSettings::unhinted(skrifa::instance::Size::unscaled(), self.location);

        let Ok(_) = outline.draw(draw_settings, &mut path) else {
            return;
        };

        let transform = self.last_transform();
        self.scene.fill(
            Fill::NonZero,
            transform.to_kurbo(),
            &conv_brush(brush, self.cpal, &self.foreground_brush),
            brush_transform
                .map(conv_skrifa_transform)
                .map(|it| it.to_kurbo()),
            &path.0,
        );
    }
}

// TODO: Move this into Peniko
fn clone_style_ref<'first>(first: &StyleRef<'first>) -> StyleRef<'first> {
    match first {
        StyleRef::Fill(fill) => StyleRef::Fill(*fill),
        StyleRef::Stroke(stroke) => StyleRef::Stroke(stroke),
    }
}

struct BezPathOutline(BezPath);

impl OutlinePen for BezPathOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(Point::new(x.into(), y.into()));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(Point::new(x.into(), y.into()));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0.quad_to(
            Point::new(cx0.into(), cy0.into()),
            Point::new(x.into(), y.into()),
        );
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            Point::new(cx0.into(), cy0.into()),
            Point::new(cx1.into(), cy1.into()),
            Point::new(x.into(), y.into()),
        );
    }

    fn close(&mut self) {
        self.0.close_path();
    }
}

impl DrawColorGlyphs<'_> {
    fn last_transform(&self) -> Transform {
        self.transform_stack
            .last()
            .copied()
            .unwrap_or(Transform::IDENTITY)
    }
}

fn conv_skrifa_transform(transform: skrifa::color::Transform) -> Transform {
    Transform {
        matrix: [transform.xx, transform.xy, transform.yx, transform.yy],
        translation: [transform.dx, transform.dy],
    }
}

fn conv_brush(
    brush: skrifa::color::Brush,
    cpal: &Cpal<'_>,
    foreground_brush: &BrushRef<'_>,
) -> Brush {
    match brush {
        skrifa::color::Brush::Solid {
            palette_index,
            alpha,
        } => color_index(cpal, palette_index)
            .map(|it| Brush::Solid(it.with_alpha_factor(alpha)))
            .unwrap_or(apply_alpha(foreground_brush.to_owned(), alpha)),

        skrifa::color::Brush::LinearGradient {
            p0,
            p1,
            color_stops,
            extend,
        } => Brush::Gradient(
            Gradient::new_linear(conv_point(p0), conv_point(p1))
                .with_extend(conv_extend(extend))
                .with_stops(ColorStopsConverter(color_stops, cpal, foreground_brush)),
        ),
        skrifa::color::Brush::RadialGradient {
            c0,
            r0,
            c1,
            r1,
            color_stops,
            extend,
        } => Brush::Gradient(
            Gradient::new_two_point_radial(conv_point(c0), r0, conv_point(c1), r1)
                .with_extend(conv_extend(extend))
                .with_stops(ColorStopsConverter(color_stops, cpal, foreground_brush)),
        ),
        skrifa::color::Brush::SweepGradient {
            c0,
            start_angle,
            end_angle,
            color_stops,
            extend,
        } => Brush::Gradient(
            Gradient::new_sweep(conv_point(c0), start_angle, end_angle)
                .with_extend(conv_extend(extend))
                .with_stops(ColorStopsConverter(color_stops, cpal, foreground_brush)),
        ),
    }
}

fn apply_alpha(mut brush: Brush, alpha: f32) -> Brush {
    match &mut brush {
        Brush::Solid(color) => *color = color.with_alpha_factor(alpha),
        Brush::Gradient(grad) => grad
            .stops
            .iter_mut()
            .for_each(|it| it.color = it.color.with_alpha_factor(alpha)),
        // Cannot apply an alpha factor to
        Brush::Image(_) => {}
    }
    brush
}

fn color_index(cpal: &'_ Cpal<'_>, palette_index: u16) -> Option<Color> {
    // The "application determined" foreground colour should be used
    // This will be handled by the caller
    if palette_index == 0xFFFF {
        return None;
    }
    let actual_colors = cpal.color_records_array().unwrap().unwrap();
    // TODO: Error reporting in the `None` case
    let color = actual_colors.get(usize::from(palette_index))?;
    Some(Color::rgba8(
        color.red,
        color.green,
        color.blue,
        color.alpha,
    ))
}

fn conv_point(point: skrifa::raw::types::Point<f32>) -> Point {
    Point::new(point.x.into(), point.y.into())
}

fn conv_extend(extend: skrifa::color::Extend) -> Extend {
    match extend {
        skrifa::color::Extend::Pad => Extend::Pad,
        skrifa::color::Extend::Repeat => Extend::Reflect,
        skrifa::color::Extend::Reflect => Extend::Repeat,
        // TODO: Error reporting on unknown variant?
        _ => Extend::Pad,
    }
}

struct ColorStopsConverter<'a>(
    &'a [skrifa::color::ColorStop],
    &'a Cpal<'a>,
    &'a BrushRef<'a>,
);

impl ColorStopsSource for ColorStopsConverter<'_> {
    fn collect_stops(&self, vec: &mut ColorStops) {
        for item in self.0 {
            let color = color_index(self.1, item.palette_index);
            let color = match color {
                Some(color) => color,
                // If we should use the "application defined fallback colour",
                // then *try* and determine that from the existing brush
                None => match self.2 {
                    BrushRef::Solid(c) => *c,
                    // TODO: Report a warning? if either of these cases are reached
                    // In theory, it's possible to have a gradient containing images and other gradients
                    // but implementing that just for this case isn't worthwhile
                    BrushRef::Gradient(grad) => grad
                        .stops
                        .first()
                        .map(|it| it.color)
                        .unwrap_or(Color::TRANSPARENT),
                    BrushRef::Image(_) => Color::BLACK,
                },
            };
            let color = color.with_alpha_factor(item.alpha);
            vec.push(ColorStop {
                color,
                offset: item.offset,
            });
        }
    }
}
