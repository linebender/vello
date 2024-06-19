use peniko::{kurbo::Rect, Brush, Color};
use skrifa::{
    color::ColorPainter, instance::Size, outline::DrawSettings, raw::tables::cpal::Cpal,
    OutlineGlyph, OutlineGlyphCollection,
};

use crate::{Encoding, Transform};

pub(crate) struct GlyphStackEntry<'a> {
    // glyph: GlyphId,
    outline: OutlineGlyph<'a>,
    transform: Transform,
}

pub(crate) struct EncodeColorGlyph<'a> {
    pub(crate) encoding: &'a mut Encoding,
    pub(crate) outlines: &'a OutlineGlyphCollection<'a>,
    /// A stack of clip and layers which are used to hack around the different imaging models
    ///
    /// Skrifa assumes that we can create clipping and layers separately, which is not the case
    pub(crate) glyph_stack: Vec<GlyphStackEntry<'a>>,
    pub(crate) transforms: Vec<Transform>,
    pub(crate) cpal: Cpal<'a>,
}

impl<'a> ColorPainter for EncodeColorGlyph<'a> {
    fn push_transform(&mut self, transform: skrifa::color::Transform) {
        let transform = Transform {
            matrix: [transform.xx, transform.xy, transform.yx, transform.yy],
            translation: [transform.dx, transform.dy],
        };
        let last_transform = self
            .transforms
            .last()
            .copied()
            .unwrap_or(Transform::IDENTITY);
        let transform = last_transform * transform;
        self.transforms.push(transform);
        self.encoding.encode_transform(transform);
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
        self.encoding.encode_transform(
            self.transforms
                .last()
                .copied()
                .unwrap_or(Transform::IDENTITY),
        );
    }

    fn push_clip_glyph(&mut self, glyph_id: skrifa::GlyphId) {
        let Some(outline) = self.outlines.get(glyph_id) else {
            eprintln!("Didn't get expected outline");
            return;
        };
        let mut path = self.encoding.encode_path(true);
        let draw_settings = DrawSettings::unhinted(Size::unscaled(), [].as_slice());

        let Ok(_) = outline.draw(draw_settings, &mut path) else {
            return;
        };

        if path.finish(false) == 0 {
            // If the layer shape is invalid, encode a valid empty path. This suppresses
            // all drawing until the layer is popped.
            self.encoding
                .encode_shape(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        }
        self.encoding
            .encode_begin_clip(peniko::Mix::Clip.into(), 1.0);
        self.glyph_stack.push(GlyphStackEntry {
            // glyph: glyph_id,
            outline,
            transform: self
                .transforms
                .last()
                .copied()
                .unwrap_or(Transform::IDENTITY),
        });
    }

    fn push_clip_box(&mut self, clip_box: skrifa::raw::types::BoundingBox<f32>) {
        self.encoding.encode_shape(
            &Rect::new(
                clip_box.x_min.into(),
                clip_box.y_min.into(),
                clip_box.x_max.into(),
                clip_box.y_max.into(),
            ),
            true,
        );
        self.encoding
            .encode_begin_clip(peniko::Mix::Clip.into(), 1.0);
    }

    fn pop_clip(&mut self) {
        self.encoding.encode_end_clip()
    }

    fn fill(&mut self, brush: skrifa::color::Brush<'_>) {
        match brush {
            skrifa::color::Brush::Solid {
                palette_index,
                alpha,
            } => {
                if let Some(last) = self.glyph_stack.last() {
                    self.encoding.encode_transform(last.transform);
                    self.encoding.encode_fill_style(peniko::Fill::NonZero);
                    let mut path = self.encoding.encode_path(true);
                    let draw_settings = DrawSettings::unhinted(Size::unscaled(), [].as_slice());
                    last.outline.draw(draw_settings, &mut path).unwrap();
                    let prior_transform = self
                        .transforms
                        .last()
                        .copied()
                        .unwrap_or(Transform::IDENTITY);
                    if path.finish(true) != 0 {
                        if self.encoding.encode_transform(prior_transform) {
                            self.encoding.swap_last_path_tags();
                        }
                        let color_index =
                            self.cpal.color_record_indices()[usize::from(palette_index)];
                        let actual_colors = self.cpal.color_records_array().unwrap().unwrap();
                        let color = actual_colors[usize::from(color_index.get())];
                        let brush = Brush::Solid(Color::rgba8(
                            color.red,
                            color.green,
                            color.blue,
                            color.alpha,
                        ));
                        self.encoding.encode_brush(&brush, alpha)
                    } else {
                        self.encoding.encode_transform(prior_transform);
                    }
                }
            }
            e => eprintln!("Unsupported brush {e:?}"),
        }
    }

    fn push_layer(&mut self, _composite_mode: skrifa::color::CompositeMode) {
        // if let Some(last) = self.glyph_stack.last() {
        //     let mut path = self.encoding.encode_path(true);
        //     let draw_settings = DrawSettings::unhinted(Size::unscaled(), [].as_slice());

        //     let Ok(_) = last.outline.draw(draw_settings, &mut path) else {
        //         return;
        //     };

        //     if path.finish(false) == 0 {
        //         // If the layer shape is invalid, encode a valid empty path. This suppresses
        //         // all drawing until the layer is popped.
        //         self.encoding
        //             .encode_shape(&Rect::new(0.0, 0.0, 0.0, 0.0), true);
        //     }
        //     // TODO: Optimise
        //     self.encoding
        //         .encode_begin_clip(peniko::Mix::Clip.into(), 1.0);
        // }
    }

    fn pop_layer(&mut self) {
        // if !self.glyph_stack.is_empty() {
        //     self.encoding.encode_end_clip();
        // }
    }
    fn fill_glyph(
        &mut self,
        glyph_id: skrifa::GlyphId,
        brush_transform: Option<skrifa::color::Transform>,
        brush: skrifa::color::Brush<'_>,
    ) {
        match brush {
            skrifa::color::Brush::Solid {
                palette_index,
                alpha,
            } => {
                let Some(outline) = self.outlines.get(glyph_id) else {
                    eprintln!("Didn't get expected outline");
                    return;
                };
                self.encoding.encode_fill_style(peniko::Fill::NonZero);
                let mut path = self.encoding.encode_path(true);
                let draw_settings = DrawSettings::unhinted(Size::unscaled(), [].as_slice());
                outline.draw(draw_settings, &mut path).unwrap();
                if path.finish(true) != 0 {
                    if let Some(transform) = brush_transform {
                        let transform = Transform {
                            matrix: [transform.xx, transform.xy, transform.yx, transform.yy],
                            translation: [transform.dx, transform.dy],
                        };
                        if self.encoding.encode_transform(transform) {
                            self.encoding.swap_last_path_tags()
                        }
                    }
                    // let color_index = self.cpal.color_record_indices()[usize::from(palette_index)];
                    let actual_colors = self.cpal.color_records_array().unwrap().unwrap();
                    let color = actual_colors[usize::from(palette_index)];
                    let brush = Brush::Solid(Color::rgba8(
                        color.red,
                        color.green,
                        color.blue,
                        color.alpha,
                    ));
                    self.encoding.encode_brush(&brush, alpha)
                }
            }
            e => eprintln!("Unsupported brush {e:?}"),
        }
    }
}
