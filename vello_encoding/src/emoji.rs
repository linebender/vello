use skrifa::{
    color::ColorPainter,
    instance::{Location, Size},
    outline::DrawSettings,
    OutlineGlyphCollection,
};

use crate::Encoding;

pub(crate) struct EncodeColorGlyph<'a> {
    pub(crate) encoding: &'a mut Encoding,
    pub(crate) outlines: &'a OutlineGlyphCollection<'a>,
}

impl<'a> ColorPainter for EncodeColorGlyph<'a> {
    fn push_transform(&mut self, transform: skrifa::color::Transform) {}

    fn pop_transform(&mut self) {
        ()
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
            // Don't create a clip?
        } else {
            // Create a clip?
        }
    }

    fn push_clip_box(&mut self, clip_box: skrifa::raw::types::BoundingBox<f32>) {
        ()
    }

    fn pop_clip(&mut self) {
        ()
    }

    fn fill(&mut self, brush: skrifa::color::Brush<'_>) {
        todo!()
    }

    fn push_layer(&mut self, composite_mode: skrifa::color::CompositeMode) {
        todo!()
    }

    fn pop_layer(&mut self) {
        todo!()
    }
}
