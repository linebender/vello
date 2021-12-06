use std::ops::RangeBounds;

use swash::scale::{ScaleContext, Scaler};
use swash::zeno::{Vector, Verb};
use swash::{FontRef, GlyphId};

use piet::kurbo::{Point, Rect, Size};
use piet::{
    Error, FontFamily, HitTestPoint, HitTestPosition, LineMetric, Text, TextAttribute, TextLayout,
    TextLayoutBuilder, TextStorage,
};

use crate::encoder::GlyphEncoder;
use crate::render_ctx::{self, FillMode};
use crate::PietGpuRenderContext;
use crate::stages::Transform;

// This is very much a hack to get things working.
// On Windows, can set this to "c:\\Windows\\Fonts\\seguiemj.ttf" to get color emoji
const FONT_DATA: &[u8] = include_bytes!("../third-party/Roboto-Regular.ttf");

#[derive(Clone)]
pub struct Font {
    // Storing the font_ref is ok for static font data, but the better way to do
    // this is to store the CacheKey.
    font_ref: FontRef<'static>,
}

#[derive(Clone)]
pub struct PietGpuText {
    font: Font,
}

#[derive(Clone)]
pub struct PietGpuTextLayout {
    font: Font,
    size: f64,
    glyphs: Vec<Glyph>,
}

pub struct PietGpuTextLayoutBuilder {
    font: Font,
    text: String,
    size: f64,
}

#[derive(Clone, Debug)]
struct Glyph {
    glyph_id: GlyphId,
    x: f32,
    y: f32,
}


struct TextRenderCtx<'a> {
    scaler: Scaler<'a>,
}

impl PietGpuText {
    pub(crate) fn new(font: Font) -> PietGpuText {
        PietGpuText { font }
    }
}

impl Text for PietGpuText {
    type TextLayout = PietGpuTextLayout;
    type TextLayoutBuilder = PietGpuTextLayoutBuilder;

    fn load_font(&mut self, _data: &[u8]) -> Result<FontFamily, Error> {
        Ok(FontFamily::default())
    }

    fn new_text_layout(&mut self, text: impl TextStorage) -> Self::TextLayoutBuilder {
        PietGpuTextLayoutBuilder::new(&self.font, &text.as_str())
    }

    fn font_family(&mut self, _family_name: &str) -> Option<FontFamily> {
        Some(FontFamily::default())
    }
}

impl TextLayout for PietGpuTextLayout {
    fn size(&self) -> Size {
        Size::ZERO
    }

    fn image_bounds(&self) -> Rect {
        Rect::ZERO
    }

    fn line_text(&self, _line_number: usize) -> Option<&str> {
        None
    }

    fn line_metric(&self, _line_number: usize) -> Option<LineMetric> {
        None
    }

    fn line_count(&self) -> usize {
        0
    }

    fn hit_test_point(&self, _point: Point) -> HitTestPoint {
        HitTestPoint::default()
    }

    fn hit_test_text_position(&self, _text_position: usize) -> HitTestPosition {
        HitTestPosition::default()
    }

    fn text(&self) -> &str {
        ""
    }
}

impl Font {
    pub fn new() -> Font {
        let font_ref = FontRef::from_index(FONT_DATA, 0).expect("error parsing font");
        Font { font_ref }
    }

    fn make_path<'a>(&self, glyph_id: GlyphId, tc: &mut TextRenderCtx<'a>) -> GlyphEncoder {
        let mut encoder = GlyphEncoder::default();
        if tc.scaler.has_color_outlines() {
            if let Some(outline) = tc.scaler.scale_color_outline(glyph_id) {
                // TODO: be more sophisticated choosing a palette
                let palette = self.font_ref.color_palettes().next().unwrap();
                let mut i = 0;
                while let Some(layer) = outline.get(i) {
                    if let Some(color_ix) = layer.color_index() {
                        let color = palette.get(color_ix);
                        append_outline(&mut encoder, layer.verbs(), layer.points());
                        encoder.fill_color(*bytemuck::from_bytes(&color));
                    }
                    i += 1;
                }
                return encoder;
            }
        }
        if let Some(outline) = tc.scaler.scale_outline(glyph_id) {
            append_outline(&mut encoder, outline.verbs(), outline.points());
        }
        encoder
    }
}

impl PietGpuTextLayout {
    pub(crate) fn make_layout(font: &Font, text: &str, size: f64) -> PietGpuTextLayout {
        let mut glyphs = Vec::new();
        let mut x = 0.0;
        let y = 0.0;
        for c in text.chars() {
            let glyph_id = font.font_ref.charmap().map(c);
            let glyph = Glyph { glyph_id, x, y };
            glyphs.push(glyph);
            let adv = font.font_ref.glyph_metrics(&[]).advance_width(glyph_id);
            x += adv;
        }
        PietGpuTextLayout {
            glyphs,
            font: font.clone(),
            size,
        }
    }

    pub(crate) fn draw_text(&self, ctx: &mut PietGpuRenderContext, pos: Point) {
        let mut scale_ctx = ScaleContext::new();
        let scaler = scale_ctx.builder(self.font.font_ref).size(2048.).build();
        let mut tc = TextRenderCtx { scaler };
        // Should we use ppem from font, or let swash scale?
        const DEFAULT_UPEM: u16 = 2048;
        let scale = self.size as f32 / DEFAULT_UPEM as f32;
        let mut inv_transform = None;
        // TODO: handle y offsets also
        let mut last_x = 0.0;
        ctx.set_fill_mode(FillMode::Nonzero);
        for glyph in &self.glyphs {
            let transform = match &mut inv_transform {
                None => {
                    let inv_scale = scale.recip();
                    let translate = render_ctx::to_f32_2(pos);
                    inv_transform = Some(Transform {
                        mat: [inv_scale, 0.0, 0.0, -inv_scale],
                        translate: [
                            -translate[0] * inv_scale - glyph.x,
                            translate[1] * inv_scale,
                        ],
                    });
                    let tpos = render_ctx::to_f32_2(pos);
                    let translate = [tpos[0] + scale * glyph.x, tpos[1]];
                    Transform {
                        mat: [scale, 0.0, 0.0, -scale],
                        translate,
                    }
                }
                Some(inv) => {
                    let delta_x = glyph.x - last_x;
                    inv.translate[0] -= delta_x;
                    Transform {
                        mat: [1.0, 0.0, 0.0, 1.0],
                        translate: [delta_x, 0.0],
                    }
                }
            };
            last_x = glyph.x;
            //println!("{:?}, {:?}", transform.mat, transform.translate);
            ctx.encode_transform(transform);
            let glyph = self.font.make_path(glyph.glyph_id, &mut tc);
            ctx.encode_glyph(&glyph);
            if !glyph.is_color() {
                ctx.fill_glyph(0xff_ff_ff_ff);
            }
        }
        if let Some(transform) = inv_transform {
            ctx.encode_transform(transform);
        }
    }
}

impl PietGpuTextLayoutBuilder {
    pub(crate) fn new(font: &Font, text: &str) -> PietGpuTextLayoutBuilder {
        PietGpuTextLayoutBuilder {
            font: font.clone(),
            text: text.to_owned(),
            size: 12.0,
        }
    }
}

impl TextLayoutBuilder for PietGpuTextLayoutBuilder {
    type Out = PietGpuTextLayout;

    fn max_width(self, _width: f64) -> Self {
        self
    }

    fn alignment(self, _alignment: piet::TextAlignment) -> Self {
        self
    }

    fn default_attribute(mut self, attribute: impl Into<TextAttribute>) -> Self {
        let attribute = attribute.into();
        match attribute {
            TextAttribute::FontSize(size) => self.size = size,
            _ => (),
        }
        self
    }

    fn range_attribute(
        self,
        _range: impl RangeBounds<usize>,
        _attribute: impl Into<TextAttribute>,
    ) -> Self {
        self
    }

    fn build(self) -> Result<Self::Out, Error> {
        Ok(PietGpuTextLayout::make_layout(
            &self.font, &self.text, self.size,
        ))
    }
}

fn append_outline(encoder: &mut GlyphEncoder, verbs: &[Verb], points: &[Vector]) {
    let mut path_encoder = encoder.path_encoder();
    let mut i = 0;
    for verb in verbs {
        match verb {
            Verb::MoveTo => {
                let p = points[i];
                path_encoder.move_to(p.x, p.y);
                i += 1;
            }
            Verb::LineTo => {
                let p = points[i];
                path_encoder.line_to(p.x, p.y);
                i += 1;
            }
            Verb::QuadTo => {
                let p1 = points[i];
                let p2 = points[i + 1];
                path_encoder.quad_to(p1.x, p1.y, p2.x, p2.y);
                i += 2;
            }
            Verb::CurveTo => {
                let p1 = points[i];
                let p2 = points[i + 1];
                let p3 = points[i + 2];
                path_encoder.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                i += 3;
            }
            Verb::Close => path_encoder.close_path(),
        }
    }
    path_encoder.path();
    let n_pathseg = path_encoder.n_pathseg();
    encoder.finish_path(n_pathseg);
}
