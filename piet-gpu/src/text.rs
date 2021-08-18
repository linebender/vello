use std::ops::RangeBounds;

use swash::scale::ScaleContext;
use swash::zeno::{Vector, Verb};
use swash::{FontRef, GlyphId};

use piet::kurbo::{Point, Rect, Size};
use piet::{
    Error, FontFamily, HitTestPoint, HitTestPosition, LineMetric, Text, TextAttribute, TextLayout,
    TextLayoutBuilder, TextStorage,
};

use piet_gpu_types::scene::{CubicSeg, Element, LineSeg, QuadSeg, Transform};

use crate::render_ctx::{self, FillMode};
use crate::PietGpuRenderContext;

// This is very much a hack to get things working.
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

#[derive(Default)]
pub struct PathEncoder {
    elements: Vec<Element>,
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

    fn make_path(&self, glyph_id: GlyphId) -> PathEncoder {
        /*
        let mut encoder = PathEncoder::default();
        self.face.outline_glyph(glyph_id, &mut encoder);
        encoder
        */
        // Should the scale context be in the font? In the RenderCtx?
        let mut scale_context = ScaleContext::new();
        let mut scaler = scale_context.builder(self.font_ref).size(2048.).build();
        let mut encoder = PathEncoder::default();
        if let Some(outline) = scaler.scale_outline(glyph_id) {
            let verbs = outline.verbs();
            let points = outline.points();
            let elements = &mut encoder.elements;
            let mut i = 0;
            let mut start_pt = [0.0f32; 2];
            let mut last_pt = [0.0f32; 2];
            for verb in verbs {
                match verb {
                    Verb::MoveTo => {
                        start_pt = convert_swash_point(points[i]);
                        last_pt = start_pt;
                        i += 1;
                    }
                    Verb::LineTo => {
                        let p1 = convert_swash_point(points[i]);
                        elements.push(Element::Line(LineSeg {
                            p0: last_pt,
                            p1,
                        }));
                        last_pt = p1;
                        i += 1;
                    }
                    Verb::QuadTo => {
                        let p1 = convert_swash_point(points[i]);
                        let p2 = convert_swash_point(points[i + 1]);
                        elements.push(Element::Quad(QuadSeg {
                            p0: last_pt,
                            p1,
                            p2,
                        }));
                        last_pt = p2;
                        i += 2;
                    }
                    Verb::CurveTo => {
                        let p1 = convert_swash_point(points[i]);
                        let p2 = convert_swash_point(points[i + 1]);
                        let p3 = convert_swash_point(points[i + 2]);
                        elements.push(Element::Cubic(CubicSeg {
                            p0: last_pt,
                            p1,
                            p2,
                            p3,
                        }));
                        last_pt = p3;
                        i += 3;
                    }
                    Verb::Close => {
                        if start_pt != last_pt {
                            elements.push(Element::Line(LineSeg {
                                p0: last_pt,
                                p1: start_pt,
                            }));
                        }
                    }
                }
            }
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
            let path = self.font.make_path(glyph.glyph_id);
            ctx.append_path_encoder(&path);
            ctx.fill_glyph(0xff_ff_ff_ff);
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

impl PathEncoder {
    pub(crate) fn elements(&self) -> &[Element] {
        &self.elements
    }
}

fn convert_swash_point(v: Vector) -> [f32; 2] {
    [v.x, v.y]
}
