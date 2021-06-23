use std::ops::RangeBounds;

use ttf_parser::{Face, GlyphId, OutlineBuilder};

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
    face: Face<'static>,
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
    start_pt: [f32; 2],
    cur_pt: [f32; 2],
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
        let face = Face::from_slice(FONT_DATA, 0).expect("error parsing font");
        Font { face }
    }

    fn make_path(&self, glyph_id: GlyphId) -> PathEncoder {
        let mut encoder = PathEncoder::default();
        self.face.outline_glyph(glyph_id, &mut encoder);
        encoder
    }
}

impl PietGpuTextLayout {
    pub(crate) fn make_layout(font: &Font, text: &str, size: f64) -> PietGpuTextLayout {
        let mut glyphs = Vec::new();
        let mut x = 0.0;
        let y = 0.0;
        for c in text.chars() {
            if let Some(glyph_id) = font.face.glyph_index(c) {
                let glyph = Glyph { glyph_id, x, y };
                glyphs.push(glyph);
                if let Some(adv) = font.face.glyph_hor_advance(glyph_id) {
                    x += adv as f32;
                }
            }
        }
        PietGpuTextLayout {
            glyphs,
            font: font.clone(),
            size: size,
        }
    }

    pub(crate) fn draw_text(&self, ctx: &mut PietGpuRenderContext, pos: Point) {
        const DEFAULT_UPEM: u16 = 1024;
        let scale = self.size as f32 / self.font.face.units_per_em().unwrap_or(DEFAULT_UPEM) as f32;
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

impl OutlineBuilder for PathEncoder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.start_pt = [x, y];
        self.cur_pt = [x, y];
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = [x, y];
        let seg = LineSeg {
            p0: self.cur_pt,
            p1: p1,
        };
        self.cur_pt = p1;
        self.elements.push(Element::Line(seg));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p1 = [x1, y1];
        let p2 = [x, y];
        let seg = QuadSeg {
            p0: self.cur_pt,
            p1: p1,
            p2: p2,
        };
        self.cur_pt = p2;
        self.elements.push(Element::Quad(seg));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p1 = [x1, y1];
        let p2 = [x2, y2];
        let p3 = [x, y];
        let seg = CubicSeg {
            p0: self.cur_pt,
            p1: p1,
            p2: p2,
            p3: p3,
        };
        self.cur_pt = p3;
        self.elements.push(Element::Cubic(seg));
    }

    fn close(&mut self) {
        if self.cur_pt != self.start_pt {
            let seg = LineSeg {
                p0: self.cur_pt,
                p1: self.start_pt,
            };
            self.elements.push(Element::Line(seg));
        }
    }
}

impl PathEncoder {
    pub(crate) fn elements(&self) -> &[Element] {
        &self.elements
    }
}
