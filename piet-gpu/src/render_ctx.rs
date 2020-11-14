use std::{borrow::Cow, ops::RangeBounds};

use piet_gpu_types::encoder::{Encode, Encoder};

use piet_gpu_types::scene::{CubicSeg, Element, Fill, LineSeg, QuadSeg, SetLineWidth, Stroke};

use piet::{
    kurbo::Size,
    kurbo::{Affine, PathEl, Point, Rect, Shape},
    HitTestPosition, TextAttribute, TextStorage,
};

use piet::{
    Color, Error, FixedGradient, FontFamily, HitTestPoint, ImageFormat, InterpolationMode,
    IntoBrush, LineMetric, RenderContext, StrokeStyle, Text, TextLayout, TextLayoutBuilder,
};

pub struct PietGpuImage;

#[derive(Clone)]
pub struct PietGpuTextLayout;

pub struct PietGpuTextLayoutBuilder;

#[derive(Clone)]
pub struct PietGpuText;

pub struct PietGpuRenderContext {
    encoder: Encoder,
    elements: Vec<Element>,
    // Will probably need direct accesss to hal Device to create images etc.
    inner_text: PietGpuText,
    stroke_width: f32,
    // We're tallying these cpu-side for expedience, but will probably
    // move this to some kind of readback from element processing.
    path_count: usize,
    pathseg_count: usize,
}

#[derive(Clone)]
pub enum PietGpuBrush {
    Solid(u32),
    Gradient,
}

const TOLERANCE: f64 = 0.25;

impl PietGpuRenderContext {
    pub fn new() -> PietGpuRenderContext {
        let encoder = Encoder::new();
        let elements = Vec::new();
        let inner_text = PietGpuText;
        let stroke_width = 0.0;
        PietGpuRenderContext {
            encoder,
            elements,
            inner_text,
            stroke_width,
            path_count: 0,
            pathseg_count: 0,
        }
    }

    pub fn get_scene_buf(&mut self) -> &[u8] {
        self.elements.encode(&mut self.encoder);
        self.encoder.buf()
    }

    pub fn path_count(&self) -> usize {
        self.path_count
    }

    pub fn pathseg_count(&self) -> usize {
        self.pathseg_count
    }
}

impl RenderContext for PietGpuRenderContext {
    type Brush = PietGpuBrush;
    type Image = PietGpuImage;
    type Text = PietGpuText;
    type TextLayout = PietGpuTextLayout;

    fn status(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn solid_brush(&mut self, color: Color) -> Self::Brush {
        PietGpuBrush::Solid(color.as_rgba_u32())
    }

    fn gradient(&mut self, _gradient: impl Into<FixedGradient>) -> Result<Self::Brush, Error> {
        Ok(Self::Brush::Gradient)
    }

    fn clear(&mut self, _color: Color) {}

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        let width = width as f32;
        if self.stroke_width != width {
            self.elements
                .push(Element::SetLineWidth(SetLineWidth { width }));
            self.stroke_width = width;
        }
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, false);
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                let stroke = Stroke { rgba_color };
                self.elements.push(Element::Stroke(stroke));
                self.path_count += 1;
            }
            _ => (),
        }
    }

    fn stroke_styled(
        &mut self,
        _shape: impl Shape,
        _brush: &impl IntoBrush<Self>,
        _width: f64,
        _style: &StrokeStyle,
    ) {
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();
        let path = shape.path_elements(TOLERANCE);
        self.encode_path(path, true);
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                let fill = Fill { rgba_color };
                self.elements.push(Element::Fill(fill));
                self.path_count += 1;
            }
            _ => (),
        }
    }

    fn fill_even_odd(&mut self, _shape: impl Shape, _brush: &impl IntoBrush<Self>) {}

    fn clip(&mut self, _shape: impl Shape) {}

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(&mut self, _layout: &Self::TextLayout, _pos: impl Into<Point>) {}

    fn save(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn restore(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn transform(&mut self, _transform: Affine) {}

    fn make_image(
        &mut self,
        _width: usize,
        _height: usize,
        _buf: &[u8],
        _format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        Ok(PietGpuImage)
    }

    fn draw_image(
        &mut self,
        _image: &Self::Image,
        _rect: impl Into<Rect>,
        _interp: InterpolationMode,
    ) {
    }

    fn draw_image_area(
        &mut self,
        _image: &Self::Image,
        _src_rect: impl Into<Rect>,
        _dst_rect: impl Into<Rect>,
        _interp: InterpolationMode,
    ) {
    }

    fn blurred_rect(&mut self, _rect: Rect, _blur_radius: f64, _brush: &impl IntoBrush<Self>) {}

    fn current_transform(&self) -> Affine {
        Default::default()
    }
}

impl PietGpuRenderContext {
    fn encode_line_seg(&mut self, seg: LineSeg, is_fill: bool) {
        if is_fill {
            self.elements.push(Element::FillLine(seg));
        } else {
            self.elements.push(Element::StrokeLine(seg));
        }
        self.pathseg_count += 1;
    }

    fn encode_quad_seg(&mut self, seg: QuadSeg, is_fill: bool) {
        if is_fill {
            self.elements.push(Element::FillQuad(seg));
        } else {
            self.elements.push(Element::StrokeQuad(seg));
        }
        self.pathseg_count += 1;
    }

    fn encode_cubic_seg(&mut self, seg: CubicSeg, is_fill: bool) {
        if is_fill {
            self.elements.push(Element::FillCubic(seg));
        } else {
            self.elements.push(Element::StrokeCubic(seg));
        }
        self.pathseg_count += 1;
    }

    fn encode_path(&mut self, path: impl Iterator<Item = PathEl>, is_fill: bool) {
        let flatten = false;
        if flatten {
            let mut start_pt = None;
            let mut last_pt = None;
            piet::kurbo::flatten(path, TOLERANCE, |el| {
                match el {
                    PathEl::MoveTo(p) => {
                        let scene_pt = to_f32_2(p);
                        start_pt = Some(scene_pt);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::LineTo(p) => {
                        let scene_pt = to_f32_2(p);
                        let seg = LineSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_pt,
                        };
                        self.encode_line_seg(seg, is_fill);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::ClosePath => {
                        if let (Some(start), Some(last)) = (start_pt.take(), last_pt.take()) {
                            if last != start {
                                let seg = LineSeg {
                                    p0: last,
                                    p1: start,
                                };
                                self.encode_line_seg(seg, is_fill);
                            }
                        }
                    }
                    _ => (),
                }
                //println!("{:?}", el);
            });
        } else {
            let mut start_pt = None;
            let mut last_pt = None;
            for el in path {
                match el {
                    PathEl::MoveTo(p) => {
                        let scene_pt = to_f32_2(p);
                        start_pt = Some(scene_pt);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::LineTo(p) => {
                        let scene_pt = to_f32_2(p);
                        let seg = LineSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_pt,
                        };
                        self.encode_line_seg(seg, is_fill);
                        last_pt = Some(scene_pt);
                    }
                    PathEl::QuadTo(p1, p2) => {
                        let scene_p1 = to_f32_2(p1);
                        let scene_p2 = to_f32_2(p2);
                        let seg = QuadSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_p1,
                            p2: scene_p2,
                        };
                        self.encode_quad_seg(seg, is_fill);
                        last_pt = Some(scene_p2);
                    }
                    PathEl::CurveTo(p1, p2, p3) => {
                        let scene_p1 = to_f32_2(p1);
                        let scene_p2 = to_f32_2(p2);
                        let scene_p3 = to_f32_2(p3);
                        let seg = CubicSeg {
                            p0: last_pt.unwrap(),
                            p1: scene_p1,
                            p2: scene_p2,
                            p3: scene_p3,
                        };
                        self.encode_cubic_seg(seg, is_fill);
                        last_pt = Some(scene_p3);
                    }
                    PathEl::ClosePath => {
                        if let (Some(start), Some(last)) = (start_pt.take(), last_pt.take()) {
                            if last != start {
                                let seg = LineSeg {
                                    p0: last,
                                    p1: start,
                                };
                                self.encode_line_seg(seg, is_fill);
                            }
                        }
                    }
                }
                //println!("{:?}", el);
            }
        }
    }
}

impl Text for PietGpuText {
    type TextLayout = PietGpuTextLayout;
    type TextLayoutBuilder = PietGpuTextLayoutBuilder;

    fn load_font(&mut self, _data: &[u8]) -> Result<FontFamily, Error> {
        Ok(FontFamily::default())
    }

    fn new_text_layout(&mut self, _text: impl TextStorage) -> Self::TextLayoutBuilder {
        PietGpuTextLayoutBuilder
    }

    fn font_family(&mut self, _family_name: &str) -> Option<FontFamily> {
        Some(FontFamily::default())
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

    fn default_attribute(self, _attribute: impl Into<TextAttribute>) -> Self {
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
        Ok(PietGpuTextLayout)
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

impl IntoBrush<PietGpuRenderContext> for PietGpuBrush {
    fn make_brush<'b>(
        &'b self,
        _piet: &mut PietGpuRenderContext,
        _bbox: impl FnOnce() -> Rect,
    ) -> std::borrow::Cow<'b, PietGpuBrush> {
        Cow::Borrowed(self)
    }
}

fn to_f32_2(point: Point) -> [f32; 2] {
    [point.x as f32, point.y as f32]
}
