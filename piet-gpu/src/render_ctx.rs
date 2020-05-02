use std::borrow::Cow;

use piet_gpu_types::encoder::{Encode, Encoder, Ref};
use piet_gpu_types::scene;
use piet_gpu_types::scene::{Bbox, PietCircle, PietFill, PietItem, PietStrokePolyLine, SimpleGroup};

use piet::kurbo::{Affine, PathEl, Point, Rect, Shape};

use piet::{
    Color, Error, FixedGradient, Font, FontBuilder, HitTestPoint, HitTestTextPosition, ImageFormat,
    InterpolationMode, IntoBrush, LineMetric, RenderContext, StrokeStyle, Text, TextLayout,
    TextLayoutBuilder,
};

pub struct PietGpuImage;

pub struct PietGpuFont;

pub struct PietGpuFontBuilder;

#[derive(Clone)]
pub struct PietGpuTextLayout;

pub struct PietGpuTextLayoutBuilder;

pub struct PietGpuText;

pub struct PietGpuRenderContext {
    encoder: Encoder,
    bboxes: Vec<Bbox>,
    items: Vec<PietItem>,
    // Will probably need direct accesss to hal Device to create images etc.
    inner_text: PietGpuText,
}

#[derive(Clone)]
pub enum PietGpuBrush {
    Solid(u32),
    Gradient,
}

const TOLERANCE: f64 = 0.25;

impl PietGpuRenderContext {
    pub fn new() -> PietGpuRenderContext {
        let mut encoder = Encoder::new();
        let _reserve_root = encoder.alloc_chunk(PietItem::fixed_size() as u32);
        let bboxes = Vec::new();
        let items = Vec::new();
        let inner_text = PietGpuText;
        PietGpuRenderContext {
            encoder,
            bboxes,
            items,
            inner_text,
        }
    }

    pub fn get_scene_buf(&mut self) -> &[u8] {
        let n_items = self.bboxes.len() as u32;
        let bboxes = self.bboxes.encode(&mut self.encoder).transmute();
        let items = self.items.encode(&mut self.encoder).transmute();
        let offset = scene::Point { xy: [0.0, 0.0] };
        let simple_group = SimpleGroup {
            n_items,
            bboxes,
            items,
            offset,
        };
        let root_item = PietItem::Group(simple_group);
        root_item.encode_to(&mut self.encoder.buf_mut()[0..PietItem::fixed_size()]);
        self.encoder.buf()
    }

    fn push_item(&mut self, item: PietItem, bbox: Rect) {
        let scene_bbox = Bbox {
            bbox: [
                bbox.x0.floor() as i16,
                bbox.y0.floor() as i16,
                bbox.x1.ceil() as i16,
                bbox.y1.ceil() as i16,
            ],
        };
        self.items.push(item);
        self.bboxes.push(scene_bbox);
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
        let bbox = shape.bounding_box();
        let brush = brush.make_brush(self, || bbox).into_owned();
        let path = shape.to_bez_path(TOLERANCE);
        let (n_points, points) = flatten_shape(&mut self.encoder, path);
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                let poly_line = PietStrokePolyLine {
                    rgba_color,
                    width: width as f32,
                    n_points,
                    points,
                };
                let bbox = bbox.inset(-0.5 * width);
                self.push_item(PietItem::Poly(poly_line), bbox);
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
        let bbox = shape.bounding_box();
        let brush = brush.make_brush(self, || shape.bounding_box()).into_owned();

        if let Some(circle) = shape.as_circle() {
            match brush {
                PietGpuBrush::Solid(rgba_color) => {
                    let piet_circle = PietCircle {
                        rgba_color,
                        center: to_scene_point(circle.center),
                        radius: circle.radius as f32,
                    };
                    let bbox = circle.bounding_box();
                    self.push_item(PietItem::Circle(piet_circle), bbox);
                }
                _ => {}
            }
            return;
        }
        let path = shape.to_bez_path(TOLERANCE);
        let (n_points, points) = flatten_shape(&mut self.encoder, path);
        match brush {
            PietGpuBrush::Solid(rgba_color) => {
                let fill = PietFill {
                    flags: 0,
                    rgba_color,
                    n_points,
                    points,
                };
                self.push_item(PietItem::Fill(fill), bbox);
            }
            _ => (),
        }
    }

    fn fill_even_odd(&mut self, _shape: impl Shape, _brush: &impl IntoBrush<Self>) {}

    fn clip(&mut self, _shape: impl Shape) {}

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(
        &mut self,
        _layout: &Self::TextLayout,
        pos: impl Into<Point>,
        brush: &impl IntoBrush<Self>,
    ) {
        let _pos = pos.into();

        let brush: PietGpuBrush = brush.make_brush(self, || Rect::ZERO).into_owned();

        match brush {
            PietGpuBrush::Solid(_rgba) => {
                // TODO: draw text
            }
            _ => {}
        }
    }

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

fn flatten_shape(
    encoder: &mut Encoder,
    path: impl Iterator<Item = PathEl>,
) -> (u32, Ref<scene::Point>) {
    let mut points = Vec::new();
    let mut start_pt = None;
    let mut last_pt = None;
    kurbo::flatten(path, TOLERANCE, |el| {
        match el {
            PathEl::MoveTo(p) => {
                let scene_pt = to_scene_point(p);
                start_pt = Some(clone_scene_pt(&scene_pt));
                if !points.is_empty() {
                    points.push(scene::Point {
                        xy: [std::f32::NAN, std::f32::NAN],
                    });
                }
                last_pt = Some(clone_scene_pt(&scene_pt));
                points.push(scene_pt);
            }
            PathEl::LineTo(p) => {
                let scene_pt = to_scene_point(p);
                last_pt = Some(clone_scene_pt(&scene_pt));
                points.push(scene_pt);
            }
            PathEl::ClosePath => {
                if let (Some(start), Some(last)) = (start_pt.take(), last_pt.take()) {
                    if start.xy != last.xy {
                        points.push(start);
                    }
                }
            }
            _ => (),
        }
        //println!("{:?}", el);
    });
    let n_points = points.len() as u32;
    let points_ref = points.encode(encoder).transmute();
    (n_points, points_ref)
}

impl Text for PietGpuText {
    type Font = PietGpuFont;
    type FontBuilder = PietGpuFontBuilder;
    type TextLayout = PietGpuTextLayout;
    type TextLayoutBuilder = PietGpuTextLayoutBuilder;

    fn new_font_by_name(&mut self, _name: &str, _size: f64) -> Self::FontBuilder {
        unimplemented!();
    }

    fn new_text_layout(
        &mut self,
        _font: &Self::Font,
        _text: &str,
        _width: f64,
    ) -> Self::TextLayoutBuilder {
        unimplemented!();
    }
}

impl Font for PietGpuFont {}

impl FontBuilder for PietGpuFontBuilder {
    type Out = PietGpuFont;

    fn build(self) -> Result<Self::Out, Error> {
        unimplemented!();
    }
}

impl TextLayoutBuilder for PietGpuTextLayoutBuilder {
    type Out = PietGpuTextLayout;

    fn build(self) -> Result<Self::Out, Error> {
        unimplemented!()
    }
}

impl TextLayout for PietGpuTextLayout {
    fn width(&self) -> f64 {
        0.0
    }

    fn update_width(&mut self, _new_width: f64) -> Result<(), Error> {
        unimplemented!()
    }

    fn line_text(&self, _line_number: usize) -> Option<&str> {
        unimplemented!()
    }

    fn line_metric(&self, _line_number: usize) -> Option<LineMetric> {
        unimplemented!()
    }

    fn line_count(&self) -> usize {
        unimplemented!()
    }

    fn hit_test_point(&self, _point: Point) -> HitTestPoint {
        unimplemented!()
    }

    fn hit_test_text_position(&self, _text_position: usize) -> Option<HitTestTextPosition> {
        unimplemented!()
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

fn to_scene_point(point: Point) -> scene::Point {
    scene::Point {
        xy: [point.x as f32, point.y as f32],
    }
}

// TODO: allow #[derive(Clone)] in piet-gpu-derive.
fn clone_scene_pt(p: &scene::Point) -> scene::Point {
    scene::Point { xy: p.xy }
}
