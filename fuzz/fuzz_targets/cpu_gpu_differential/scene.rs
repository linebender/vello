// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bounded, structured scene model decoded from fuzzer bytes with `arbitrary`.
//!
//! Every value converts to a well-formed drawing primitive, so the fuzzer explores rendering
//! behaviour rather than input validation. Coordinates are kept near the viewport with subpixel
//! fractions to concentrate on anti-aliasing and clipping edges.

use crate::config::IMAGE_QUALITY_ALL;
use crate::images::ImageTable;
use arbitrary::{Arbitrary, Unstructured};
use std::f64::consts::TAU;
use vello_common::color::{AlphaColor, ColorSpaceTag, DynamicColor, Srgb};
use vello_common::kurbo::{Affine, BezPath, Cap, Dashes, Join, Point, Rect, Shape, Stroke};
use vello_common::paint::{Image, PaintType};
use vello_common::peniko::{
    BlendMode, ColorStop, ColorStops, Compose, Extend, Fill, Gradient, GradientKind, ImageQuality,
    ImageSampler, LinearGradientPosition, Mix, RadialGradientPosition, SweepGradientPosition,
};

pub(crate) const WIDTH: u16 = 100;
pub(crate) const HEIGHT: u16 = 100;
pub(crate) const MAX_COMMANDS: usize = 48;
pub(crate) const MAX_PATH_SEGMENTS: usize = 8;
pub(crate) const MAX_LAYER_DEPTH: usize = 8;
pub(crate) const MAX_CLIP_DEPTH: usize = 8;
pub(crate) const MIN_GRADIENT_STOPS: usize = 2;
pub(crate) const MAX_GRADIENT_STOPS: usize = 4;
pub(crate) const MAX_DASHES: usize = 4;

#[derive(Debug)]
pub(crate) struct FuzzScene {
    pub(crate) background: FuzzColor,
    pub(crate) commands: Vec<Command>,
}

impl<'a> Arbitrary<'a> for FuzzScene {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let background = FuzzColor::arbitrary(input).unwrap_or(FuzzColor {
            red: 0,
            green: 0,
            blue: 0,
            alpha: 0,
        });
        let command_count = input.int_in_range(0..=MAX_COMMANDS).unwrap_or(0);
        let mut commands = Vec::with_capacity(command_count);
        for _ in 0..command_count {
            let Ok(command) = Command::arbitrary(input) else {
                break;
            };
            commands.push(command);
        }
        Ok(Self {
            background,
            commands,
        })
    }

    fn size_hint(_depth: usize) -> (usize, Option<usize>) {
        (0, None)
    }
}

/// Layers and non-isolated clip paths form independent stacks on both backends, so the two
/// kinds of push/pop may interleave freely; only underflow and depth are guarded during replay.
#[derive(Arbitrary, Debug)]
pub(crate) enum Command {
    Draw(DrawCommand),
    PushLayer(LayerKind),
    PopLayer,
    PushClip(ClipCommand),
    PopClip,
}

/// `push_clip_rect` / `push_clip_path`, which use the transform and fill rule current at the call.
#[derive(Arbitrary, Debug)]
pub(crate) struct ClipCommand {
    pub(crate) shape: FuzzShape,
    pub(crate) fill_rule: FuzzFillRule,
    pub(crate) transform: FuzzTransform,
}

/// The paint transform maps gradient and image space into user space and is only applied to
/// the renderer when the paint is not solid, so it never appears for a solid draw.
#[derive(Arbitrary, Debug)]
pub(crate) struct DrawCommand {
    pub(crate) shape: FuzzShape,
    pub(crate) paint: FuzzPaint,
    pub(crate) paint_transform: FuzzTransform,
    pub(crate) transform: FuzzTransform,
    pub(crate) style: DrawStyle,
}

#[derive(Arbitrary, Debug)]
pub(crate) enum DrawStyle {
    Fill(FuzzFillRule),
    Stroke(FuzzStroke),
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzStroke {
    pub(crate) width: u8,
    pub(crate) join: FuzzJoin,
    pub(crate) miter_limit: u8,
    pub(crate) start_cap: FuzzCap,
    pub(crate) end_cap: FuzzCap,
    pub(crate) dashes: FuzzDashes,
}

impl FuzzStroke {
    /// Stroke width in user space, 0.25 to about 8.
    pub(crate) fn width(&self) -> f64 {
        0.25 + f64::from(self.width) / 32.0
    }

    /// Miter limit from 1 (always bevel) to 17; kurbo's default is 4.
    pub(crate) fn miter_limit(&self) -> f64 {
        1.0 + f64::from(self.miter_limit) / 16.0
    }

    pub(crate) fn to_stroke(&self) -> Stroke {
        Stroke {
            width: self.width(),
            join: self.join.to_join(),
            miter_limit: self.miter_limit(),
            start_cap: self.start_cap.to_cap(),
            end_cap: self.end_cap.to_cap(),
            dash_pattern: Dashes::from_slice(&self.dashes.lengths()),
            dash_offset: self.dashes.offset(),
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzJoin {
    Bevel,
    Miter,
    Round,
}

impl FuzzJoin {
    pub(crate) fn to_join(&self) -> Join {
        match self {
            Self::Bevel => Join::Bevel,
            Self::Miter => Join::Miter,
            Self::Round => Join::Round,
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzCap {
    Butt,
    Square,
    Round,
}

impl FuzzCap {
    pub(crate) fn to_cap(&self) -> Cap {
        match self {
            Self::Butt => Cap::Butt,
            Self::Square => Cap::Square,
            Self::Round => Cap::Round,
        }
    }
}

/// Dash pattern of up to [`MAX_DASHES`] lengths; an empty pattern means a solid stroke.
#[derive(Debug)]
pub(crate) struct FuzzDashes {
    pub(crate) offset: u8,
    pub(crate) lengths: Vec<u8>,
}

impl<'a> Arbitrary<'a> for FuzzDashes {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let offset = u8::arbitrary(input)?;
        let count = input.int_in_range(0..=MAX_DASHES)?;
        let mut lengths = Vec::with_capacity(count);
        for _ in 0..count {
            lengths.push(u8::arbitrary(input)?);
        }
        Ok(Self { offset, lengths })
    }

    fn size_hint(_depth: usize) -> (usize, Option<usize>) {
        (2, None)
    }
}

impl FuzzDashes {
    pub(crate) fn offset(&self) -> f64 {
        f64::from(self.offset) / 4.0
    }

    /// Dash lengths in user space, at least one pixel so long paths stay cheap to stroke.
    pub(crate) fn lengths(&self) -> Vec<f64> {
        self.lengths
            .iter()
            .map(|length| 1.0 + f64::from(*length) / 8.0)
            .collect()
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzPaint {
    Solid(FuzzColor),
    Gradient(FuzzGradient),
    Image(FuzzImagePaint),
}

impl FuzzPaint {
    pub(crate) fn to_paint(&self, images: &ImageTable) -> PaintType {
        match self {
            Self::Solid(color) => color.to_color().into(),
            Self::Gradient(gradient) => gradient.to_gradient().into(),
            Self::Image(image) => image.to_image(images).into(),
        }
    }

    pub(crate) fn uses_paint_transform(&self) -> bool {
        !matches!(self, Self::Solid(_))
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzImagePaint {
    pub(crate) image: FuzzImage,
    pub(crate) x_extend: FuzzExtend,
    pub(crate) y_extend: FuzzExtend,
    pub(crate) quality: FuzzImageQuality,
    pub(crate) alpha: u8,
}

impl FuzzImagePaint {
    pub(crate) fn alpha(&self) -> f32 {
        f32::from(self.alpha) / 255.0
    }

    /// Sampling quality after the `--image-quality` restriction; the generated test uses the
    /// same value so it reproduces what was rendered.
    pub(crate) fn quality(&self) -> ImageQuality {
        if *IMAGE_QUALITY_ALL {
            self.quality.to_quality()
        } else {
            ImageQuality::Low
        }
    }

    pub(crate) fn to_image(&self, images: &ImageTable) -> Image {
        Image {
            image: images.source(self.image).clone(),
            sampler: ImageSampler {
                x_extend: self.x_extend.to_extend(),
                y_extend: self.y_extend.to_extend(),
                quality: self.quality(),
                alpha: self.alpha(),
            },
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzImageQuality {
    Low,
    Medium,
    High,
}

impl FuzzImageQuality {
    pub(crate) fn to_quality(&self) -> ImageQuality {
        match self {
            Self::Low => ImageQuality::Low,
            Self::Medium => ImageQuality::Medium,
            Self::High => ImageQuality::High,
        }
    }
}

/// Assets from the snapshot suite, so a generated test can load the same image with
/// `load_image!`. Small images stress extend modes; the alpha and luma ones the format handling.
#[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FuzzImage {
    Rgb2x2,
    Rgb2x3,
    Rgb10x10,
    Rgba10x10,
    Luma10x10,
    LumaA10x10,
    ColorGrid16x16,
}

impl FuzzImage {
    pub(crate) const ALL: [Self; 7] = [
        Self::Rgb2x2,
        Self::Rgb2x3,
        Self::Rgb10x10,
        Self::Rgba10x10,
        Self::Luma10x10,
        Self::LumaA10x10,
        Self::ColorGrid16x16,
    ];

    /// File stem in `vello_tests/tests/assets`.
    pub(crate) fn asset_name(self) -> &'static str {
        match self {
            Self::Rgb2x2 => "rgb_image_2x2",
            Self::Rgb2x3 => "rgb_image_2x3",
            Self::Rgb10x10 => "rgb_image_10x10",
            Self::Rgba10x10 => "rgba_image_10x10",
            Self::Luma10x10 => "luma_image_10x10",
            Self::LumaA10x10 => "lumaa_image_10x10",
            Self::ColorGrid16x16 => "color_grid_16x16",
        }
    }
}

/// Gradient in user space, so the draw transform applies to it as well.
#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzGradient {
    pub(crate) kind: FuzzGradientKind,
    pub(crate) extend: FuzzExtend,
    pub(crate) interpolation: FuzzInterpolation,
    pub(crate) stops: FuzzColorStops,
}

impl FuzzGradient {
    pub(crate) fn to_gradient(&self) -> Gradient {
        Gradient {
            kind: self.kind.to_kind(),
            extend: self.extend.to_extend(),
            interpolation_cs: self.interpolation.to_color_space(),
            stops: self.stops.to_stops(),
            ..Default::default()
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzGradientKind {
    Linear {
        start: FuzzPoint,
        end: FuzzPoint,
    },
    Radial {
        start_center: FuzzPoint,
        start_radius: u8,
        end_center: FuzzPoint,
        end_radius: u8,
    },
    Sweep {
        center: FuzzPoint,
        start_angle: u8,
        sweep: u8,
    },
}

impl FuzzGradientKind {
    /// Radius in user space, 0 to about 64; zero and equal radii exercise the degenerate paths.
    pub(crate) fn radius(radius: u8) -> f32 {
        f32::from(radius) / 4.0
    }

    /// Start angle in radians, covering the full circle.
    pub(crate) fn start_angle(start_angle: u8) -> f32 {
        f32::from(start_angle) * std::f32::consts::TAU / 256.0
    }

    /// End angle in radians, strictly after the start and at most a full turn later.
    pub(crate) fn end_angle(start_angle: u8, sweep: u8) -> f32 {
        Self::start_angle(start_angle) + (f32::from(sweep) + 1.0) * std::f32::consts::TAU / 256.0
    }

    pub(crate) fn to_kind(&self) -> GradientKind {
        match self {
            Self::Linear { start, end } => GradientKind::Linear(LinearGradientPosition {
                start: start.to_point(),
                end: end.to_point(),
            }),
            Self::Radial {
                start_center,
                start_radius,
                end_center,
                end_radius,
            } => GradientKind::Radial(RadialGradientPosition {
                start_center: start_center.to_point(),
                start_radius: Self::radius(*start_radius),
                end_center: end_center.to_point(),
                end_radius: Self::radius(*end_radius),
            }),
            Self::Sweep {
                center,
                start_angle,
                sweep,
            } => GradientKind::Sweep(SweepGradientPosition {
                center: center.to_point(),
                start_angle: Self::start_angle(*start_angle),
                end_angle: Self::end_angle(*start_angle, *sweep),
            }),
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzExtend {
    Pad,
    Repeat,
    Reflect,
}

impl FuzzExtend {
    pub(crate) fn to_extend(&self) -> Extend {
        match self {
            Self::Pad => Extend::Pad,
            Self::Repeat => Extend::Repeat,
            Self::Reflect => Extend::Reflect,
        }
    }
}

/// Interpolation colour spaces the snapshot suite covers.
#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzInterpolation {
    Srgb,
    LinearSrgb,
    Oklab,
}

impl FuzzInterpolation {
    pub(crate) fn to_color_space(&self) -> ColorSpaceTag {
        match self {
            Self::Srgb => ColorSpaceTag::Srgb,
            Self::LinearSrgb => ColorSpaceTag::LinearSrgb,
            Self::Oklab => ColorSpaceTag::Oklab,
        }
    }
}

/// Between [`MIN_GRADIENT_STOPS`] and [`MAX_GRADIENT_STOPS`] stops, sorted by offset when
/// converted; duplicate offsets are allowed to produce hard colour edges.
#[derive(Debug)]
pub(crate) struct FuzzColorStops(pub(crate) Vec<FuzzColorStop>);

impl<'a> Arbitrary<'a> for FuzzColorStops {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let count = input.int_in_range(MIN_GRADIENT_STOPS..=MAX_GRADIENT_STOPS)?;
        let mut stops = Vec::with_capacity(count);
        for _ in 0..count {
            stops.push(FuzzColorStop::arbitrary(input)?);
        }
        Ok(Self(stops))
    }

    fn size_hint(_depth: usize) -> (usize, Option<usize>) {
        (MIN_GRADIENT_STOPS * 5, None)
    }
}

impl FuzzColorStops {
    /// Stops in offset order, as the renderers expect.
    pub(crate) fn sorted(&self) -> Vec<&FuzzColorStop> {
        let mut stops: Vec<_> = self.0.iter().collect();
        stops.sort_by_key(|stop| stop.offset);
        stops
    }

    pub(crate) fn to_stops(&self) -> ColorStops {
        let stops: Vec<ColorStop> = self
            .sorted()
            .into_iter()
            .map(|stop| ColorStop {
                offset: stop.offset(),
                color: DynamicColor::from_alpha_color(stop.color.to_color()),
            })
            .collect();
        ColorStops::from(stops.as_slice())
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzColorStop {
    pub(crate) offset: u8,
    pub(crate) color: FuzzColor,
}

impl FuzzColorStop {
    pub(crate) fn offset(&self) -> f32 {
        f32::from(self.offset) / 255.0
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum LayerKind {
    Clip(FuzzShape),
    Opacity(u8),
    Blend(FuzzMix),
}

impl LayerKind {
    pub(crate) fn opacity(opacity: u8) -> f32 {
        f32::from(opacity) / 255.0
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzShape {
    Rect(FuzzRect),
    Path(FuzzPath),
}

impl FuzzShape {
    pub(crate) fn to_path(&self) -> BezPath {
        match self {
            Self::Rect(rect) => rect.to_rect().to_path(0.1),
            Self::Path(path) => path.to_path(),
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzRect {
    pub(crate) x0: FuzzCoord,
    pub(crate) y0: FuzzCoord,
    pub(crate) x1: FuzzCoord,
    pub(crate) y1: FuzzCoord,
}

impl FuzzRect {
    pub(crate) fn to_rect(&self) -> Rect {
        Rect::new(
            self.x0.to_f64(),
            self.y0.to_f64(),
            self.x1.to_f64(),
            self.y1.to_f64(),
        )
    }
}

#[derive(Debug)]
pub(crate) struct FuzzPath {
    pub(crate) start: FuzzPoint,
    pub(crate) segments: Vec<FuzzPathSegment>,
    pub(crate) close: bool,
}

impl<'a> Arbitrary<'a> for FuzzPath {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let start = FuzzPoint::arbitrary(input)?;
        let segment_count = input.int_in_range(0..=MAX_PATH_SEGMENTS)?;
        let mut segments = Vec::with_capacity(segment_count);
        for _ in 0..segment_count {
            segments.push(FuzzPathSegment::arbitrary(input)?);
        }
        Ok(Self {
            start,
            segments,
            close: bool::arbitrary(input)?,
        })
    }

    fn size_hint(_depth: usize) -> (usize, Option<usize>) {
        (4, None)
    }
}

impl FuzzPath {
    pub(crate) fn to_path(&self) -> BezPath {
        let mut path = BezPath::new();
        path.move_to(self.start.to_tuple());
        for segment in &self.segments {
            match segment {
                FuzzPathSegment::Line(point) => path.line_to(point.to_tuple()),
                FuzzPathSegment::Quad(control, end) => {
                    path.quad_to(control.to_tuple(), end.to_tuple());
                }
                FuzzPathSegment::Cubic(control_1, control_2, end) => {
                    path.curve_to(control_1.to_tuple(), control_2.to_tuple(), end.to_tuple());
                }
            }
        }
        if self.close {
            path.close_path();
        }
        path
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzPathSegment {
    Line(FuzzPoint),
    Quad(FuzzPoint, FuzzPoint),
    Cubic(FuzzPoint, FuzzPoint, FuzzPoint),
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzPoint {
    pub(crate) x: FuzzCoord,
    pub(crate) y: FuzzCoord,
}

impl FuzzPoint {
    pub(crate) fn to_tuple(&self) -> (f64, f64) {
        (self.x.to_f64(), self.y.to_f64())
    }

    pub(crate) fn to_point(&self) -> Point {
        Point::new(self.x.to_f64(), self.y.to_f64())
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzCoord {
    pub(crate) whole: i8,
    pub(crate) fraction: u8,
}

impl FuzzCoord {
    pub(crate) fn to_f64(&self) -> f64 {
        // Cover positions outside both sides of the viewport while retaining subpixel edges.
        f64::from(self.whole) + 32.0 + f64::from(self.fraction & 0b11) * 0.25
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) struct FuzzTransform {
    pub(crate) translate_x: i8,
    pub(crate) translate_y: i8,
    pub(crate) scale_x: i8,
    pub(crate) scale_y: i8,
    pub(crate) rotation: u8,
}

impl FuzzTransform {
    pub(crate) fn to_affine(&self) -> Affine {
        let translation = Affine::translate((
            f64::from(self.translate_x) * 0.5,
            f64::from(self.translate_y) * 0.5,
        ));
        let rotation = Affine::rotate(f64::from(self.rotation) * TAU / 256.0);
        let scale = Affine::scale_non_uniform(
            f64::from(self.scale_x) / 32.0,
            f64::from(self.scale_y) / 32.0,
        );
        translation * rotation * scale
    }
}

#[derive(Arbitrary, Clone, Copy, Debug)]
pub(crate) struct FuzzColor {
    pub(crate) red: u8,
    pub(crate) green: u8,
    pub(crate) blue: u8,
    pub(crate) alpha: u8,
}

impl FuzzColor {
    pub(crate) fn to_color(self) -> AlphaColor<Srgb> {
        AlphaColor::from_rgba8(self.red, self.green, self.blue, self.alpha)
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzFillRule {
    NonZero,
    EvenOdd,
}

impl FuzzFillRule {
    pub(crate) fn to_fill(&self) -> Fill {
        match self {
            Self::NonZero => Fill::NonZero,
            Self::EvenOdd => Fill::EvenOdd,
        }
    }
}

#[derive(Arbitrary, Debug)]
pub(crate) enum FuzzMix {
    Normal,
    Multiply,
    Screen,
    Darken,
    Lighten,
    Overlay,
    Difference,
}

impl FuzzMix {
    pub(crate) fn to_blend_mode(&self) -> BlendMode {
        let mix = match self {
            Self::Normal => Mix::Normal,
            Self::Multiply => Mix::Multiply,
            Self::Screen => Mix::Screen,
            Self::Darken => Mix::Darken,
            Self::Lighten => Mix::Lighten,
            Self::Overlay => Mix::Overlay,
            Self::Difference => Mix::Difference,
        };
        BlendMode::new(mix, Compose::SrcOver)
    }
}
