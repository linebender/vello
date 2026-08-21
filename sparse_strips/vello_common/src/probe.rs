// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Helpers for performing probing to verify the basic capabilities of the device we are
//! running on.

use crate::color::{AlphaColor, palette::css};
use crate::filter_effects::{EdgeMode, Filter, FilterPrimitive};
#[cfg(not(feature = "std"))]
use crate::kurbo::common::FloatFuncs as _;
use crate::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape};
use crate::paint::{Image, ImageSource, PaintType};
use crate::peniko::{
    BlendMode, ColorStop, ColorStops, Compose, Extend, Gradient, ImageAlphaType, ImageQuality,
    ImageSampler, LinearGradientPosition, Mix,
};
use crate::pixmap::Pixmap;
use alloc::vec::Vec;

const REFERENCE_RGBA: &[u8] = include_bytes!("../assets/probe.rgba");

const ELEMENTS_PER_ROW: usize = 3;
const ELEMENT_MARGIN: f64 = 1.0;

const RECT_SIZE: f64 = 10.0;
const CIRCLE_RADIUS: f64 = 5.0;
const CIRCLE_CENTER_OFFSET_X: f64 = 1.5;
const IMAGE_SOURCE_SIZE: f64 = 5.0;
const PATH_TOLERANCE: f64 = 0.1;

/// The active elements used in the probe.
pub const PROBE_ELEMENTS: [ProbeFeature; 8] = [
    ProbeFeature::SolidRect,
    ProbeFeature::AlphaBlending,
    ProbeFeature::Gradient,
    ProbeFeature::ImageNearest,
    // Temporarily disabled.
    // ProbeFeature::Filter,
    ProbeFeature::ImageBilinear,
    ProbeFeature::OpacityLayer,
    ProbeFeature::Blending,
    ProbeFeature::Transformed,
];
/// Per-channel absolute tolerance used when comparing probe pixels.
const CHANNEL_TOLERANCE: u8 = 3;

/// Result of running the renderer probe.
#[derive(Debug, Clone)]
pub enum Probe<E> {
    /// The probe matched the bundled reference image.
    Success,
    /// The probe did not match the bundled reference image.
    Error(ProbeResult),
    /// Rendering the probe scene produces an error.
    RenderError(E),
}

/// Probe failure output.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// The expected probe image.
    pub expected: ProbeImage,
    /// The actual probe image.
    pub actual: ProbeImage,
}

/// A feature exercised by the renderer probe.
///
/// Each discriminant is the stable bit index used by [`ProbeStatistics::difference_mask`].
/// Existing discriminants must not be changed when features are reordered, disabled, or added.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeFeature {
    /// Drawing a solid rectangle.
    SolidRect = 0,
    /// Alpha blending overlapping shapes.
    AlphaBlending = 1,
    /// Drawing a linear gradient.
    Gradient = 2,
    /// Drawing an image with nearest-neighbor sampling.
    ImageNearest = 3,
    /// Applying a filter effect.
    Filter = 4,
    /// Drawing an image with bilinear sampling.
    ImageBilinear = 5,
    /// Drawing within a layer with reduced opacity.
    OpacityLayer = 6,
    /// Drawing within a layer with a blend mode.
    Blending = 7,
    /// Drawing with a non-identity transform.
    Transformed = 8,
}

/// Summary of the differences between the expected and actual probe images.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProbeStatistics {
    /// Number of active features exercised by the probe.
    pub element_count: u8,
    /// Width and height of the actual probe image.
    pub actual_size: (u16, u16),
    /// Number of pixels whose channels differ by more than the probe tolerance.
    pub different_pixel_count: u32,
    /// Largest absolute difference between corresponding red, green, blue, and alpha channels.
    pub max_channel_discrepancy: [u8; 4],
    /// Bitmask identifying probe features containing a pixel outside the probe tolerance.
    ///
    /// Bit `n` corresponds to the [`ProbeFeature`] whose discriminant is `n`.
    pub difference_mask: u32,
}

impl ProbeStatistics {
    /// Returns whether `feature` contained a pixel outside the probe tolerance.
    pub fn differs(&self, feature: ProbeFeature) -> bool {
        self.difference_mask & (1_u32 << feature as u8) != 0
    }
}

impl ProbeResult {
    /// Return the statistics of the probe.
    pub fn statistics(&self) -> ProbeStatistics {
        let layout = GridLayout::from_elements(&PROBE_ELEMENTS);
        let mut statistics = ProbeStatistics {
            element_count: PROBE_ELEMENTS.len() as u8,
            actual_size: (self.actual.width, self.actual.height),
            ..Default::default()
        };

        for (pixel_index, (expected, actual)) in self
            .expected
            .data
            .chunks_exact(4)
            .zip(self.actual.data.chunks_exact(4))
            .enumerate()
        {
            if expected[3] != 0 || actual[3] != 0 {
                for (max_discrepancy, (expected, actual)) in statistics
                    .max_channel_discrepancy
                    .iter_mut()
                    .zip(expected.iter().zip(actual))
                {
                    *max_discrepancy = (*max_discrepancy).max(expected.abs_diff(*actual));
                }
            }

            if !pixels_within_tolerance(expected, actual, CHANNEL_TOLERANCE) {
                statistics.different_pixel_count += 1;

                let cell_index = layout.cell_index_for_pixel(pixel_index);
                if let Some(feature) = PROBE_ELEMENTS.get(cell_index) {
                    statistics.difference_mask |= 1_u32 << *feature as u8;
                }
            }
        }

        statistics
    }
}

/// A probe image stored as RGBA8 bytes.
#[derive(Debug, Clone)]
pub struct ProbeImage {
    /// Width of the image in pixels.
    pub width: u16,
    /// Height of the image in pixels.
    pub height: u16,
    /// The image data as RGBA8 bytes.
    pub data: Vec<u8>,
}

impl<E> Probe<E> {
    /// Returns `true` when the probe matched the bundled reference image.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Construct a new probe result by inspecting the provided pixmap and comparing it
    /// against the reference output.
    pub fn from_actual(actual: Pixmap) -> Self {
        let (width, height) = canvas_size();
        let expected = ProbeImage {
            width,
            height,
            data: REFERENCE_RGBA.to_vec(),
        };
        let actual = ProbeImage::from_pixmap(actual);
        let matches_reference = expected.width == actual.width
            && expected.height == actual.height
            && expected.data.len() == actual.data.len()
            && expected
                .data
                .chunks_exact(4)
                .zip(actual.data.chunks_exact(4))
                .all(|(expected, actual)| {
                    pixels_within_tolerance(expected, actual, CHANNEL_TOLERANCE)
                });

        if matches_reference {
            Self::Success
        } else {
            Self::Error(ProbeResult { expected, actual })
        }
    }
}

impl ProbeImage {
    fn from_pixmap(pixmap: Pixmap) -> Self {
        Self {
            width: pixmap.width(),
            height: pixmap.height(),
            data: pixmap.take(ImageAlphaType::Alpha),
        }
    }
}

/// API necessary to draw the probe scene.
pub trait ProbeRenderer {
    fn set_transform(&mut self, transform: Affine);
    fn set_paint(&mut self, paint: PaintType);
    fn fill_path(&mut self, path: &BezPath);
    fn fill_rect(&mut self, rect: &Rect);
    fn push_layer(&mut self, blend_mode: Option<BlendMode>, opacity: Option<f32>);
    fn push_filter_layer(&mut self, filter: Filter);
    fn pop_layer(&mut self);
    fn set_paint_transform(&mut self, paint_transform: Affine);
    fn reset_paint_transform(&mut self);
}

#[derive(Clone, Copy, Debug)]
struct GridLayout {
    columns: usize,
    rows: usize,
    cell_width: f64,
    cell_height: f64,
}

impl GridLayout {
    fn from_elements(elements: &[ProbeFeature]) -> Self {
        let columns = ELEMENTS_PER_ROW.min(elements.len());
        let rows = elements.len().div_ceil(columns);
        let (cell_width, cell_height) = elements
            .iter()
            .copied()
            .map(ProbeFeature::bounds)
            .fold((0.0_f64, 0.0_f64), |(max_w, max_h), (w, h)| {
                (max_w.max(w), max_h.max(h))
            });

        Self {
            columns,
            rows,
            cell_width,
            cell_height,
        }
    }

    fn canvas_size(self) -> (u16, u16) {
        let (cell_stride_x, cell_stride_y) = self.cell_stride();
        // Margin only exists between cells, so subtract one.
        let width = self.columns as f64 * cell_stride_x - ELEMENT_MARGIN;
        let height = self.rows as f64 * cell_stride_y - ELEMENT_MARGIN;
        (width.ceil() as u16, height.ceil() as u16)
    }

    fn canvas_rect(self) -> Rect {
        let (width, height) = self.canvas_size();
        Rect::new(0.0, 0.0, f64::from(width), f64::from(height))
    }

    fn cell_stride(self) -> (f64, f64) {
        (
            self.cell_width + ELEMENT_MARGIN,
            self.cell_height + ELEMENT_MARGIN,
        )
    }

    fn cell_rect(self, index: usize) -> Rect {
        let column = index % self.columns;
        let row = index / self.columns;
        let (cell_stride_x, cell_stride_y) = self.cell_stride();
        let x0 = column as f64 * cell_stride_x;
        let y0 = row as f64 * cell_stride_y;
        Rect::new(x0, y0, x0 + self.cell_width, y0 + self.cell_height)
    }

    fn cell_index_for_pixel(self, pixel_index: usize) -> usize {
        let (cell_stride_x, cell_stride_y) = self.cell_stride();
        let image_width = usize::from(self.canvas_size().0);
        let x = pixel_index % image_width;
        let y = pixel_index / image_width;
        let column = x / cell_stride_x as usize;
        let row = y / cell_stride_y as usize;
        row * self.columns + column
    }
}

impl ProbeFeature {
    fn bounds(self) -> (f64, f64) {
        let (width, height) = match self {
            Self::SolidRect
            | Self::Gradient
            | Self::ImageNearest
            | Self::ImageBilinear
            | Self::Filter
            | Self::OpacityLayer => (RECT_SIZE, RECT_SIZE),
            Self::Transformed => (
                RECT_SIZE * core::f64::consts::SQRT_2,
                RECT_SIZE * core::f64::consts::SQRT_2,
            ),
            Self::AlphaBlending | Self::Blending => (
                CIRCLE_RADIUS * 2.0 + CIRCLE_CENTER_OFFSET_X * 2.0,
                CIRCLE_RADIUS * 2.0,
            ),
        };
        (width + ELEMENT_MARGIN * 2.0, height + ELEMENT_MARGIN * 2.0)
    }
}

/// Return the canvas size of the shared probe scene.
pub fn canvas_size() -> (u16, u16) {
    GridLayout::from_elements(&PROBE_ELEMENTS).canvas_size()
}

/// Return the pixmap that is referenced when drawing images in the scene.
pub fn probe_image_pixmap() -> Pixmap {
    let mut pixmap = Pixmap::new(IMAGE_SOURCE_SIZE as u16, IMAGE_SOURCE_SIZE as u16);
    for y in 0..pixmap.height() {
        for x in 0..pixmap.width() {
            pixmap.set_pixel(
                x,
                y,
                AlphaColor::from_rgba8(255, 0, 0, 255)
                    .premultiply()
                    .to_rgba8(),
            );
        }
    }
    pixmap.set_may_have_transparency(false);
    pixmap
}

fn image_paint(image: ImageSource, quality: ImageQuality) -> PaintType {
    Image {
        image,
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        },
    }
    .into()
}

/// Draw the full shared probe scene into a rendering context.
pub fn draw_scene<T: ProbeRenderer>(ctx: &mut T, image: ImageSource) {
    let layout = GridLayout::from_elements(&PROBE_ELEMENTS);
    let image_nearest = image_paint(image.clone(), ImageQuality::Low);
    let image_bilinear = image_paint(image, ImageQuality::Medium);
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_paint(css::WHITE.into());
    ctx.fill_rect(&layout.canvas_rect());

    for (index, element) in PROBE_ELEMENTS.iter().copied().enumerate() {
        draw_probe_element(
            ctx,
            layout.cell_rect(index),
            element,
            &image_nearest,
            &image_bilinear,
        );
    }
}

fn pixels_within_tolerance(expected: &[u8], actual: &[u8], channel_tolerance: u8) -> bool {
    if expected[3] == 0 && actual[3] == 0 {
        return true;
    }

    expected
        .iter()
        .zip(actual)
        .all(|(expected, actual)| expected.abs_diff(*actual) <= channel_tolerance)
}

fn draw_probe_element(
    ctx: &mut impl ProbeRenderer,
    cell: Rect,
    element: ProbeFeature,
    image_nearest: &PaintType,
    image_bilinear: &PaintType,
) {
    match element {
        ProbeFeature::SolidRect => {
            ctx.set_paint(css::BLUE.into());
            ctx.fill_rect(&centered_rect(cell, RECT_SIZE, RECT_SIZE));
        }
        ProbeFeature::Transformed => {
            draw_transformed_rect(ctx, centered_rect(cell, RECT_SIZE, RECT_SIZE));
        }
        ProbeFeature::AlphaBlending => {
            let center = cell.center();
            ctx.set_paint(css::YELLOW.with_alpha(0.5).into());
            ctx.fill_path(
                &Circle::new((center.x - CIRCLE_CENTER_OFFSET_X, center.y), CIRCLE_RADIUS)
                    .to_path(PATH_TOLERANCE),
            );
            ctx.set_paint(css::GREEN.with_alpha(0.5).into());
            ctx.fill_path(
                &Circle::new((center.x + CIRCLE_CENTER_OFFSET_X, center.y), CIRCLE_RADIUS)
                    .to_path(PATH_TOLERANCE),
            );
        }
        ProbeFeature::Gradient => {
            let rect = centered_rect(cell, RECT_SIZE, RECT_SIZE);
            ctx.set_paint(linear_gradient(&rect).into());
            ctx.fill_rect(&rect);
        }
        ProbeFeature::ImageNearest => draw_centered_padded_image(ctx, cell, image_nearest),
        ProbeFeature::Filter => draw_blurred_rect(ctx, centered_rect(cell, RECT_SIZE, RECT_SIZE)),
        ProbeFeature::ImageBilinear => draw_centered_padded_image(ctx, cell, image_bilinear),
        ProbeFeature::OpacityLayer => {
            draw_opacity_layer_rect(ctx, centered_rect(cell, RECT_SIZE, RECT_SIZE));
        }
        ProbeFeature::Blending => draw_layered_difference_circles(ctx, cell),
    }
}

fn centered_rect(cell: Rect, width: f64, height: f64) -> Rect {
    let center = cell.center();
    Rect::new(
        center.x - width * 0.5,
        center.y - height * 0.5,
        center.x + width * 0.5,
        center.y + height * 0.5,
    )
}

fn draw_centered_padded_image(ctx: &mut impl ProbeRenderer, cell: Rect, image_paint: &PaintType) {
    let dst_rect = centered_rect(cell, RECT_SIZE, RECT_SIZE);
    let image_origin = (
        dst_rect.x0 + (RECT_SIZE - IMAGE_SOURCE_SIZE) * 0.5,
        dst_rect.y0 + (RECT_SIZE - IMAGE_SOURCE_SIZE) * 0.5,
    );
    ctx.set_paint(image_paint.clone());
    ctx.set_paint_transform(Affine::translate(image_origin));
    ctx.fill_rect(&dst_rect);
    ctx.reset_paint_transform();
}

fn draw_transformed_rect(ctx: &mut impl ProbeRenderer, rect: Rect) {
    let center = rect.center();
    ctx.set_transform(
        Affine::translate((center.x, center.y))
            * Affine::rotate(core::f64::consts::FRAC_PI_4)
            * Affine::translate((-center.x, -center.y)),
    );
    ctx.set_paint(css::BLUE.into());
    ctx.fill_rect(&rect);
    ctx.set_transform(Affine::IDENTITY);
}

#[allow(dead_code, reason = "Will be re-enabled in the future.")]
fn draw_blurred_rect(ctx: &mut impl ProbeRenderer, rect: Rect) {
    let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 0.5,
        edge_mode: EdgeMode::None,
    });
    ctx.push_filter_layer(blur);
    ctx.set_paint(css::REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

fn draw_opacity_layer_rect(ctx: &mut impl ProbeRenderer, rect: Rect) {
    ctx.push_layer(None, Some(0.5));
    ctx.set_paint(css::ORANGE_RED.into());
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

fn draw_layered_difference_circles(ctx: &mut impl ProbeRenderer, cell: Rect) {
    let center = cell.center();

    ctx.push_layer(None, None);
    ctx.set_paint(css::YELLOW.with_alpha(0.5).into());
    ctx.fill_path(
        &Circle::new((center.x - CIRCLE_CENTER_OFFSET_X, center.y), CIRCLE_RADIUS)
            .to_path(PATH_TOLERANCE),
    );

    ctx.push_layer(
        Some(BlendMode::new(Mix::Difference, Compose::SrcOver)),
        None,
    );
    ctx.set_paint(css::GREEN.with_alpha(0.5).into());
    ctx.fill_path(
        &Circle::new((center.x + CIRCLE_CENTER_OFFSET_X, center.y), CIRCLE_RADIUS)
            .to_path(PATH_TOLERANCE),
    );
    ctx.pop_layer();
    ctx.pop_layer();
}

fn linear_gradient(rect: &Rect) -> Gradient {
    Gradient {
        kind: LinearGradientPosition {
            start: Point::new(rect.x0, rect.y0),
            end: Point::new(rect.x1, rect.y0),
        }
        .into(),
        stops: ColorStops::from(
            [
                ColorStop::from((0.0, css::BLUE)),
                ColorStop::from((1.0, css::RED)),
            ]
            .as_slice(),
        ),
        extend: Extend::Pad,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn probe_result_reports_pixel_and_cell_differences() {
        let (width, height) = canvas_size();
        let pixel_count = usize::from(width) * usize::from(height);
        let expected = ProbeImage {
            width,
            height,
            data: vec![255; pixel_count * 4],
        };
        let mut actual = expected.clone();
        let layout = GridLayout::from_elements(&PROBE_ELEMENTS);

        let set_channel = |actual: &mut ProbeImage, cell_index: usize, channel: usize, value| {
            let center = layout.cell_rect(cell_index).center();
            let x = center.x.floor() as usize;
            let y = center.y.floor() as usize;
            actual.data[(y * usize::from(width) + x) * 4 + channel] = value;
        };

        // This stays within the probe tolerance.
        set_channel(&mut actual, 0, 0, 254);

        set_channel(&mut actual, 1, 0, 249);
        set_channel(&mut actual, 5, 1, 0);
        set_channel(&mut actual, 5, 3, 100);

        let result = ProbeResult { expected, actual };
        let statistics = result.statistics();
        assert_eq!(
            statistics,
            ProbeStatistics {
                element_count: PROBE_ELEMENTS.len() as u8,
                actual_size: (width, height),
                different_pixel_count: 2,
                max_channel_discrepancy: [6, 255, 0, 155],
                difference_mask: (1 << 1) | (1 << 6),
            }
        );
        assert!(statistics.differs(ProbeFeature::AlphaBlending));
        assert!(statistics.differs(ProbeFeature::OpacityLayer));
        assert!(!statistics.differs(ProbeFeature::Filter));
        assert!(!statistics.differs(ProbeFeature::ImageBilinear));
    }

    #[test]
    fn probe_statistics_reports_actual_size() {
        let result = ProbeResult {
            expected: ProbeImage {
                width: 1,
                height: 1,
                data: vec![0; 4],
            },
            actual: ProbeImage {
                width: 2,
                height: 1,
                data: vec![0; 8],
            },
        };

        assert_eq!(result.statistics().actual_size, (2, 1));
    }
}
