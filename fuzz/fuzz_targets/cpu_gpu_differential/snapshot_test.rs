// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Converts a decoded fuzz scene into a replayable `vello_test` snapshot test.

use crate::config::{DECODE_OUTPUT, TOLERANCE};
use crate::scene::{
    ClipCommand, Command, DrawCommand, DrawStyle, FuzzCap, FuzzColor, FuzzExtend, FuzzFillRule,
    FuzzGradient, FuzzGradientKind, FuzzImage, FuzzImagePaint, FuzzInterpolation, FuzzJoin,
    FuzzMix, FuzzPaint, FuzzPath, FuzzPathSegment, FuzzPoint, FuzzRect, FuzzScene, FuzzShape,
    FuzzStroke, FuzzTransform, HEIGHT, LayerKind, MAX_CLIP_DEPTH, MAX_LAYER_DEPTH, WIDTH,
};
use std::collections::BTreeSet;
use std::fmt::Write;
use std::path::Path;
use vello_common::kurbo::Affine;

struct ReplaySource {
    body: String,
    next_path: usize,
    layer_depth: usize,
    clip_depth: usize,
    /// Paint transform currently set on `ctx`, mirroring `replay.rs`.
    paint_transform: Affine,
    /// Images referenced so far; each gets an `image_<asset>` binding at the top of the test.
    images: BTreeSet<FuzzImage>,
}

impl ReplaySource {
    /// Returns the test body and whether it needs the `load_image!` macro.
    fn generate(scene: &FuzzScene) -> (String, bool) {
        let mut source = Self {
            body: String::new(),
            next_path: 0,
            layer_depth: 0,
            clip_depth: 0,
            paint_transform: Affine::IDENTITY,
            images: BTreeSet::new(),
        };
        writeln!(source.body, "    ctx.set_transform(Affine::IDENTITY);").unwrap();
        writeln!(
            source.body,
            "    ctx.set_paint({});",
            Self::color(&scene.background)
        )
        .unwrap();
        writeln!(
            source.body,
            "    ctx.fill_rect(&Rect::new(0.0, 0.0, {WIDTH}.0, {HEIGHT}.0));"
        )
        .unwrap();

        for command in &scene.commands {
            match command {
                Command::Draw(draw) => source.draw(draw),
                Command::PushLayer(layer) if source.layer_depth < MAX_LAYER_DEPTH => {
                    source.push_layer(layer);
                    source.layer_depth += 1;
                }
                Command::PopLayer if source.layer_depth > 0 => {
                    writeln!(source.body, "    ctx.pop_layer();").unwrap();
                    source.layer_depth -= 1;
                }
                Command::PushClip(clip) if source.clip_depth < MAX_CLIP_DEPTH => {
                    source.push_clip(clip);
                    source.clip_depth += 1;
                }
                Command::PopClip if source.clip_depth > 0 => {
                    writeln!(source.body, "    ctx.pop_clip_path();").unwrap();
                    source.clip_depth -= 1;
                }
                Command::PushLayer(_)
                | Command::PopLayer
                | Command::PushClip(_)
                | Command::PopClip => {}
            }
        }
        for _ in 0..source.layer_depth {
            writeln!(source.body, "    ctx.pop_layer();").unwrap();
        }
        // Image sources are obtained up front, like the fuzz harness registers them once per
        // renderer rather than per draw.
        let mut body = String::new();
        for image in &source.images {
            let asset = image.asset_name();
            writeln!(
                body,
                "    let {} = ctx.get_image_source(load_image!(\"{asset}\"));",
                Self::image_binding(*image)
            )
            .unwrap();
        }
        body.push_str(&source.body);
        (body, !source.images.is_empty())
    }

    fn image_binding(image: FuzzImage) -> String {
        format!("image_{}", image.asset_name())
    }

    fn set_transform(&mut self, transform: &FuzzTransform) {
        let [a, b, c, d, e, f] = transform.to_affine().as_coeffs();
        writeln!(
            self.body,
            "\n    ctx.set_transform(Affine::new([{a:?}, {b:?}, {c:?}, {d:?}, {e:?}, {f:?}]));"
        )
        .unwrap();
    }

    fn set_fill_rule(&mut self, fill_rule: &FuzzFillRule) {
        let fill_rule = match fill_rule {
            FuzzFillRule::NonZero => "NonZero",
            FuzzFillRule::EvenOdd => "EvenOdd",
        };
        writeln!(
            self.body,
            "    ctx.set_fill_rule(vello_common::peniko::Fill::{fill_rule});"
        )
        .unwrap();
    }

    fn draw(&mut self, draw: &DrawCommand) {
        self.set_transform(&draw.transform);
        if draw.paint.uses_paint_transform() {
            let paint_transform = draw.paint_transform.to_affine();
            if paint_transform != self.paint_transform {
                let [a, b, c, d, e, f] = paint_transform.as_coeffs();
                writeln!(
                    self.body,
                    "    ctx.set_paint_transform(Affine::new([{a:?}, {b:?}, {c:?}, {d:?}, {e:?}, {f:?}]));"
                )
                .unwrap();
                self.paint_transform = paint_transform;
            }
        }
        let paint = self.paint(&draw.paint);
        writeln!(self.body, "    ctx.set_paint({paint});").unwrap();

        match &draw.style {
            DrawStyle::Fill(fill_rule) => {
                self.set_fill_rule(fill_rule);
                match &draw.shape {
                    FuzzShape::Rect(rect) => {
                        writeln!(self.body, "    ctx.fill_rect(&{});", Self::rect(rect)).unwrap();
                    }
                    FuzzShape::Path(path) => {
                        let path = self.path(path);
                        writeln!(self.body, "    ctx.fill_path(&{path});").unwrap();
                    }
                }
            }
            DrawStyle::Stroke(stroke) => {
                writeln!(self.body, "    ctx.set_stroke({});", Self::stroke(stroke)).unwrap();
                match &draw.shape {
                    FuzzShape::Rect(rect) => {
                        writeln!(self.body, "    ctx.stroke_rect(&{});", Self::rect(rect)).unwrap();
                    }
                    FuzzShape::Path(path) => {
                        let path = self.path(path);
                        writeln!(self.body, "    ctx.stroke_path(&{path});").unwrap();
                    }
                }
            }
        }
    }

    fn push_clip(&mut self, clip: &ClipCommand) {
        self.set_transform(&clip.transform);
        self.set_fill_rule(&clip.fill_rule);
        match &clip.shape {
            FuzzShape::Rect(rect) => {
                writeln!(self.body, "    ctx.push_clip_rect(&{});", Self::rect(rect)).unwrap();
            }
            FuzzShape::Path(path) => {
                let path = self.path(path);
                writeln!(self.body, "    ctx.push_clip_path(&{path});").unwrap();
            }
        }
    }

    fn push_layer(&mut self, layer: &LayerKind) {
        match layer {
            LayerKind::Clip(shape) => {
                let path = match shape {
                    FuzzShape::Rect(rect) => format!(
                        "vello_common::kurbo::Shape::to_path(&{}, 0.1)",
                        Self::rect(rect)
                    ),
                    FuzzShape::Path(path) => self.path(path),
                };
                writeln!(self.body, "    ctx.push_clip_layer(&{path});").unwrap();
            }
            LayerKind::Opacity(opacity) => {
                let opacity = LayerKind::opacity(*opacity);
                writeln!(self.body, "    ctx.push_opacity_layer({opacity:?});").unwrap();
            }
            LayerKind::Blend(mix) => {
                let mix = match mix {
                    FuzzMix::Normal => "Normal",
                    FuzzMix::Multiply => "Multiply",
                    FuzzMix::Screen => "Screen",
                    FuzzMix::Darken => "Darken",
                    FuzzMix::Lighten => "Lighten",
                    FuzzMix::Overlay => "Overlay",
                    FuzzMix::Difference => "Difference",
                };
                writeln!(
                    self.body,
                    "    ctx.push_blend_layer(vello_common::peniko::BlendMode::new(\n\
                    \x20       vello_common::peniko::Mix::{mix},\n\
                    \x20       vello_common::peniko::Compose::SrcOver,\n\
                    \x20   ));"
                )
                .unwrap();
            }
        }
    }

    fn path(&mut self, path: &FuzzPath) -> String {
        let name = format!("path_{}", self.next_path);
        self.next_path += 1;
        writeln!(
            self.body,
            "    let mut {name} = vello_common::kurbo::BezPath::new();"
        )
        .unwrap();
        let (x, y) = path.start.to_tuple();
        writeln!(self.body, "    {name}.move_to(({x:?}, {y:?}));").unwrap();
        for segment in &path.segments {
            match segment {
                FuzzPathSegment::Line(point) => {
                    let (x, y) = point.to_tuple();
                    writeln!(self.body, "    {name}.line_to(({x:?}, {y:?}));").unwrap();
                }
                FuzzPathSegment::Quad(control, end) => {
                    let (control_x, control_y) = control.to_tuple();
                    let (end_x, end_y) = end.to_tuple();
                    writeln!(
                        self.body,
                        "    {name}.quad_to(({control_x:?}, {control_y:?}), ({end_x:?}, {end_y:?}));"
                    )
                    .unwrap();
                }
                FuzzPathSegment::Cubic(control_1, control_2, end) => {
                    let (control_1_x, control_1_y) = control_1.to_tuple();
                    let (control_2_x, control_2_y) = control_2.to_tuple();
                    let (end_x, end_y) = end.to_tuple();
                    writeln!(
                        self.body,
                        "    {name}.curve_to(\n\
                        \x20       ({control_1_x:?}, {control_1_y:?}),\n\
                        \x20       ({control_2_x:?}, {control_2_y:?}),\n\
                        \x20       ({end_x:?}, {end_y:?}),\n\
                        \x20   );"
                    )
                    .unwrap();
                }
            }
        }
        if path.close {
            writeln!(self.body, "    {name}.close_path();").unwrap();
        }
        name
    }

    fn color(color: &FuzzColor) -> String {
        format!(
            "AlphaColor::<Srgb>::from_rgba8({}, {}, {}, {})",
            color.red, color.green, color.blue, color.alpha
        )
    }

    fn point(point: &FuzzPoint) -> String {
        let (x, y) = point.to_tuple();
        format!("vello_common::kurbo::Point::new({x:?}, {y:?})")
    }

    fn paint(&mut self, paint: &FuzzPaint) -> String {
        match paint {
            FuzzPaint::Solid(color) => Self::color(color),
            FuzzPaint::Gradient(gradient) => Self::gradient(gradient),
            FuzzPaint::Image(image) => {
                self.images.insert(image.image);
                Self::image(image)
            }
        }
    }

    fn extend(extend: &FuzzExtend) -> &'static str {
        match extend {
            FuzzExtend::Pad => "Pad",
            FuzzExtend::Repeat => "Repeat",
            FuzzExtend::Reflect => "Reflect",
        }
    }

    fn image(image: &FuzzImagePaint) -> String {
        format!(
            "vello_common::paint::Image {{\n\
             \x20       image: {}.clone(),\n\
             \x20       sampler: vello_common::peniko::ImageSampler {{\n\
             \x20           x_extend: vello_common::peniko::Extend::{},\n\
             \x20           y_extend: vello_common::peniko::Extend::{},\n\
             \x20           quality: vello_common::peniko::ImageQuality::{:?},\n\
             \x20           alpha: {:?},\n\
             \x20       }},\n\
             \x20   }}",
            Self::image_binding(image.image),
            Self::extend(&image.x_extend),
            Self::extend(&image.y_extend),
            image.quality(),
            image.alpha()
        )
    }

    fn gradient(gradient: &FuzzGradient) -> String {
        let kind = match &gradient.kind {
            FuzzGradientKind::Linear { start, end } => format!(
                "vello_common::peniko::GradientKind::Linear(\n\
                 \x20           vello_common::peniko::LinearGradientPosition {{\n\
                 \x20               start: {},\n\
                 \x20               end: {},\n\
                 \x20           }},\n\
                 \x20       )",
                Self::point(start),
                Self::point(end)
            ),
            FuzzGradientKind::Radial {
                start_center,
                start_radius,
                end_center,
                end_radius,
            } => format!(
                "vello_common::peniko::GradientKind::Radial(\n\
                 \x20           vello_common::peniko::RadialGradientPosition {{\n\
                 \x20               start_center: {},\n\
                 \x20               start_radius: {:?},\n\
                 \x20               end_center: {},\n\
                 \x20               end_radius: {:?},\n\
                 \x20           }},\n\
                 \x20       )",
                Self::point(start_center),
                FuzzGradientKind::radius(*start_radius),
                Self::point(end_center),
                FuzzGradientKind::radius(*end_radius)
            ),
            FuzzGradientKind::Sweep {
                center,
                start_angle,
                sweep,
            } => format!(
                "vello_common::peniko::GradientKind::Sweep(\n\
                 \x20           vello_common::peniko::SweepGradientPosition {{\n\
                 \x20               center: {},\n\
                 \x20               start_angle: {:?},\n\
                 \x20               end_angle: {:?},\n\
                 \x20           }},\n\
                 \x20       )",
                Self::point(center),
                FuzzGradientKind::start_angle(*start_angle),
                FuzzGradientKind::end_angle(*start_angle, *sweep)
            ),
        };
        let extend = Self::extend(&gradient.extend);
        let color_space = match gradient.interpolation {
            FuzzInterpolation::Srgb => "Srgb",
            FuzzInterpolation::LinearSrgb => "LinearSrgb",
            FuzzInterpolation::Oklab => "Oklab",
        };
        let mut stops = String::new();
        for stop in gradient.stops.sorted() {
            write!(
                stops,
                "\n            vello_common::peniko::ColorStop {{\n\
                 \x20               offset: {:?},\n\
                 \x20               color: vello_common::color::DynamicColor::from_alpha_color({}),\n\
                 \x20           }},",
                stop.offset(),
                Self::color(&stop.color)
            )
            .unwrap();
        }
        format!(
            "vello_common::peniko::Gradient {{\n\
             \x20       kind: {kind},\n\
             \x20       extend: vello_common::peniko::Extend::{extend},\n\
             \x20       interpolation_cs: vello_common::color::ColorSpaceTag::{color_space},\n\
             \x20       stops: vello_common::peniko::ColorStops::from(&[{stops}\n\
             \x20       ][..]),\n\
             \x20       ..Default::default()\n\
             \x20   }}"
        )
    }

    fn stroke(stroke: &FuzzStroke) -> String {
        let join = match stroke.join {
            FuzzJoin::Bevel => "Bevel",
            FuzzJoin::Miter => "Miter",
            FuzzJoin::Round => "Round",
        };
        let cap = |cap: &FuzzCap| match cap {
            FuzzCap::Butt => "Butt",
            FuzzCap::Square => "Square",
            FuzzCap::Round => "Round",
        };
        let dashes = stroke
            .dashes
            .lengths()
            .iter()
            .map(|length| format!("{length:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "vello_common::kurbo::Stroke {{\n\
             \x20       width: {:?},\n\
             \x20       join: vello_common::kurbo::Join::{join},\n\
             \x20       miter_limit: {:?},\n\
             \x20       start_cap: vello_common::kurbo::Cap::{},\n\
             \x20       end_cap: vello_common::kurbo::Cap::{},\n\
             \x20       dash_pattern: vello_common::kurbo::Dashes::from_slice(&[{dashes}]),\n\
             \x20       dash_offset: {:?},\n\
             \x20   }}",
            stroke.width(),
            stroke.miter_limit(),
            cap(&stroke.start_cap),
            cap(&stroke.end_cap),
            stroke.dashes.offset()
        )
    }

    fn rect(rect: &FuzzRect) -> String {
        format!(
            "Rect::new({:?}, {:?}, {:?}, {:?})",
            rect.x0.to_f64(),
            rect.y0.to_f64(),
            rect.x1.to_f64(),
            rect.y1.to_f64()
        )
    }
}

/// Test function name derived from the artifact file name, e.g. `fuzz_regression_mismatch_1a2b3c`.
/// The prefix also names the reference and diff images, which keeps fuzz findings recognisable
/// among the snapshot suite's files.
fn regression_test_name(output: &Path) -> String {
    let artifact_name = output
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let artifact_name: String = artifact_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("fuzz_regression_{artifact_name}")
}

/// Writes the decoded scene as a ready-to-register snapshot test when decode mode is enabled.
///
/// Returns whether decode mode handled the input.
pub(crate) fn write(scene: &FuzzScene) -> bool {
    let Some(output) = DECODE_OUTPUT.as_ref() else {
        return false;
    };
    write_to(scene, output);
    println!("Decoded scene written to {}", output.display());
    true
}

/// Writes the scene as a snapshot test named after `output`'s file stem.
pub(crate) fn write_to(scene: &FuzzScene, output: &Path) {
    let test_name = regression_test_name(output);
    let (replay, uses_images) = ReplaySource::generate(scene);
    let load_image_import = if uses_images {
        "use crate::load_image;\n"
    } else {
        ""
    };
    // `vello_test` adds its own default GPU tolerance of one, so subtract it to reproduce the
    // fuzz target's tolerances exactly.
    let gpu_tolerance = TOLERANCE.channel.saturating_sub(1);
    let diff_pixels = TOLERANCE.max_outlier_pixels;
    let source = format!(
        "// Generated from a Vello CPU/GPU differential fuzz artifact.\n\
         // `fuzz/validate.sh <artifact>` adds it to vello_tests/tests/fuzz_regression.rs.\n\n\
         {load_image_import}\
         use crate::renderer::Renderer;\n\
         use vello_common::color::{{AlphaColor, Srgb}};\n\
         use vello_common::kurbo::{{Affine, Rect}};\n\
         use vello_dev_macros::vello_test;\n\n\
         #[vello_test(\n\
        \x20   width = {WIDTH},\n\
        \x20   height = {HEIGHT},\n\
        \x20   transparent,\n\
        \x20   gpu_tolerance = {gpu_tolerance},\n\
        \x20   diff_pixels = {diff_pixels}\n\
         )]\n\
         fn {test_name}(ctx: &mut impl Renderer) {{\n\
         {replay}\
         }}\n"
    );
    std::fs::write(output, source)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output.display()));
    // Best effort: the generated source is valid without formatting, so a missing rustfmt only
    // affects style.
    let _ = std::process::Command::new("rustfmt")
        .args(["--edition", "2024"])
        .arg(output)
        .status();
}
