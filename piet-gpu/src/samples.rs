use crate::PicoSvg;
use kurbo::BezPath;
use piet_scene::*;

use crate::SimpleText;

#[allow(unused)]
const N_CIRCLES: usize = 0;

#[allow(unused)]
pub fn render_svg(sb: &mut SceneBuilder, svg: &PicoSvg, print_stats: bool) {
    use crate::pico_svg::*;
    let start = std::time::Instant::now();
    for item in &svg.items {
        match item {
            Item::Fill(fill) => {
                sb.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    &fill.color.into(),
                    None,
                    convert_bez_path(&fill.path),
                );
            }
            Item::Stroke(stroke) => {
                sb.stroke(
                    &simple_stroke(stroke.width as f32),
                    Affine::IDENTITY,
                    &stroke.color.into(),
                    None,
                    convert_bez_path(&stroke.path),
                );
            }
        }
    }
    if print_stats {
        println!("flattening and encoding time: {:?}", start.elapsed());
    }
}

#[allow(unused)]
pub fn render_tiger(sb: &mut SceneBuilder, print_stats: bool) {
    use super::pico_svg::*;
    let xml_str = std::str::from_utf8(include_bytes!("../Ghostscript_Tiger.svg")).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(xml_str, 8.0).unwrap();
    if print_stats {
        println!("parsing time: {:?}", start.elapsed());
    }
    render_svg(sb, &svg, print_stats);
}

pub fn render_scene(sb: &mut SceneBuilder) {
    render_cardioid(sb);
    render_clip_test(sb);
    render_alpha_test(sb);
    //render_tiger(sb, false);
}

#[allow(unused)]
fn render_cardioid(sb: &mut SceneBuilder) {
    let n = 601;
    let dth = std::f32::consts::PI * 2.0 / (n as f32);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = vec![];
    for i in 1..n {
        let mut p0 = center;
        let a0 = i as f32 * dth;
        p0.x += a0.cos() * r;
        p0.y += a0.sin() * r;
        let mut p1 = center;
        let a1 = ((i * 2) % n) as f32 * dth;
        p1.x += a1.cos() * r;
        p1.y += a1.sin() * r;
        path.push(PathElement::MoveTo(p0));
        path.push(PathElement::LineTo(p1));
    }
    sb.stroke(
        &simple_stroke(2.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 0, 0)),
        None,
        &path,
    );
}

#[allow(unused)]
fn render_clip_test(sb: &mut SceneBuilder) {
    const N: usize = 16;
    const X0: f32 = 50.0;
    const Y0: f32 = 450.0;
    // Note: if it gets much larger, it will exceed the 1MB scratch buffer.
    // But this is a pretty demanding test.
    const X1: f32 = 550.0;
    const Y1: f32 = 950.0;
    let step = 1.0 / ((N + 1) as f32);
    for i in 0..N {
        let t = ((i + 1) as f32) * step;
        let path = &[
            PathElement::MoveTo((X0, Y0).into()),
            PathElement::LineTo((X1, Y0).into()),
            PathElement::LineTo((X1, Y0 + t * (Y1 - Y0)).into()),
            PathElement::LineTo((X1 + t * (X0 - X1), Y1).into()),
            PathElement::LineTo((X0, Y1).into()),
            PathElement::Close,
        ];
        sb.push_layer(Mix::Clip.into(), Affine::IDENTITY, path);
    }
    let rect = Rect {
        min: Point::new(X0, Y0),
        max: Point::new(X1, Y1),
    };
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 0, 0)),
        None,
        rect.elements(),
    );
    for _ in 0..N {
        sb.pop_layer();
    }
}

#[allow(unused)]
fn render_alpha_test(sb: &mut SceneBuilder) {
    // Alpha compositing tests.
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgb8(255, 0, 0).into(),
        None,
        make_diamond(1024.0, 100.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgba8(0, 255, 0, 0x80).into(),
        None,
        make_diamond(1024.0, 125.0),
    );
    sb.push_layer(
        Mix::Clip.into(),
        Affine::IDENTITY,
        make_diamond(1024.0, 150.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgba8(0, 0, 255, 0x80).into(),
        None,
        make_diamond(1024.0, 175.0),
    );
    sb.pop_layer();
}

#[allow(unused)]
pub fn render_blend_grid(sb: &mut SceneBuilder) {
    const BLEND_MODES: &[Mix] = &[
        Mix::Normal,
        Mix::Multiply,
        Mix::Darken,
        Mix::Screen,
        Mix::Lighten,
        Mix::Overlay,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];
    for (ix, &blend) in BLEND_MODES.iter().enumerate() {
        let i = ix % 4;
        let j = ix / 4;
        let transform = Affine::translate(i as f32 * 225., j as f32 * 225.);
        let square = blend_square(blend.into());
        sb.append(&square, Some(transform));
    }
}

#[allow(unused)]
fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), 200., 200.);
    let stops = &[
        GradientStop {
            color: Color::rgb8(0, 0, 0),
            offset: 0.0,
        },
        GradientStop {
            color: Color::rgb8(255, 255, 255),
            offset: 1.0,
        },
    ][..];
    let linear = Brush::LinearGradient(LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(200.0, 0.0),
        stops: stops.into(),
        extend: ExtendMode::Pad,
    });
    sb.fill(Fill::NonZero, transform, &linear, None, rect.elements());
    const GRADIENTS: &[(f32, f32, Color)] = &[
        (150., 0., Color::rgb8(64, 240, 255)),
        (175., 100., Color::rgb8(240, 96, 255)),
        (125., 200., Color::rgb8(255, 192, 64)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = c.clone();
        color2.a = 0;
        let stops = &[
            GradientStop {
                color: c.clone(),
                offset: 0.0,
            },
            GradientStop {
                color: color2,
                offset: 1.0,
            },
        ][..];
        let rad = Brush::RadialGradient(RadialGradient {
            center0: Point::new(*x, *y),
            center1: Point::new(*x, *y),
            radius0: 0.0,
            radius1: 100.0,
            stops: stops.into(),
            extend: ExtendMode::Pad,
        });
        sb.fill(Fill::NonZero, transform, &rad, None, rect.elements());
    }
    const COLORS: &[Color] = &[
        Color::rgb8(0, 0, 255),
        Color::rgb8(0, 255, 0),
        Color::rgb8(255, 0, 0),
    ];
    sb.push_layer(Mix::Normal.into(), transform, rect.elements());
    for (i, c) in COLORS.iter().enumerate() {
        let stops = &[
            GradientStop {
                color: Color::rgb8(255, 255, 255),
                offset: 0.0,
            },
            GradientStop {
                color: c.clone(),
                offset: 1.0,
            },
        ][..];
        let linear = Brush::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(0.0, 200.0),
            stops: stops.into(),
            extend: ExtendMode::Pad,
        });
        sb.push_layer(blend, transform, rect.elements());
        // squash the ellipse
        let a = transform
            * Affine::translate(100., 100.)
            * Affine::rotate(std::f32::consts::FRAC_PI_3 * (i * 2 + 1) as f32)
            * Affine::scale(1.0, 0.357)
            * Affine::translate(-100., -100.);
        sb.fill(
            Fill::NonZero,
            a,
            &linear,
            None,
            make_ellipse(100., 100., 90., 90.),
        );
        sb.pop_layer();
    }
    sb.pop_layer();
}

#[allow(unused)]
fn blend_square(blend: BlendMode) -> SceneFragment {
    let mut fragment = SceneFragment::default();
    let mut sb = SceneBuilder::for_fragment(&mut fragment);
    render_blend_square(&mut sb, blend, Affine::IDENTITY);
    sb.finish();
    fragment
}

#[allow(unused)]
pub fn render_anim_frame(sb: &mut SceneBuilder, text: &mut SimpleText, i: usize) {
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 128, 128)),
        None,
        Rect::from_origin_size(Point::new(0.0, 0.0), 1000.0, 1000.0).elements(),
    );
    let text_size = 60.0 + 40.0 * (0.01 * i as f32).sin();
    let s = "\u{1f600}hello piet-gpu text!";
    text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate(110.0, 600.0),
        s,
    );
    text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate(110.0, 700.0),
        s,
    );
    let th = (std::f32::consts::PI / 180.0) * (i as f32);
    let center = Point::new(500.0, 500.0);
    let mut p1 = center;
    p1.x += 400.0 * th.cos();
    p1.y += 400.0 * th.sin();
    sb.stroke(
        &simple_stroke(5.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 0, 0)),
        None,
        &[PathElement::MoveTo(center), PathElement::LineTo(p1)],
    );
}

#[allow(unused)]
pub fn render_brush_transform(sb: &mut SceneBuilder, i: usize) {
    let th = (std::f32::consts::PI / 180.0) * (i as f32);
    let stops = &[
        GradientStop {
            color: Color::rgb8(255, 0, 0),
            offset: 0.0,
        },
        GradientStop {
            color: Color::rgb8(0, 255, 0),
            offset: 0.5,
        },
        GradientStop {
            color: Color::rgb8(0, 0, 255),
            offset: 1.0,
        },
    ][..];
    let linear = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(0.0, 200.0),
        stops: stops.into(),
        extend: ExtendMode::Pad,
    }
    .into();
    sb.fill(
        Fill::NonZero,
        Affine::translate(200.0, 200.0),
        &linear,
        Some(Affine::rotate(th).around_center(200.0, 100.0)),
        Rect::from_origin_size(Point::default(), 400.0, 200.0).elements(),
    );
    sb.stroke(
        &simple_stroke(40.0),
        Affine::translate(800.0, 200.0),
        &linear,
        Some(Affine::rotate(th).around_center(200.0, 100.0)),
        Rect::from_origin_size(Point::default(), 400.0, 200.0).elements(),
    );

}

fn convert_bez_path<'a>(path: &'a BezPath) -> impl Iterator<Item = PathElement> + 'a + Clone {
    path.elements()
        .iter()
        .map(|el| PathElement::from_kurbo(*el))
}

fn make_ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> impl Iterator<Item = PathElement> + Clone {
    let a = 0.551915024494;
    let arx = a * rx;
    let ary = a * ry;
    let elements = [
        PathElement::MoveTo(Point::new(cx + rx, cy)),
        PathElement::CurveTo(
            Point::new(cx + rx, cy + ary),
            Point::new(cx + arx, cy + ry),
            Point::new(cx, cy + ry),
        ),
        PathElement::CurveTo(
            Point::new(cx - arx, cy + ry),
            Point::new(cx - rx, cy + ary),
            Point::new(cx - rx, cy),
        ),
        PathElement::CurveTo(
            Point::new(cx - rx, cy - ary),
            Point::new(cx - arx, cy - ry),
            Point::new(cx, cy - ry),
        ),
        PathElement::CurveTo(
            Point::new(cx + arx, cy - ry),
            Point::new(cx + rx, cy - ary),
            Point::new(cx + rx, cy),
        ),
        PathElement::Close,
    ];
    (0..elements.len()).map(move |i| elements[i])
}

fn make_diamond(cx: f32, cy: f32) -> impl Iterator<Item = PathElement> + Clone {
    const SIZE: f32 = 50.0;
    let elements = [
        PathElement::MoveTo(Point::new(cx, cy - SIZE)),
        PathElement::LineTo(Point::new(cx + SIZE, cy)),
        PathElement::LineTo(Point::new(cx, cy + SIZE)),
        PathElement::LineTo(Point::new(cx - SIZE, cy)),
        PathElement::Close,
    ];
    (0..elements.len()).map(move |i| elements[i])
}

fn simple_stroke(width: f32) -> Stroke<[f32; 0]> {
    Stroke {
        width,
        join: Join::Round,
        miter_limit: 1.4,
        start_cap: Cap::Round,
        end_cap: Cap::Round,
        dash_pattern: [],
        dash_offset: 0.0,
        scale: true,
    }
}
