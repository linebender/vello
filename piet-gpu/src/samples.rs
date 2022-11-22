use crate::PicoSvg;
use piet_scene::kurbo::{Affine, BezPath, Ellipse, PathEl, Point, Rect};
use piet_scene::*;

use crate::SimpleText;

pub fn render_funky_paths(sb: &mut SceneBuilder) {
    use PathEl::*;
    let missing_movetos = &[
        LineTo((100.0, 100.0).into()),
        LineTo((100.0, 200.0).into()),
        ClosePath,
        LineTo((0.0, 400.0).into()),
        LineTo((100.0, 400.0).into()),
    ][..];
    let only_movetos = &[MoveTo((0.0, 0.0).into()), MoveTo((100.0, 100.0).into())][..];
    let empty: &[PathEl] = &[];
    sb.fill(
        Fill::NonZero,
        Affine::translate((100.0, 100.0)),
        &Color::rgb8(0, 0, 255).into(),
        None,
        &missing_movetos,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgb8(0, 0, 255).into(),
        None,
        &empty,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgb8(0, 0, 255).into(),
        None,
        &only_movetos,
    );
    sb.stroke(
        &Stroke::new(8.0),
        Affine::translate((100.0, 100.0)),
        &Color::rgb8(0, 255, 255).into(),
        None,
        &missing_movetos,
    );
}

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
                    &fill.path,
                );
            }
            Item::Stroke(stroke) => {
                sb.stroke(
                    &Stroke::new(stroke.width as f32),
                    Affine::IDENTITY,
                    &stroke.color.into(),
                    None,
                    &stroke.path,
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
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = vec![];
    for i in 1..n {
        let mut p0 = center;
        let a0 = i as f64 * dth;
        p0.x += a0.cos() * r;
        p0.y += a0.sin() * r;
        let mut p1 = center;
        let a1 = ((i * 2) % n) as f64 * dth;
        p1.x += a1.cos() * r;
        p1.y += a1.sin() * r;
        path.push(PathEl::MoveTo(p0));
        path.push(PathEl::LineTo(p1));
    }
    sb.stroke(
        &Stroke::new(2.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 0, 0)),
        None,
        &&path[..],
    );
}

#[allow(unused)]
fn render_clip_test(sb: &mut SceneBuilder) {
    const N: usize = 16;
    const X0: f64 = 50.0;
    const Y0: f64 = 450.0;
    // Note: if it gets much larger, it will exceed the 1MB scratch buffer.
    // But this is a pretty demanding test.
    const X1: f64 = 550.0;
    const Y1: f64 = 950.0;
    let step = 1.0 / ((N + 1) as f64);
    for i in 0..N {
        let t = ((i + 1) as f64) * step;
        let path = &[
            PathEl::MoveTo((X0, Y0).into()),
            PathEl::LineTo((X1, Y0).into()),
            PathEl::LineTo((X1, Y0 + t * (Y1 - Y0)).into()),
            PathEl::LineTo((X1 + t * (X0 - X1), Y1).into()),
            PathEl::LineTo((X0, Y1).into()),
            PathEl::ClosePath,
        ][..];
        sb.push_layer(Mix::Clip.into(), Affine::IDENTITY, &path);
    }
    let rect = Rect::new(X0, Y0, X1, Y1);
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 0, 0)),
        None,
        &rect,
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
        &&make_diamond(1024.0, 100.0)[..],
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgba8(0, 255, 0, 0x80).into(),
        None,
        &&make_diamond(1024.0, 125.0)[..],
    );
    sb.push_layer(
        Mix::Clip.into(),
        Affine::IDENTITY,
        &&make_diamond(1024.0, 150.0)[..],
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Color::rgba8(0, 0, 255, 0x80).into(),
        None,
        &&make_diamond(1024.0, 175.0)[..],
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
        let transform = Affine::translate((i as f64 * 225., j as f64 * 225.));
        let square = blend_square(blend.into());
        sb.append(&square, Some(transform));
    }
}

#[allow(unused)]
fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), (200., 200.));
    let linear = LinearGradient::new((0.0, 0.0), (0.0, 200.0)).stops([Color::BLACK, Color::WHITE]);
    sb.fill(Fill::NonZero, transform, &linear.into(), None, &rect);
    const GRADIENTS: &[(f64, f64, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = c.clone();
        color2.a = 0;
        let radial = RadialGradient::new((*x, *y), 100.0).stops([c.clone(), color2]);
        sb.fill(Fill::NonZero, transform, &radial.into(), None, &rect);
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    sb.push_layer(Mix::Normal.into(), transform, &rect);
    for (i, c) in COLORS.iter().enumerate() {
        // let stops = &[
        //     GradientStop {
        //         color: Color::rgb8(255, 255, 255),
        //         offset: 0.0,
        //     },
        //     GradientStop {
        //         color: c.clone(),
        //         offset: 1.0,
        //     },
        // ][..];
        let linear = LinearGradient::new((0.0, 0.0), (0.0, 200.0)).stops([Color::WHITE, c.clone()]);
        sb.push_layer(blend, transform, &rect);
        // squash the ellipse
        let a = transform
            * Affine::translate((100., 100.))
            * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
            * Affine::scale_non_uniform(1.0, 0.357)
            * Affine::translate((-100., -100.));
        sb.fill(
            Fill::NonZero,
            a,
            &linear.into(),
            None,
            &Ellipse::new((100., 100.), (90., 90.), 0.),
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
        &Rect::from_origin_size(Point::new(0.0, 0.0), (1000.0, 1000.0)),
    );
    let text_size = 60.0 + 40.0 * (0.01 * i as f32).sin();
    let s = "\u{1f600}hello piet-gpu text!";
    text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate((110.0, 600.0)),
        s,
    );
    text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate((110.0, 700.0)),
        s,
    );
    let th = (std::f64::consts::PI / 180.0) * (i as f64);
    let center = Point::new(500.0, 500.0);
    let mut p1 = center;
    p1.x += 400.0 * th.cos();
    p1.y += 400.0 * th.sin();
    sb.stroke(
        &Stroke::new(5.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 0, 0)),
        None,
        &&[PathEl::MoveTo(center), PathEl::LineTo(p1)][..],
    );
}

#[allow(unused)]
pub fn render_brush_transform(sb: &mut SceneBuilder, i: usize) {
    let th = (std::f64::consts::PI / 180.0) * (i as f64);
    let linear = LinearGradient::new((0.0, 0.0), (0.0, 200.0))
        .stops([Color::RED, Color::GREEN, Color::BLUE])
        .into();
    sb.fill(
        Fill::NonZero,
        Affine::translate((200.0, 200.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
    sb.stroke(
        &Stroke::new(40.0),
        Affine::translate((800.0, 200.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
}

fn around_center(xform: Affine, center: Point) -> Affine {
    Affine::translate(center.to_vec2()) * xform * Affine::translate(-center.to_vec2())
}

fn make_diamond(cx: f64, cy: f64) -> [PathEl; 5] {
    const SIZE: f64 = 50.0;
    [
        PathEl::MoveTo(Point::new(cx, cy - SIZE)),
        PathEl::LineTo(Point::new(cx + SIZE, cy)),
        PathEl::LineTo(Point::new(cx, cy + SIZE)),
        PathEl::LineTo(Point::new(cx - SIZE, cy)),
        PathEl::ClosePath,
    ]
}
