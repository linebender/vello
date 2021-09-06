//! Various synthetic scenes for exercising the renderer.

use rand::{Rng, RngCore};

use piet::kurbo::{BezPath, Circle, Line, Point, Rect, Shape};
use piet::{
    Color, FixedGradient, FixedLinearGradient, GradientStop, Text, TextAttribute, TextLayoutBuilder,
};

use crate::{PicoSvg, RenderContext, Vec2};

const N_CIRCLES: usize = 0;

pub fn render_svg(rc: &mut impl RenderContext, filename: &str, scale: f64) {
    let xml_str = std::fs::read_to_string(filename).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(&xml_str, scale).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub fn render_scene(rc: &mut impl RenderContext) {
    const WIDTH: usize = 2048;
    const HEIGHT: usize = 1536;
    let mut rng = rand::thread_rng();
    for _ in 0..N_CIRCLES {
        let color = Color::from_rgba32_u32(rng.next_u32());
        let center = Point::new(
            rng.gen_range(0.0, WIDTH as f64),
            rng.gen_range(0.0, HEIGHT as f64),
        );
        let radius = rng.gen_range(0.0, 50.0);
        let circle = Circle::new(center, radius);
        rc.fill(circle, &color);
    }
    let _ = rc.save();
    let mut path = BezPath::new();
    path.move_to((200.0, 150.0));
    path.line_to((100.0, 200.0));
    path.line_to((150.0, 250.0));
    path.close_path();
    rc.clip(path);

    let mut path = BezPath::new();
    path.move_to((100.0, 150.0));
    path.line_to((200.0, 200.0));
    path.line_to((150.0, 250.0));
    path.close_path();
    rc.fill(path, &Color::rgb8(128, 0, 128));
    let _ = rc.restore();
    rc.stroke(
        piet::kurbo::Line::new((100.0, 100.0), (200.0, 150.0)),
        &Color::WHITE,
        5.0,
    );
    //render_cardioid(rc);
    render_clip_test(rc);
    render_alpha_test(rc);
    render_gradient_test(rc);
    render_text_test(rc);
    //render_tiger(rc);
}

#[allow(unused)]
fn render_cardioid(rc: &mut impl RenderContext) {
    let n = 601;
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = BezPath::new();
    for i in 1..n {
        let p0 = center + Vec2::from_angle(i as f64 * dth) * r;
        let p1 = center + Vec2::from_angle(((i * 2) % n) as f64 * dth) * r;
        //rc.fill(&Circle::new(p0, 8.0), &Color::WHITE);
        path.move_to(p0);
        path.line_to(p1);
        //rc.stroke(Line::new(p0, p1), &Color::BLACK, 2.0);
    }
    rc.stroke(&path, &Color::BLACK, 2.0);
}

#[allow(unused)]
fn render_clip_test(rc: &mut impl RenderContext) {
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
        rc.save();
        let mut path = BezPath::new();
        path.move_to((X0, Y0));
        path.line_to((X1, Y0));
        path.line_to((X1, Y0 + t * (Y1 - Y0)));
        path.line_to((X1 + t * (X0 - X1), Y1));
        path.line_to((X0, Y1));
        path.close_path();
        rc.clip(path);
    }
    let rect = piet::kurbo::Rect::new(X0, Y0, X1, Y1);
    rc.fill(rect, &Color::BLACK);
    for _ in 0..N {
        rc.restore();
    }
}

#[allow(unused)]
fn render_alpha_test(rc: &mut impl RenderContext) {
    // Alpha compositing tests.
    rc.fill(
        diamond(Point::new(1024.0, 100.0)),
        &Color::Rgba32(0xff0000ff),
    );
    rc.fill(
        diamond(Point::new(1024.0, 125.0)),
        &Color::Rgba32(0x00ff0080),
    );
    rc.save();
    rc.clip(diamond(Point::new(1024.0, 150.0)));
    rc.fill(
        diamond(Point::new(1024.0, 175.0)),
        &Color::Rgba32(0x0000ff80),
    );
    rc.restore();
}

#[allow(unused)]
fn render_gradient_test(rc: &mut impl RenderContext) {
    let stops = vec![
        GradientStop {
            color: Color::rgb8(0, 255, 0),
            pos: 0.0,
        },
        GradientStop {
            color: Color::BLACK,
            pos: 1.0,
        },
    ];
    let lin = FixedLinearGradient {
        start: Point::new(0.0, 100.0),
        end: Point::new(0.0, 300.0),
        stops,
    };
    let brush = FixedGradient::Linear(lin);
    //let brush = Color::rgb8(0, 128, 0);
    rc.fill(Rect::new(100.0, 100.0, 300.0, 300.0), &brush);
}

fn diamond(origin: Point) -> impl Shape {
    let mut path = BezPath::new();
    const SIZE: f64 = 50.0;
    path.move_to((origin.x, origin.y - SIZE));
    path.line_to((origin.x + SIZE, origin.y));
    path.line_to((origin.x, origin.y + SIZE));
    path.line_to((origin.x - SIZE, origin.y));
    path.close_path();
    return path;
}

#[allow(unused)]
fn render_text_test(rc: &mut impl RenderContext) {
    rc.save();
    //rc.transform(Affine::new([0.2, 0.0, 0.0, -0.2, 200.0, 800.0]));
    let layout = rc
        .text()
        .new_text_layout("\u{1f600}hello piet-gpu text!")
        .default_attribute(TextAttribute::FontSize(100.0))
        .build()
        .unwrap();
    rc.draw_text(&layout, Point::new(110.0, 600.0));
    rc.draw_text(&layout, Point::new(110.0, 700.0));
    rc.restore();
}

#[allow(unused)]
fn render_tiger(rc: &mut impl RenderContext) {
    let xml_str = std::str::from_utf8(include_bytes!("../Ghostscript_Tiger.svg")).unwrap();
    let start = std::time::Instant::now();
    let svg = PicoSvg::load(xml_str, 8.0).unwrap();
    println!("parsing time: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub fn render_anim_frame(rc: &mut impl RenderContext, i: usize) {
    rc.fill(
        Rect::new(0.0, 0.0, 1000.0, 1000.0),
        &Color::rgb8(128, 128, 128),
    );
    let text_size = 60.0 + 40.0 * (0.01 * i as f64).sin();
    rc.save().unwrap();
    //rc.transform(Affine::new([0.2, 0.0, 0.0, -0.2, 200.0, 800.0]));
    let layout = rc
        .text()
        .new_text_layout("\u{1f600}hello piet-gpu text!")
        .default_attribute(TextAttribute::FontSize(text_size))
        .build()
        .unwrap();
    rc.draw_text(&layout, Point::new(110.0, 600.0));
    rc.draw_text(&layout, Point::new(110.0, 700.0));
    rc.restore().unwrap();
    let th = (std::f64::consts::PI / 180.0) * (i as f64);
    let center = Point::new(500.0, 500.0);
    let p1 = center + 400.0 * Vec2::from_angle(th);
    let line = Line::new(center, p1);
    rc.stroke(line, &Color::rgb8(128, 0, 0), 5.0);
}
