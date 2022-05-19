//! Various synthetic scenes for exercising the renderer.

use rand::{Rng, RngCore};

use crate::{Blend, BlendMode, Colrv1RadialGradient, CompositionMode, PietGpuRenderContext};
use piet::kurbo::{Affine, BezPath, Circle, Line, Point, Rect, Shape};
use piet::{
    Color, GradientStop, LinearGradient, Text, TextAttribute, TextLayoutBuilder, UnitPoint,
};

use crate::{PicoSvg, RenderContext, Vec2};

const N_CIRCLES: usize = 0;

pub fn render_blend_test(rc: &mut PietGpuRenderContext, i: usize, blend: Blend) {
    rc.fill(Rect::new(400., 400., 800., 800.), &Color::rgb8(0, 0, 200));
    rc.save().unwrap();
    rc.blend(Rect::new(0., 0., 1000., 1000.), blend);
    rc.transform(Affine::translate(Vec2::new(600., 600.)) * Affine::rotate(0.01 * i as f64));
    rc.fill(Rect::new(0., 0., 400., 400.), &Color::rgba8(255, 0, 0, 255));
    rc.restore().unwrap();
}

pub fn render_svg(rc: &mut impl RenderContext, svg: &PicoSvg) {
    let start = std::time::Instant::now();
    svg.render(rc);
    println!("flattening and encoding time: {:?}", start.elapsed());
}

pub fn render_scene(rc: &mut PietGpuRenderContext) {
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
fn render_gradient_test(rc: &mut PietGpuRenderContext) {
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
    let rad = Colrv1RadialGradient {
        center0: Point::new(200.0, 200.0),
        center1: Point::new(250.0, 200.0),
        radius0: 50.0,
        radius1: 100.0,
        stops,
    };
    let brush = rc.radial_gradient_colrv1(&rad);
    //let brush = FixedGradient::Radial(rad);
    //let brush = Color::rgb8(0, 128, 0);
    let transform = Affine::new([1.0, 0.0, 0.0, 0.5, 0.0, 100.0]);
    rc.fill_transform(Rect::new(100.0, 100.0, 300.0, 300.0), &brush, transform);
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

pub fn render_blend_square(rc: &mut PietGpuRenderContext, blend: Blend) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::new(0., 0., 200., 200.);
    let stops = vec![
        GradientStop {
            color: Color::BLACK,
            pos: 0.0,
        },
        GradientStop {
            color: Color::WHITE,
            pos: 1.0,
        },
    ];
    let linear = LinearGradient::new(UnitPoint::LEFT, UnitPoint::RIGHT, stops);
    rc.fill(rect, &linear);
    const GRADIENTS: &[(f64, f64, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let stops = vec![
            GradientStop {
                color: c.clone(),
                pos: 0.0,
            },
            GradientStop {
                color: Color::rgba8(0, 0, 0, 0),
                pos: 1.0,
            },
        ];
        let rad = Colrv1RadialGradient {
            center0: Point::new(*x, *y),
            center1: Point::new(*x, *y),
            radius0: 0.0,
            radius1: 100.0,
            stops,
        };
        let brush = rc.radial_gradient_colrv1(&rad);
        rc.fill(Rect::new(0., 0., 200., 200.), &brush);
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    let _ = rc.with_save(|rc| {
        // Isolation (this can be removed for non-isolated version)
        rc.blend(rect, BlendMode::Normal.into());
        for (i, c) in COLORS.iter().enumerate() {
            let stops = vec![
                GradientStop {
                    color: Color::WHITE,
                    pos: 0.0,
                },
                GradientStop {
                    color: c.clone(),
                    pos: 1.0,
                },
            ];
            // squash the ellipse
            let a = Affine::translate((100., 100.))
                * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
                * Affine::scale_non_uniform(1.0, 0.357)
                * Affine::translate((-100., -100.));
            let linear = LinearGradient::new(UnitPoint::TOP, UnitPoint::BOTTOM, stops);
            let _ = rc.with_save(|rc| {
                rc.blend(rect, blend);
                rc.transform(a);
                rc.fill(Circle::new((100., 100.), 90.), &linear);
                Ok(())
            });
        }
        Ok(())
    });
}

pub fn render_blend_grid(rc: &mut PietGpuRenderContext) {
    const BLEND_MODES: &[BlendMode] = &[
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Darken,
        BlendMode::Screen,
        BlendMode::Lighten,
        BlendMode::Overlay,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];
    for (ix, &blend) in BLEND_MODES.iter().enumerate() {
        let _ = rc.with_save(|rc| {
            let i = ix % 4;
            let j = ix / 4;
            rc.transform(Affine::translate((i as f64 * 225., j as f64 * 225.)));
            render_blend_square(rc, blend.into());
            Ok(())
        });
    }
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
