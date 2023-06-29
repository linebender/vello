use crate::{ExampleScene, SceneConfig, SceneParams, SceneSet};
use vello::kurbo::{Affine, BezPath, Ellipse, PathEl, Point, Rect};
use vello::peniko::*;
use vello::*;

const FLOWER_IMAGE: &[u8] = include_bytes!("../../assets/splash-flower.jpg");

macro_rules! scene {
    ($name: ident) => {
        scene!($name: false)
    };
    ($name: ident: animated) => {
        scene!($name: true)
    };
    ($name: ident: $animated: literal) => {
        ExampleScene {
            config: SceneConfig {
                animated: $animated,
                name: stringify!($name).to_owned(),
            },
            function: Box::new($name),
        }
    };
}

pub fn test_scenes() -> SceneSet {
    let splash_scene = ExampleScene {
        config: SceneConfig {
            animated: false,
            name: "splash_with_tiger".to_owned(),
        },
        function: Box::new(splash_with_tiger()),
    };
    let mmark_scene = ExampleScene {
        config: SceneConfig {
            animated: false,
            name: "mmark".to_owned(),
        },
        function: Box::new(crate::mmark::MMark::new(80_000)),
    };
    let scenes = vec![
        splash_scene,
        mmark_scene,
        scene!(funky_paths),
        scene!(cardioid_and_friends),
        scene!(animated_text: animated),
        scene!(gradient_extend),
        scene!(two_point_radial),
        scene!(brush_transform: animated),
        scene!(blend_grid),
        scene!(conflation_artifacts),
        scene!(labyrinth),
        scene!(base_color_test: animated),
        scene!(clip_test: animated),
    ];

    SceneSet { scenes }
}

// Scenes

fn funky_paths(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;
    let missing_movetos = [
        LineTo((100.0, 100.0).into()),
        LineTo((100.0, 200.0).into()),
        ClosePath,
        LineTo((0.0, 400.0).into()),
        LineTo((100.0, 400.0).into()),
    ];
    let only_movetos = [MoveTo((0.0, 0.0).into()), MoveTo((100.0, 100.0).into())];
    let empty: [PathEl; 0] = [];
    sb.fill(
        Fill::NonZero,
        Affine::translate((100.0, 100.0)),
        Color::rgb8(0, 0, 255),
        None,
        &missing_movetos,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(0, 0, 255),
        None,
        &empty,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(0, 0, 255),
        None,
        &only_movetos,
    );
    sb.stroke(
        &Stroke::new(8.0),
        Affine::translate((100.0, 100.0)),
        Color::rgb8(0, 255, 255),
        None,
        &missing_movetos,
    );
}

fn cardioid_and_friends(sb: &mut SceneBuilder, _: &mut SceneParams) {
    render_cardioid(sb);
    render_clip_test(sb);
    render_alpha_test(sb);
    //render_tiger(sb, false);
}

fn animated_text(sb: &mut SceneBuilder, params: &mut SceneParams) {
    // Uses the static array address as a cache key for expedience. Real code
    // should use a better strategy.
    let piet_logo = params
        .images
        .from_bytes(FLOWER_IMAGE.as_ptr() as usize, FLOWER_IMAGE)
        .unwrap();

    use PathEl::*;
    let rect = Rect::from_origin_size(Point::new(0.0, 0.0), (1000.0, 1000.0));
    let star = [
        MoveTo((50.0, 0.0).into()),
        LineTo((21.0, 90.0).into()),
        LineTo((98.0, 35.0).into()),
        LineTo((2.0, 35.0).into()),
        LineTo((79.0, 90.0).into()),
        ClosePath,
    ];
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 128, 128)),
        None,
        &rect,
    );
    let text_size = 60.0 + 40.0 * (params.time as f32).sin();
    let s = "\u{1f600}hello vello text!";
    params.text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate((110.0, 600.0)),
        s,
    );
    params.text.add_run(
        sb,
        None,
        text_size,
        Color::WHITE,
        Affine::translate((110.0, 700.0)),
        // Add a skew to simulate an oblique font.
        Some(Affine::skew(20f64.to_radians().tan(), 0.0)),
        &Stroke::new(1.0),
        s,
    );
    let t = ((params.time).sin() * 0.5 + 0.5) as f32;
    let weight = t * 700.0 + 200.0;
    let width = t * 150.0 + 50.0;
    params.text.add_var_run(
        sb,
        None,
        72.0,
        &[("wght", weight), ("wdth", width)],
        Color::WHITE,
        Affine::translate((110.0, 800.0)),
        // Add a skew to simulate an oblique font.
        None,
        Fill::NonZero,
        "And some vello\ntext with a newline",
    );
    let th = params.time;
    let center = Point::new(500.0, 500.0);
    let mut p1 = center;
    p1.x += 400.0 * th.cos();
    p1.y += 400.0 * th.sin();
    sb.stroke(
        &Stroke::new(5.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 0, 0)),
        None,
        &[PathEl::MoveTo(center), PathEl::LineTo(p1)],
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((150.0, 150.0)) * Affine::scale(0.2),
        Color::RED,
        None,
        &rect,
    );
    let alpha = params.time.sin() as f32 * 0.5 + 0.5;
    sb.push_layer(Mix::Normal, alpha, Affine::IDENTITY, &rect);
    sb.fill(
        Fill::NonZero,
        Affine::translate((100.0, 100.0)) * Affine::scale(0.2),
        Color::BLUE,
        None,
        &rect,
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((200.0, 200.0)) * Affine::scale(0.2),
        Color::GREEN,
        None,
        &rect,
    );
    sb.pop_layer();
    sb.fill(
        Fill::NonZero,
        Affine::translate((400.0, 100.0)),
        Color::PURPLE,
        None,
        &star,
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((500.0, 100.0)),
        Color::PURPLE,
        None,
        &star,
    );
    sb.draw_image(
        &piet_logo,
        Affine::translate((800.0, 50.0)) * Affine::rotate(20f64.to_radians()),
    );
}

fn brush_transform(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let th = params.time;
    let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([
        Color::RED,
        Color::GREEN,
        Color::BLUE,
    ]);
    sb.fill(
        Fill::NonZero,
        Affine::rotate(25f64.to_radians()) * Affine::scale_non_uniform(2.0, 1.0),
        &Gradient::new_radial((200.0, 200.0), 80.0).with_stops([
            Color::RED,
            Color::GREEN,
            Color::BLUE,
        ]),
        None,
        &Rect::from_origin_size((100.0, 100.0), (200.0, 200.0)),
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((200.0, 600.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
    sb.stroke(
        &Stroke::new(40.0),
        Affine::translate((800.0, 600.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
}

fn gradient_extend(sb: &mut SceneBuilder, params: &mut SceneParams) {
    fn square(sb: &mut SceneBuilder, is_radial: bool, transform: Affine, extend: Extend) {
        let colors = [Color::RED, Color::rgb8(0, 255, 0), Color::BLUE];
        let width = 300f64;
        let height = 300f64;
        let gradient: Brush = if is_radial {
            let center = (width * 0.5, height * 0.5);
            let radius = (width * 0.25) as f32;
            Gradient::new_two_point_radial(center, radius * 0.25, center, radius)
                .with_stops(colors)
                .with_extend(extend)
                .into()
        } else {
            Gradient::new_linear((width * 0.35, height * 0.5), (width * 0.65, height * 0.5))
                .with_stops(colors)
                .with_extend(extend)
                .into()
        };
        sb.fill(
            Fill::NonZero,
            transform,
            &gradient,
            None,
            &Rect::new(0.0, 0.0, width, height),
        );
    }
    let extend_modes = [Extend::Pad, Extend::Repeat, Extend::Reflect];
    for (x, extend) in extend_modes.iter().enumerate() {
        for y in 0..2 {
            let is_radial = y & 1 != 0;
            let transform = Affine::translate((x as f64 * 350.0 + 50.0, y as f64 * 350.0 + 100.0));
            square(sb, is_radial, transform, *extend);
        }
    }
    for (i, label) in ["Pad", "Repeat", "Reflect"].iter().enumerate() {
        let x = i as f64 * 350.0 + 50.0;
        params.text.add(
            sb,
            None,
            32.0,
            Some(&Color::WHITE.into()),
            Affine::translate((x, 70.0)),
            label,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn two_point_radial(sb: &mut SceneBuilder, _params: &mut SceneParams) {
    fn make(
        sb: &mut SceneBuilder,
        x0: f64,
        y0: f64,
        r0: f32,
        x1: f64,
        y1: f64,
        r1: f32,
        transform: Affine,
        extend: Extend,
    ) {
        let colors = [Color::RED, Color::YELLOW, Color::rgb8(6, 85, 186)];
        let width = 400f64;
        let height = 200f64;
        let rect = Rect::new(0.0, 0.0, width, height);
        sb.fill(Fill::NonZero, transform, Color::WHITE, None, &rect);
        sb.fill(
            Fill::NonZero,
            transform,
            &Gradient::new_two_point_radial((x0, y0), r0, (x1, y1), r1)
                .with_stops(colors)
                .with_extend(extend),
            None,
            &Rect::new(0.0, 0.0, width, height),
        );
        let r0 = r0 as f64 - 1.0;
        let r1 = r1 as f64 - 1.0;
        let stroke_width = 1.0;
        sb.stroke(
            &Stroke::new(stroke_width),
            transform,
            Color::BLACK,
            None,
            &Ellipse::new((x0, y0), (r0, r0), 0.0),
        );
        sb.stroke(
            &Stroke::new(stroke_width),
            transform,
            Color::BLACK,
            None,
            &Ellipse::new((x1, y1), (r1, r1), 0.0),
        );
    }

    // These demonstrate radial gradient patterns similar to the examples shown
    // at <https://learn.microsoft.com/en-us/typography/opentype/spec/colr#radial-gradients>

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let y = 100.0;
        let x0 = 140.0;
        let x1 = x0 + 140.0;
        let r0 = 20.0;
        let r1 = 50.0;
        make(
            sb,
            x0,
            y,
            r0,
            x1,
            y,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 20.0)),
            *mode,
        );
    }

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let y = 100.0;
        let x0 = 140.0;
        let x1 = x0 + 140.0;
        let r0 = 20.0;
        let r1 = 50.0;
        make(
            sb,
            x1,
            y,
            r1,
            x0,
            y,
            r0,
            Affine::translate((i as f64 * 420.0 + 20.0, 240.0)),
            *mode,
        );
    }

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let y = 100.0;
        let x0 = 140.0;
        let x1 = x0 + 140.0;
        let r0 = 50.0;
        let r1 = 50.0;
        make(
            sb,
            x0,
            y,
            r0,
            x1,
            y,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 460.0)),
            *mode,
        );
    }

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let x0 = 140.0;
        let y0 = 125.0;
        let r0 = 20.0;
        let x1 = 190.0;
        let y1 = 100.0;
        let r1 = 95.0;
        make(
            sb,
            x0,
            y0,
            r0,
            x1,
            y1,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 680.0)),
            *mode,
        );
    }

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let x0 = 140.0;
        let y0 = 125.0;
        let r0 = 20.0;
        let x1 = 190.0;
        let y1 = 100.0;
        let r1 = 96.0;
        // Shift p0 so the outer edges of both circles touch
        let p0 = Point::new(x1, y1)
            + ((Point::new(x0, y0) - Point::new(x1, y1)).normalize() * (r1 - r0));
        make(
            sb,
            p0.x,
            p0.y,
            r0 as f32,
            x1,
            y1,
            r1 as f32,
            Affine::translate((i as f64 * 420.0 + 20.0, 900.0)),
            *mode,
        );
    }
}

fn blend_grid(sb: &mut SceneBuilder, _: &mut SceneParams) {
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

// Support functions

fn render_cardioid(sb: &mut SceneBuilder) {
    let n = 601;
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = BezPath::new();
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
        Color::rgb8(0, 0, 255),
        None,
        &path,
    );
}

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
        let path = [
            PathEl::MoveTo((X0, Y0).into()),
            PathEl::LineTo((X1, Y0).into()),
            PathEl::LineTo((X1, Y0 + t * (Y1 - Y0)).into()),
            PathEl::LineTo((X1 + t * (X0 - X1), Y1).into()),
            PathEl::LineTo((X0, Y1).into()),
            PathEl::ClosePath,
        ];
        sb.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &path);
    }
    let rect = Rect::new(X0, Y0, X1, Y1);
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 255, 0)),
        None,
        &rect,
    );
    for _ in 0..N {
        sb.pop_layer();
    }
}

fn render_alpha_test(sb: &mut SceneBuilder) {
    // Alpha compositing tests.
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(255, 0, 0),
        None,
        &make_diamond(1024.0, 100.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(0, 255, 0, 0x80),
        None,
        &make_diamond(1024.0, 125.0),
    );
    sb.push_layer(
        Mix::Clip,
        1.0,
        Affine::IDENTITY,
        &make_diamond(1024.0, 150.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(0, 0, 255, 0x80),
        None,
        &make_diamond(1024.0, 175.0),
    );
    sb.pop_layer();
}

fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), (200., 200.));
    let linear =
        Gradient::new_linear((0.0, 0.0), (200.0, 0.0)).with_stops([Color::BLACK, Color::WHITE]);
    sb.fill(Fill::NonZero, transform, &linear, None, &rect);
    const GRADIENTS: &[(f64, f64, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = *c;
        color2.a = 0;
        let radial = Gradient::new_radial((*x, *y), 100.0).with_stops([*c, color2]);
        sb.fill(Fill::NonZero, transform, &radial, None, &rect);
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    sb.push_layer(Mix::Normal, 1.0, transform, &rect);
    for (i, c) in COLORS.iter().enumerate() {
        let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([Color::WHITE, *c]);
        sb.push_layer(blend, 1.0, transform, &rect);
        // squash the ellipse
        let a = transform
            * Affine::translate((100., 100.))
            * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
            * Affine::scale_non_uniform(1.0, 0.357)
            * Affine::translate((-100., -100.));
        sb.fill(
            Fill::NonZero,
            a,
            &linear,
            None,
            &Ellipse::new((100., 100.), (90., 90.), 0.),
        );
        sb.pop_layer();
    }
    sb.pop_layer();
}

fn blend_square(blend: BlendMode) -> SceneFragment {
    let mut fragment = SceneFragment::default();
    let mut sb = SceneBuilder::for_fragment(&mut fragment);
    render_blend_square(&mut sb, blend, Affine::IDENTITY);
    fragment
}

fn conflation_artifacts(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;
    const N: f64 = 50.0;
    const S: f64 = 4.0;

    let scale = Affine::scale(S);
    let x = N + 0.5; // Fractional pixel offset reveals the problem on axis-aligned edges.
    let mut y = N;

    let bg_color = Color::rgb8(255, 194, 19);
    let fg_color = Color::rgb8(12, 165, 255);

    // Two adjacent triangles touching at diagonal edge with opposing winding numbers
    sb.fill(
        Fill::NonZero,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // triangle 1
            MoveTo((0.0, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((0.0, N).into()),
            LineTo((0.0, 0.0).into()),
            // triangle 2
            MoveTo((0.0, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((N, 0.0).into()),
            LineTo((0.0, 0.0).into()),
        ],
    );

    // Adjacent rects, opposite winding
    y += S * N + 10.0;
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        bg_color,
        None,
        &Rect::new(0.0, 0.0, N, N),
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // left rect
            MoveTo((0.0, 0.0).into()),
            LineTo((0.0, N).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N * 0.5, 0.0).into()),
            // right rect
            MoveTo((N * 0.5, 0.0).into()),
            LineTo((N, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((N * 0.5, N).into()),
        ],
    );

    // Adjacent rects, same winding
    y += S * N + 10.0;
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        bg_color,
        None,
        &Rect::new(0.0, 0.0, N, N),
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // left rect
            MoveTo((0.0, 0.0).into()),
            LineTo((0.0, N).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N * 0.5, 0.0).into()),
            // right rect
            MoveTo((N * 0.5, 0.0).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N, N).into()),
            LineTo((N, 0.0).into()),
        ],
    );
}

fn labyrinth(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;

    let rows: &[[u8; 12]] = &[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    let cols: &[[u8; 10]] = &[
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    ];
    let mut path = BezPath::new();
    for (y, row) in rows.iter().enumerate() {
        for (x, flag) in row.iter().enumerate() {
            let x = x as f64;
            let y = y as f64;
            if *flag == 1 {
                path.push(MoveTo((x - 0.1, y + 0.1).into()));
                path.push(LineTo((x + 1.1, y + 0.1).into()));
                path.push(LineTo((x + 1.1, y - 0.1).into()));
                path.push(LineTo((x - 0.1, y - 0.1).into()));

                // The above is equivalent to the following stroke with width 0.2 and square
                // caps.
                //path.push(MoveTo((x, y).into()));
                //path.push(LineTo((x + 1.0, y).into()));
            }
        }
    }
    for (x, col) in cols.iter().enumerate() {
        for (y, flag) in col.iter().enumerate() {
            let x = x as f64;
            let y = y as f64;
            if *flag == 1 {
                path.push(MoveTo((x - 0.1, y - 0.1).into()));
                path.push(LineTo((x - 0.1, y + 1.1).into()));
                path.push(LineTo((x + 0.1, y + 1.1).into()));
                path.push(LineTo((x + 0.1, y - 0.1).into()));
                // The above is equivalent to the following stroke with width 0.2 and square
                // caps.
                //path.push(MoveTo((x, y).into()));
                //path.push(LineTo((x, y + 1.0).into()));
            }
        }
    }

    // Note the artifacts are clearly visible at a fractional pixel offset/translation. They
    // disappear if the translation amount below is a whole number.
    sb.fill(
        Fill::NonZero,
        Affine::translate((20.5, 20.5)) * Affine::scale(80.0),
        Color::rgba8(0x70, 0x80, 0x80, 0xff),
        None,
        &path,
    )
}

fn base_color_test(sb: &mut SceneBuilder, params: &mut SceneParams) {
    // Cycle through the hue value every 5 seconds (t % 5) * 360/5
    let color = Color::hlc((params.time % 5.0) * 72.0, 80.0, 80.0);
    params.base_color = Some(color);

    // Blend a white square over it.
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(255, 255, 255, 128),
        None,
        &Rect::new(50.0, 50.0, 500.0, 500.0),
    );
}

fn clip_test(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let clip = {
        const X0: f64 = 50.0;
        const Y0: f64 = 0.0;
        const X1: f64 = 200.0;
        const Y1: f64 = 500.0;
        [
            PathEl::MoveTo((X0, Y0).into()),
            PathEl::LineTo((X1, Y0).into()),
            PathEl::LineTo((X1, Y0 + (Y1 - Y0)).into()),
            PathEl::LineTo((X1 + (X0 - X1), Y1).into()),
            PathEl::LineTo((X0, Y1).into()),
            PathEl::ClosePath,
        ]
    };
    sb.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &clip);
    {
        let text_size = 60.0 + 40.0 * (params.time as f32).sin();
        let s = "Some clipped text!";
        params.text.add(
            sb,
            None,
            text_size,
            None,
            Affine::translate((110.0, 100.0)),
            s,
        );
    }
    sb.pop_layer();

    let large_background_rect = kurbo::Rect::new(-1000.0, -1000.0, 2000.0, 2000.0);
    let inside_clip_rect = kurbo::Rect::new(11.0, 13.399999999999999, 59.0, 56.6);
    let outside_clip_rect = kurbo::Rect::new(
        12.599999999999998,
        12.599999999999998,
        57.400000000000006,
        57.400000000000006,
    );
    let clip_rect = kurbo::Rect::new(0.0, 0.0, 74.4, 339.20000000000005);
    let scale = 2.0;

    sb.push_layer(
        BlendMode {
            mix: peniko::Mix::Normal,
            compose: peniko::Compose::SrcOver,
        },
        1.0,
        Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
        &clip_rect,
    );

    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
        peniko::Color::rgb8(0, 0, 255),
        None,
        &large_background_rect,
    );
    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([
            scale,
            0.0,
            0.0,
            scale,
            29.027636718750003,
            182.9755506427786,
        ]),
        peniko::Color::rgb8(0, 255, 0),
        None,
        &inside_clip_rect,
    );
    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([
            scale,
            0.0,
            0.0,
            scale,
            29.027636718750003,
            scale * 559.3583631427786,
        ]),
        peniko::Color::rgb8(255, 0, 0),
        None,
        &outside_clip_rect,
    );

    sb.pop_layer();
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

fn splash_screen(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let strings = [
        "Vello test",
        "  Arrow keys: switch scenes",
        "  Space: reset transform",
        "  S: toggle stats",
        "  V: toggle vsync",
        "  Q, E: rotate",
    ];
    // Tweak to make it fit with tiger
    let a = Affine::scale(0.12) * Affine::translate((-90.0, -50.0));
    for (i, s) in strings.iter().enumerate() {
        let text_size = if i == 0 { 60.0 } else { 40.0 };
        params.text.add(
            sb,
            None,
            text_size,
            None,
            a * Affine::translate((100.0, 100.0 + 60.0 * i as f64)),
            s,
        );
    }
}

fn splash_with_tiger() -> impl FnMut(&mut SceneBuilder, &mut SceneParams) {
    let contents = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../assets/Ghostscript_Tiger.svg"
    ));
    let mut tiger = crate::svg::svg_function_of("Ghostscript Tiger".to_string(), move || contents);
    move |sb, params| {
        tiger(sb, params);
        splash_screen(sb, params);
    }
}
