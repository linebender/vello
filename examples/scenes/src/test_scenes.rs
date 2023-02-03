use crate::{ExampleScene, SceneConfig, SceneParams, SceneSet};
use vello::kurbo::{Affine, BezPath, Ellipse, PathEl, Point, Rect};
use vello::peniko::*;
use vello::*;

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
    // For WASM below, must be mutable
    #[allow(unused_mut)]
    let mut scenes = vec![
        scene!(funky_paths),
        scene!(cardioid_and_friends),
        scene!(animated_text: animated),
        scene!(brush_transform: animated),
        scene!(blend_grid),
    ];
    #[cfg(target_arch = "wasm32")]
    scenes.push(ExampleScene {
        config: SceneConfig {
            animated: false,
            name: "included_tiger".to_owned(),
        },
        function: Box::new(included_tiger()),
    });

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
    params.text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate((110.0, 700.0)),
        s,
    );
    let th = params.time as f64;
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
    let alpha = (params.time as f64).sin() as f32 * 0.5 + 0.5;
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

#[cfg(target_arch = "wasm32")]
fn included_tiger() -> impl FnMut(&mut SceneBuilder, &mut SceneParams) {
    use vello::kurbo::Vec2;
    use vello_svg::usvg;
    let mut cached_scene = None;
    move |builder, params| {
        let (scene_frag, resolution) = cached_scene.get_or_insert_with(|| {
            let contents = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "../../assets/Ghostscript_Tiger.svg"
            ));
            let svg = usvg::Tree::from_str(&contents, &usvg::Options::default())
                .expect("failed to parse svg file");
            let mut new_scene = SceneFragment::new();
            let mut builder = SceneBuilder::for_fragment(&mut new_scene);
            vello_svg::render_tree(&mut builder, &svg);
            let resolution = Vec2::new(svg.size.width(), svg.size.height());
            (new_scene, resolution)
        });
        builder.append(&scene_frag, None);
        params.resolution = Some(*resolution);
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
        let mut color2 = c.clone();
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
    sb.finish();
    fragment
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
