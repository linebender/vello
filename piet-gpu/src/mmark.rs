//! A benchmark based on MotionMark 1.2's path benchmark.

use rand::{Rng, seq::SliceRandom};
use piet::{Color, RenderContext, kurbo::{BezPath, CubicBez, Line, ParamCurve, PathSeg, Point, QuadBez}};

use crate::PietGpuRenderContext;

const WIDTH: usize = 1600;
const HEIGHT: usize = 900;

const GRID_WIDTH: i64 = 80;
const GRID_HEIGHT: i64 = 40;

pub struct MMark {
    elements: Vec<Element>,
}

struct Element {
    seg: PathSeg,
    color: Color,
    width: f64,
    is_split: bool,
    grid_point: GridPoint,
}

#[derive(Clone, Copy)]
struct GridPoint((i64, i64));

impl MMark {
    pub fn new(n: usize) -> MMark {
        let mut last = GridPoint((GRID_WIDTH / 2, GRID_HEIGHT / 2));
        let elements = (0..n).map(|_| {
            let element = Element::new_rand(last);
            last = element.grid_point;
            element
        }).collect();
        MMark { elements }
    }

    pub fn draw(&mut self, ctx: &mut PietGpuRenderContext) {
        let mut rng = rand::thread_rng();
        let mut path = BezPath::new();
        let len = self.elements.len();
        for (i, element) in self.elements.iter_mut().enumerate() {
            if path.is_empty() {
                path.move_to(element.seg.start());
            }
            match element.seg {
                PathSeg::Line(l) => path.line_to(l.p1),
                PathSeg::Quad(q) => path.quad_to(q.p1, q.p2),
                PathSeg::Cubic(c) => path.curve_to(c.p1, c.p2, c.p3),
            }
            if element.is_split || i == len {
                // This gets color and width from the last element, original
                // gets it from the first, but this should not matter.
                ctx.stroke(&path, &element.color, element.width);
                path = BezPath::new(); // Should have clear method, to avoid allocations.
            }
            if rng.gen::<f32>() > 0.995 {
                element.is_split ^= true;
            }
        }
    }
}

const COLORS: &[Color] = &[
    Color::rgb8(0x10, 0x10, 0x10),
    Color::rgb8(0x80, 0x80, 0x80),
    Color::rgb8(0xc0, 0xc0, 0xc0),
    Color::rgb8(0x10, 0x10, 0x10),
    Color::rgb8(0x80, 0x80, 0x80),
    Color::rgb8(0xc0, 0xc0, 0xc0),
    Color::rgb8(0xe0, 0x10, 0x40),
];

impl Element {
    fn new_rand(last: GridPoint) -> Element {
        let mut rng = rand::thread_rng();
        let seg_type = rng.gen_range(0, 4);
        let next = GridPoint::random_point(last);
        let (grid_point, seg) = if seg_type < 2 {
            (next, PathSeg::Line(Line::new(last.coordinate(), next.coordinate())))
        } else if seg_type < 3 {
            let p2 = GridPoint::random_point(next);
            (p2, PathSeg::Quad(QuadBez::new(last.coordinate(), next.coordinate(), p2.coordinate())))
        } else {
            let p2 = GridPoint::random_point(next);
            let p3 = GridPoint::random_point(next);
            (p3, PathSeg::Cubic(CubicBez::new(last.coordinate(), next.coordinate(), p2.coordinate(), p3.coordinate())))
        };
        let color = COLORS.choose(&mut rng).unwrap().clone();
        let width = rng.gen::<f64>().powi(5) * 20.0 + 1.0;
        let is_split = rng.gen();
        Element { seg, color, width, is_split, grid_point }
    }
}

const OFFSETS: &[(i64, i64)] = &[(-4, 0), (2, 0), (1, -2), (1, 2)];

impl GridPoint {
    fn random_point(last: GridPoint) -> GridPoint {
        let mut rng = rand::thread_rng();

        let offset = OFFSETS.choose(&mut rng).unwrap();
        let mut x = last.0.0 + offset.0;
        if x < 0 || x > GRID_WIDTH {
            x -= offset.0 * 2;
        }
        let mut y = last.0.1 + offset.1;
        if y < 0 || y > GRID_HEIGHT {
            y -= offset.1 * 2;
        }
        GridPoint((x, y))
    }

    fn coordinate(&self) -> Point {
        let scale_x = WIDTH as f64 / ((GRID_WIDTH + 1) as f64);
        let scale_y = HEIGHT as f64 / ((GRID_HEIGHT + 1) as f64);
        Point::new((self.0.0 as f64 + 0.5) * scale_x, 100.0 + (self.0.1 as f64 + 0.5) * scale_y)
    }
}
