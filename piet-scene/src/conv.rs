use peniko::kurbo::{Affine, Point};

pub fn affine_to_f32(affine: &Affine) -> [f32; 6] {
    let c = affine.as_coeffs();
    [
        c[0] as f32,
        c[1] as f32,
        c[2] as f32,
        c[3] as f32,
        c[4] as f32,
        c[5] as f32,
    ]
}

pub fn affine_from_f32(coeffs: &[f32; 6]) -> Affine {
    Affine::new([
        coeffs[0] as f64,
        coeffs[1] as f64,
        coeffs[2] as f64,
        coeffs[3] as f64,
        coeffs[4] as f64,
        coeffs[5] as f64,
    ])
}

pub fn point_to_f32(point: Point) -> [f32; 2] {
    [point.x as f32, point.y as f32]
}
