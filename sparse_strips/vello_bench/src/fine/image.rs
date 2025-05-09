use crate::fine::{default_blend, fill_single};
use criterion::{Bencher, Criterion};
use std::sync::Arc;
use vello_common::coarse::WideTile;
use vello_common::encode::EncodeExt;
use vello_common::kurbo::Affine;
use vello_common::paint::Image;
use vello_common::peniko;
use vello_common::peniko::{ColorStops, GradientKind, ImageQuality};
use vello_common::pixmap::Pixmap;
use vello_cpu::fine::{Fine, FineType};

pub fn image(c: &mut Criterion) {
    transform::none(c);
    transform::scale(c);
    transform::rotate(c);
}

mod transform {
    use crate::fine::image::{get_image, image_base};
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_common::tile::Tile;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    fn none<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let im = get_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::IDENTITY)
    }

    #[vello_bench]
    fn scale<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let im = get_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::scale(3.0))
    }

    #[vello_bench]
    fn rotate<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let im = get_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(
            b,
            fine,
            im,
            Affine::rotate_about(
                1.0,
                Point::new(WideTile::WIDTH as f64 / 2.0, Tile::HEIGHT as f64 / 2.0),
            ),
        );
    }
}

fn get_image(extend: peniko::Extend, quality: ImageQuality) -> Image {
    let data = include_bytes!("../../../../vello_tests/snapshots/big_colr.png");

    let pixmap = Pixmap::from_png(&data[..]).unwrap();
    Image {
        pixmap: Arc::new(pixmap),
        x_extend: extend,
        y_extend: extend,
        quality,
    }
}

fn image_base<F: FineType>(
    b: &mut Bencher<'_>,
    fine: &mut Fine<F>,
    image: Image,
    transform: Affine,
) {
    let mut paints = vec![];

    let paint = image.encode_into(&mut paints, transform);

    fill_single(
        &paint,
        &paints,
        WideTile::WIDTH as usize,
        b,
        default_blend(),
        fine,
    );
}
