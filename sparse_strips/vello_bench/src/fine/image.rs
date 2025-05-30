// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{default_blend, fill_single};
use criterion::{Bencher, Criterion};
use std::sync::Arc;
use vello_common::coarse::WideTile;
use vello_common::encode::EncodeExt;
use vello_common::kurbo::Affine;
use vello_common::paint::Image;
use vello_common::peniko;
use vello_common::peniko::ImageQuality;
use vello_common::pixmap::Pixmap;
use vello_cpu::fine::{Fine, FineType};

// TODO: Add benchmarks for images with transparency
pub fn image(c: &mut Criterion) {
    transform::none(c);
    transform::scale(c);
    transform::rotate(c);

    quality::low(c);
    quality::medium(c);
    quality::high(c);

    extend::pad(c);
    extend::repeat(c);
    extend::reflect(c);
}

mod extend {
    use crate::fine::image::{get_small_image, image_base};
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::kurbo::Affine;
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    fn extend_base<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>, extend: peniko::Extend) {
        let im = get_small_image(extend, ImageQuality::Low);
        image_base(
            b,
            fine,
            im,
            Affine::translate((WideTile::WIDTH as f64 / 2.0, 0.0)),
        );
    }

    #[vello_bench]
    fn pad<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend_base(b, fine, peniko::Extend::Pad);
    }

    #[vello_bench]
    fn repeat<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend_base(b, fine, peniko::Extend::Repeat);
    }

    #[vello_bench]
    fn reflect<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        extend_base(b, fine, peniko::Extend::Reflect);
    }
}

mod quality {
    use crate::fine::image::{get_colr_image, image_base};
    use criterion::Bencher;
    use vello_common::kurbo::Affine;
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_cpu::fine::{Fine, FineType};
    use vello_dev_macros::vello_bench;

    fn quality_base<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>, quality: ImageQuality) {
        let im = get_colr_image(peniko::Extend::Pad, quality);
        image_base(b, fine, im, Affine::scale(3.0));
    }

    #[vello_bench]
    fn low<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        quality_base(b, fine, ImageQuality::Low);
    }

    #[vello_bench]
    fn medium<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        quality_base(b, fine, ImageQuality::Medium);
    }

    #[vello_bench]
    fn high<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        quality_base(b, fine, ImageQuality::High);
    }
}

mod transform {
    use crate::fine::image::{get_colr_image, image_base};
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
        let im = get_colr_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::IDENTITY);
    }

    #[vello_bench]
    fn scale<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let im = get_colr_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::scale(3.0));
    }

    #[vello_bench]
    fn rotate<F: FineType>(b: &mut Bencher<'_>, fine: &mut Fine<F>) {
        let im = get_colr_image(peniko::Extend::Pad, ImageQuality::Low);
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

fn get_colr_image(extend: peniko::Extend, quality: ImageQuality) -> Image {
    let data = include_bytes!("../../../../vello_tests/snapshots/big_colr.png");

    let pixmap = Pixmap::from_png(&data[..]).unwrap();
    Image {
        pixmap: Arc::new(pixmap),
        x_extend: extend,
        y_extend: extend,
        quality,
    }
}

fn get_small_image(extend: peniko::Extend, quality: ImageQuality) -> Image {
    let data = include_bytes!("../../../vello_sparse_tests/tests/assets/rgb_image_2x2.png");

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
