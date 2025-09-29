// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{default_blend, fill_single};
use criterion::{Bencher, Criterion};
use std::sync::Arc;
use vello_common::coarse::WideTile;
use vello_common::encode::EncodeExt;
use vello_common::fearless_simd::Simd;
use vello_common::kurbo::Affine;
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko;
use vello_common::peniko::ImageQuality;
use vello_common::peniko::ImageSampler;
use vello_common::pixmap::Pixmap;
use vello_cpu::fine::{Fine, FineKernel};

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
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Affine;
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_cpu::fine::{Fine, FineKernel};
    use vello_dev_macros::vello_bench;

    fn extend_base<S: Simd, T: FineKernel<S>>(
        b: &mut Bencher<'_>,
        fine: &mut Fine<S, T>,
        extend: peniko::Extend,
    ) {
        let im = get_small_image(extend, ImageQuality::Low);
        image_base(
            b,
            fine,
            im,
            Affine::translate((WideTile::WIDTH as f64 / 2.0, 0.0)),
        );
    }

    #[vello_bench]
    fn pad<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        extend_base(b, fine, peniko::Extend::Pad);
    }

    #[vello_bench]
    fn repeat<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        extend_base(b, fine, peniko::Extend::Repeat);
    }

    #[vello_bench]
    fn reflect<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        extend_base(b, fine, peniko::Extend::Reflect);
    }
}

mod quality {
    use crate::fine::image::{get_colr_image, image_base};
    use criterion::Bencher;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::Affine;
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_cpu::fine::{Fine, FineKernel};
    use vello_dev_macros::vello_bench;

    fn quality_base<S: Simd, T: FineKernel<S>>(
        b: &mut Bencher<'_>,
        fine: &mut Fine<S, T>,
        quality: ImageQuality,
    ) {
        let im = get_colr_image(peniko::Extend::Pad, quality);
        image_base(b, fine, im, Affine::scale(3.0));
    }

    #[vello_bench]
    fn low<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        quality_base(b, fine, ImageQuality::Low);
    }

    #[vello_bench]
    fn medium<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        quality_base(b, fine, ImageQuality::Medium);
    }

    #[vello_bench]
    fn high<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        quality_base(b, fine, ImageQuality::High);
    }
}

mod transform {
    use crate::fine::image::{get_colr_image, image_base};
    use criterion::Bencher;
    use vello_common::coarse::WideTile;
    use vello_common::fearless_simd::Simd;
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko;
    use vello_common::peniko::ImageQuality;
    use vello_common::tile::Tile;
    use vello_cpu::fine::{Fine, FineKernel};
    use vello_dev_macros::vello_bench;

    #[vello_bench]
    fn none<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        let im = get_colr_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::IDENTITY);
    }

    #[vello_bench]
    fn scale<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
        let im = get_colr_image(peniko::Extend::Pad, ImageQuality::Low);
        image_base(b, fine, im, Affine::scale(3.0));
    }

    #[vello_bench]
    fn rotate<S: Simd, T: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, T>) {
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
        image: ImageSource::Pixmap(Arc::new(pixmap)),
        sampler: ImageSampler {
            x_extend: extend,
            y_extend: extend,
            quality,
            alpha: 1.0,
        },
    }
}

fn get_small_image(extend: peniko::Extend, quality: ImageQuality) -> Image {
    let data = include_bytes!("../../../vello_sparse_tests/tests/assets/rgb_image_2x2.png");

    let pixmap = Pixmap::from_png(&data[..]).unwrap();
    Image {
        image: ImageSource::Pixmap(Arc::new(pixmap)),
        sampler: ImageSampler {
            x_extend: extend,
            y_extend: extend,
            quality,
            alpha: 1.0,
        },
    }
}

fn image_base<S: Simd, T: FineKernel<S>>(
    b: &mut Bencher<'_>,
    fine: &mut Fine<S, T>,
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
