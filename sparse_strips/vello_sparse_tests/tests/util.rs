// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions shared across different tests.

use crate::renderer::Renderer;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, Rgba, RgbaImage, load_from_memory};
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use smallvec::smallvec;
use std::cmp::max;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::glyph::Glyph;
use vello_common::kurbo::{BezPath, Join, Point, Rect, Shape, Stroke, Vec2};
use vello_common::peniko::{Blob, ColorStop, ColorStops, Font};
use vello_common::pixmap::Pixmap;

static REFS_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_sparse_tests/snapshots")
});
static DIFFS_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_sparse_tests/diffs"));

pub(crate) fn get_ctx<T: Renderer>(width: u16, height: u16, transparent: bool) -> T {
    let mut ctx = T::new(width, height);

    if !transparent {
        let path = Rect::new(0.0, 0.0, width as f64, height as f64).to_path(0.1);

        ctx.set_paint(WHITE);
        ctx.fill_path(&path);
    }

    ctx
}

pub(crate) fn render_pixmap(ctx: &impl Renderer) -> Pixmap {
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_to_pixmap(&mut pixmap);
    pixmap
}

pub(crate) fn miter_stroke_2() -> Stroke {
    Stroke {
        width: 2.0,
        join: Join::Miter,
        ..Default::default()
    }
}

pub(crate) fn crossed_line_star() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((50.0, 10.0));
    path.line_to((75.0, 90.0));
    path.line_to((10.0, 40.0));
    path.line_to((90.0, 40.0));
    path.line_to((25.0, 90.0));
    path.line_to((50.0, 10.0));

    path
}

pub(crate) fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    let start_angle = -std::f64::consts::FRAC_PI_2;
    path.move_to(center + outer * Vec2::from_angle(start_angle));
    for i in 1..n * 2 {
        let th = start_angle + i as f64 * std::f64::consts::PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}

/// ***DO NOT USE THIS OUTSIDE OF THESE TESTS***
///
/// This function is used for _TESTING PURPOSES ONLY_. If you need to layout and shape
/// text for your application, use a proper text shaping library like `Parley`.
///
/// We use this function as a convenience for testing; to get some glyphs shaped and laid
/// out in a small amount of code without having to go through the trouble of setting up a
/// full text layout pipeline, which you absolutely should do in application code.
pub(crate) fn layout_glyphs(text: &str, font_size: f32) -> (Font, Vec<Glyph>) {
    const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");
    let font = Font::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);

    let font_ref = {
        let file_ref = FileRef::new(font.data.as_ref()).unwrap();
        match file_ref {
            FileRef::Font(f) => f,
            FileRef::Collection(collection) => collection.get(font.index).unwrap(),
        }
    };
    let font_size = skrifa::instance::Size::new(font_size);
    let axes = font_ref.axes();
    let variations: Vec<(&str, f32)> = vec![];
    let var_loc = axes.location(variations.as_slice());
    let charmap = font_ref.charmap();
    let metrics = font_ref.metrics(font_size, &var_loc);
    let line_height = metrics.ascent - metrics.descent + metrics.leading;
    let glyph_metrics = font_ref.glyph_metrics(font_size, &var_loc);

    let mut pen_x = 0_f32;
    let mut pen_y = 0_f32;

    let glyphs = text
        .chars()
        .filter_map(|ch| {
            if ch == '\n' {
                pen_y += line_height;
                pen_x = 0.0;
                return None;
            }
            let gid = charmap.map(ch).unwrap_or_default();
            let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
            let x = pen_x;
            pen_x += advance;
            Some(Glyph {
                id: gid.to_u32(),
                x,
                y: pen_y,
            })
        })
        .collect::<Vec<_>>();

    (font, glyphs)
}

pub(crate) fn stops_green_blue() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
    ])
}

pub(crate) fn stops_green_blue_with_alpha() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(GREEN.with_alpha(0.25)),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(BLUE.with_alpha(0.75)),
        },
    ])
}

pub(crate) fn stops_blue_green_red_yellow() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(RED),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW),
        },
    ])
}

pub(crate) fn pixmap_to_png(mut pixmap: Pixmap, width: u32, height: u32) -> Vec<u8> {
    pixmap.unpremultiply();

    let mut png_data = Vec::new();
    let cursor = Cursor::new(&mut png_data);
    let encoder = PngEncoder::new(cursor);
    encoder
        .write_image(pixmap.data(), width, height, ExtendedColorType::Rgba8)
        .expect("Failed to encode image");
    png_data
}

pub(crate) fn check_ref(
    ctx: &impl Renderer,
    // The name of the test.
    test_name: &str,
    // The name of the specific instance of the test that is being run
    // (e.g. test_gpu, test_cpu_u8, etc.)
    specific_name: &str,
    // Tolerance for pixel differences.
    threshold: u8,
    // Whether the test instance is the "gold standard" and should be used
    // for creating reference images.
    is_reference: bool,
) {
    let pixmap = render_pixmap(ctx);

    let encoded_image = pixmap_to_png(pixmap, ctx.width() as u32, ctx.height() as u32);
    let ref_path = REFS_PATH.join(format!("{}.png", test_name));

    let write_ref_image = || {
        let optimized =
            oxipng::optimize_from_memory(&encoded_image, &oxipng::Options::max_compression())
                .unwrap();
        std::fs::write(&ref_path, optimized).unwrap();
    };

    if !ref_path.exists() {
        if is_reference {
            write_ref_image();
            panic!("new reference image was created");
        } else {
            panic!("no reference image exists");
        }
    }

    let ref_image = load_from_memory(&std::fs::read(&ref_path).unwrap())
        .unwrap()
        .into_rgba8();
    let actual = load_from_memory(&encoded_image).unwrap().into_rgba8();

    let diff_image = get_diff(&ref_image, &actual, threshold);

    if let Some(diff_image) = diff_image {
        if std::env::var("REPLACE").is_ok() && is_reference {
            write_ref_image();
            panic!("test was replaced");
        }

        if !DIFFS_PATH.exists() {
            let _ = std::fs::create_dir_all(DIFFS_PATH.as_path());
        }

        let diff_path = DIFFS_PATH.join(format!("{}.png", specific_name));
        diff_image
            .save_with_format(&diff_path, image::ImageFormat::Png)
            .unwrap();

        panic!("test didnt match reference image");
    }
}

fn get_diff(
    expected_image: &RgbaImage,
    actual_image: &RgbaImage,
    threshold: u8,
) -> Option<RgbaImage> {
    let width = max(expected_image.width(), actual_image.width());
    let height = max(expected_image.height(), actual_image.height());

    let mut diff_image = RgbaImage::new(width * 3, height);

    let mut pixel_diff = 0;

    for x in 0..width {
        for y in 0..height {
            let actual_pixel = actual_image.get_pixel_checked(x, y);
            let expected_pixel = expected_image.get_pixel_checked(x, y);

            match (actual_pixel, expected_pixel) {
                (Some(actual), Some(expected)) => {
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    if is_pix_diff(expected, actual, threshold) {
                        pixel_diff += 1;
                        diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    } else {
                        diff_image.put_pixel(x + width, y, Rgba([0, 0, 0, 255]));
                    }
                }
                (Some(actual), None) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
                (None, Some(expected)) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
                _ => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
            }
        }
    }

    if pixel_diff > 0 {
        Some(diff_image)
    } else {
        None
    }
}

fn is_pix_diff(pixel1: &Rgba<u8>, pixel2: &Rgba<u8>, threshold: u8) -> bool {
    if pixel1.0[3] == 0 && pixel2.0[3] == 0 {
        return false;
    }

    let mut different = false;

    for i in 0..3 {
        different |= pixel1.0[i].abs_diff(pixel2.0[i]) > threshold;
    }

    different
}
