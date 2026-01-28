// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions shared across different tests.

use crate::renderer::Renderer;
use image::{Rgba, RgbaImage, load_from_memory};
use serde::Serializer;
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use smallvec::smallvec;
use std::cmp::max;
use std::sync::Arc;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::glyph::Glyph;
use vello_common::kurbo::{BezPath, Join, Point, Rect, Shape, Stroke, Vec2};
use vello_common::peniko::{Blob, ColorStop, ColorStops, FontData};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Level, RenderMode};

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

/// Aggregate diff report with statistics and individual pixel differences.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, serde::Serialize)]
pub(crate) struct DiffReport {
    /// Total number of pixels that differ.
    pub pixel_count: usize,
    /// Maximum absolute difference per channel [R, G, B, A].
    pub max_difference: [i16; 4],
    /// Individual pixel differences.
    pub pixels: Vec<PixelDiff>,
}

/// Represents a single pixel difference between reference and actual images.
#[derive(Debug, serde::Serialize)]
pub(crate) struct PixelDiff {
    /// The x coordinate of the differing pixel.
    pub x: u32,
    /// The y coordinate of the differing pixel.
    pub y: u32,
    /// The RGBA values from the target image (i.e. the saved reference).
    // Note that this field name is chosen to be the same length as `actual`
    // That makes it easier to compare the results in the printed JSON.
    #[serde(serialize_with = "hex_string")]
    pub target: [u8; 4],
    /// The RGBA values from the actual image.
    #[serde(serialize_with = "hex_string")]
    pub actual: [u8; 4],
    /// Per-channel difference (actual - target) as signed values.
    pub difference: [i16; 4],
}

/// Serialize a [`[u8; 4]`](primitive@core::array) pixel as a hex string through serde.
///
/// E.g. `[0, 255, 0, 255]` becomes #00ff00. Notice that the alpha is not included if fully opaque.
fn hex_string<S>([r, g, b, a]: &[u8; 4], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if *a != 255 {
        serializer.collect_str(&format_args!("#{r:02x}{g:02x}{b:02x}{a:02x}"))
    } else {
        serializer.collect_str(&format_args!("#{r:02x}{g:02x}{b:02x}"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
static REFS_PATH: std::sync::LazyLock<PathBuf> = std::sync::LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_sparse_tests/snapshots")
});
#[cfg(not(target_arch = "wasm32"))]
static DIFFS_PATH: std::sync::LazyLock<PathBuf> = std::sync::LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_sparse_tests/diffs")
});

/// Helper for loading png images contained within "tests/assets/**".
#[macro_export]
macro_rules! load_image {
    ($name:expr) => {{
        #[cfg(target_arch = "wasm32")]
        {
            let bytes = include_bytes!(concat!("../tests/assets/", $name, ".png"));
            std::sync::Arc::new(vello_common::pixmap::Pixmap::from_png(&bytes[..]).unwrap())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/assets/{}.png", $name));
            std::sync::Arc::new(
                vello_common::pixmap::Pixmap::from_png(std::fs::File::open(path).unwrap()).unwrap(),
            )
        }
    }};
}

pub(crate) fn get_ctx<T: Renderer>(
    width: u16,
    height: u16,
    transparent: bool,
    num_threads: u16,
    level: &str,
    render_mode: RenderMode,
) -> T {
    let level = match level {
        #[cfg(target_arch = "aarch64")]
        "neon" => Level::Neon(
            Level::try_detect()
                .unwrap_or(Level::fallback())
                .as_neon()
                .expect("neon should be available"),
        ),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        "wasm_simd128" => Level::WasmSimd128(
            Level::try_detect()
                .unwrap_or(Level::fallback())
                .as_wasm_simd128()
                .expect("wasm simd128 should be available"),
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse42" => {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                Level::Sse4_2(unsafe { vello_common::fearless_simd::Sse4_2::new_unchecked() })
            } else {
                panic!("sse4.2 feature not detected");
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx2" => {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Level::Avx2(unsafe { vello_common::fearless_simd::Avx2::new_unchecked() })
            } else {
                panic!("avx2 or fma feature not detected");
            }
        }
        "fallback" => Level::fallback(),
        _ => panic!("unknown level: {level}"),
    };

    let mut ctx = T::new(width, height, num_threads, level, render_mode);

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

pub(crate) fn layout_glyphs_roboto(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

pub(crate) fn layout_glyphs_noto_cbtf(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const NOTO_FONT: &[u8] =
        include_bytes!("../../../examples/assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf");
    let font = FontData::new(Blob::new(Arc::new(NOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

pub(crate) fn layout_glyphs_noto_colr(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const NOTO_FONT: &[u8] =
        include_bytes!("../../../examples/assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");
    let font = FontData::new(Blob::new(Arc::new(NOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

#[cfg(target_os = "macos")]
pub(crate) fn layout_glyphs_apple_color_emoji(
    text: &str,
    font_size: f32,
) -> (FontData, Vec<Glyph>) {
    let apple_font: Vec<u8> = std::fs::read("/System/Library/Fonts/Apple Color Emoji.ttc").unwrap();
    let font = FontData::new(Blob::new(Arc::new(apple_font)), 0);

    layout_glyphs(text, font_size, font)
}

/// ***DO NOT USE THIS OUTSIDE OF THESE TESTS***
///
/// This function is used for _TESTING PURPOSES ONLY_. If you need to layout and shape
/// text for your application, use a proper text shaping library like `Parley`.
///
/// We use this function as a convenience for testing; to get some glyphs shaped and laid
/// out in a small amount of code without having to go through the trouble of setting up a
/// full text layout pipeline, which you absolutely should do in application code.
fn layout_glyphs(text: &str, font_size: f32, font: FontData) -> (FontData, Vec<Glyph>) {
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn check_ref(
    ctx: &impl Renderer,
    // The name of the test.
    test_name: &str,
    // The name of the specific instance of the test that is being run
    // (e.g. test_gpu, test_cpu_u8, etc.)
    specific_name: &str,
    // Tolerance for pixel differences.
    threshold: u8,
    diff_pixels: u32,
    // Whether the test instance is the "gold standard" and should be used
    // for creating reference images.
    is_reference: bool,
    _: &[u8],
) {
    let pixmap = render_pixmap(ctx);

    let encoded_image = pixmap.into_png().unwrap();
    let ref_path = REFS_PATH.join(format!("{test_name}.png"));

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

    let diff_result = get_diff(&ref_image, &actual, threshold, diff_pixels);

    if let Some((diff_image, diff_data)) = diff_result {
        if should_replace() && is_reference {
            write_ref_image();
            panic!("test was replaced");
        }

        if !DIFFS_PATH.exists() {
            let _ = std::fs::create_dir_all(DIFFS_PATH.as_path());
        }

        let diff_path = DIFFS_PATH.join(format!("{specific_name}.png"));
        diff_image
            .save_with_format(&diff_path, image::ImageFormat::Png)
            .unwrap();

        // Save diff data as JSON
        let json_path = DIFFS_PATH.join(format!("{specific_name}.json"));
        let max_difference: [i16; 4] = diff_data.iter().fold([0; 4], |mut max, p| {
            for (m, d) in max.iter_mut().zip(&p.difference) {
                *m = (*m).max(d.abs());
            }
            max
        });
        let report = DiffReport {
            pixel_count: diff_data.len(),
            max_difference,
            pixels: diff_data,
        };
        let json_data = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&json_path, json_data).unwrap();

        panic!(
            "test didn't match reference image\n  diff image: {}\n  diff report: {}",
            diff_path.display(),
            json_path.display()
        );
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn check_ref(
    ctx: &impl Renderer,
    _test_name: &str,
    // The name of the specific instance of the test that is being run
    // (e.g. test_gpu, test_cpu_u8, etc.)
    specific_name: &str,
    // Tolerance for pixel differences.
    threshold: u8,
    diff_pixels: u32,
    // Must be `false` on `wasm32` as reference image cannot be written to filesystem.
    is_reference: bool,
    ref_data: &[u8],
) {
    assert!(!is_reference, "WASM cannot create new reference images");

    let pixmap = render_pixmap(ctx);
    let encoded_image = pixmap.into_png().unwrap();
    let actual = load_from_memory(&encoded_image).unwrap().into_rgba8();

    let ref_image = load_from_memory(ref_data).unwrap().into_rgba8();

    let diff_image = get_diff(&ref_image, &actual, threshold, diff_pixels);
    if let Some((ref img, _)) = diff_image {
        append_diff_image_to_browser_document(specific_name, img);
        panic!("test didn't match reference image. Scroll to bottom of browser to view diff.");
    }
}

#[cfg(target_arch = "wasm32")]
fn append_diff_image_to_browser_document(specific_name: &str, diff_image: &RgbaImage) {
    use image::ImageEncoder;
    use wasm_bindgen::JsCast;
    use web_sys::js_sys::{Array, Uint8Array};
    use web_sys::{Blob, BlobPropertyBag, HtmlImageElement, Url, window};

    let window = window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();

    let container = document.create_element("div").unwrap();
    container
        .set_attribute(
            "style",
            "border: 2px solid red; \
         margin: 20px; \
         padding: 20px; \
         background: #f0f0f0; \
         display: inline-block;",
        )
        .unwrap();

    let title = document.create_element("h3").unwrap();
    title.set_text_content(Some(&format!("Test Failed: {specific_name}")));
    title
        .set_attribute("style", "color: red; margin-top: 0;")
        .unwrap();
    container.append_child(&title).unwrap();

    let diff_png = {
        let mut png_data = Vec::new();
        let cursor = std::io::Cursor::new(&mut png_data);
        let encoder = image::codecs::png::PngEncoder::new(cursor);
        encoder
            .write_image(
                diff_image.as_raw(),
                diff_image.width(),
                diff_image.height(),
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        png_data
    };

    let uint8_array = Uint8Array::new_with_length(diff_png.len() as u32);
    uint8_array.copy_from(&diff_png);
    let array = Array::new();
    array.push(&uint8_array.buffer());
    let blob_property_bag = BlobPropertyBag::new();
    blob_property_bag.set_type("image/png");
    let blob = Blob::new_with_u8_array_sequence_and_options(&array, &blob_property_bag).unwrap();
    let url = Url::create_object_url_with_blob(&blob).unwrap();

    let img = document
        .create_element("img")
        .unwrap()
        .dyn_into::<HtmlImageElement>()
        .unwrap();
    img.set_src(&url);
    img.set_attribute("style", "border: 1px solid #ccc; max-width: 100%;")
        .unwrap();
    img.set_attribute("title", "Expected | Diff | Actual")
        .unwrap();

    container.append_child(&img).unwrap();
    body.append_child(&container).unwrap();
}

fn get_diff(
    expected_image: &RgbaImage,
    actual_image: &RgbaImage,
    threshold: u8,
    diff_pixels: u32,
) -> Option<(RgbaImage, Vec<PixelDiff>)> {
    let width = max(expected_image.width(), actual_image.width());
    let height = max(expected_image.height(), actual_image.height());

    let mut diff_image = RgbaImage::new(width * 3, height);
    let mut diff_data = Vec::new();

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
                        diff_data.push(PixelDiff {
                            x,
                            y,
                            target: expected.0,
                            actual: actual.0,
                            difference: [
                                i16::from(actual.0[0]) - i16::from(expected.0[0]),
                                i16::from(actual.0[1]) - i16::from(expected.0[1]),
                                i16::from(actual.0[2]) - i16::from(expected.0[2]),
                                i16::from(actual.0[3]) - i16::from(expected.0[3]),
                            ],
                        });
                    } else {
                        diff_image.put_pixel(x + width, y, Rgba([0, 0, 0, 255]));
                    }
                }
                (Some(actual), None) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: [0, 0, 0, 0],
                        actual: actual.0,
                        difference: [
                            i16::from(actual.0[0]),
                            i16::from(actual.0[1]),
                            i16::from(actual.0[2]),
                            i16::from(actual.0[3]),
                        ],
                    });
                }
                (None, Some(expected)) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: expected.0,
                        actual: [0, 0, 0, 0],
                        difference: [
                            -i16::from(expected.0[0]),
                            -i16::from(expected.0[1]),
                            -i16::from(expected.0[2]),
                            -i16::from(expected.0[3]),
                        ],
                    });
                }
                _ => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: [0, 0, 0, 0],
                        actual: [0, 0, 0, 0],
                        difference: [0, 0, 0, 0],
                    });
                }
            }
        }
    }

    if pixel_diff > diff_pixels {
        Some((diff_image, diff_data))
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
        let difference = pixel1.0[i].abs_diff(pixel2.0[i]);
        different |= difference > threshold;
    }

    different
}

#[cfg(not(target_arch = "wasm32"))]
fn should_replace() -> bool {
    match std::env::var("REPLACE") {
        Ok(value) => value == "1",
        Err(_) => false,
    }
}
