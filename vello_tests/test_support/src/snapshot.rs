// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(not(target_arch = "wasm32"))]
use crate::diff;
use crate::diff::get_diff;
#[cfg(target_arch = "wasm32")]
use image::RgbaImage;
use image::load_from_memory;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Configuration for comparing a rendered PNG to a named reference snapshot.
///
/// The reference name is independent of the run name, allowing several backends or feature
/// combinations to share a reference while producing distinct failure artifacts.
#[derive(Debug)]
pub struct Snapshot<'a> {
    /// Reference filename stem, without the `.png` extension.
    pub reference_name: &'a str,
    /// Test instance name used for diff filenames or the browser's failure heading.
    pub run_name: &'a str,
    /// Maximum permitted per-channel difference, using the existing suite's comparison rules.
    pub threshold: u8,
    /// Maximum number of pixels permitted to exceed the channel threshold.
    pub diff_pixels: u32,
    /// Whether this run may create or replace native reference images.
    ///
    /// Must be false in browser tests. Reference creation or replacement always fails the test
    /// so the new image can be reviewed. Set `REPLACE=1` to replace a mismatching reference.
    pub is_reference: bool,
    /// Existing directory containing the caller's reference snapshots.
    #[cfg(not(target_arch = "wasm32"))]
    pub snapshots_dir: &'a Path,
    /// Directory in which to write native diff PNG images and JSON reports.
    #[cfg(not(target_arch = "wasm32"))]
    pub diffs_dir: &'a Path,
    /// Embedded reference PNG, usually supplied using `include_bytes!`.
    #[cfg(target_arch = "wasm32")]
    pub reference_png: &'a [u8],
}

impl Snapshot<'_> {
    /// Compare an already-rendered PNG with this snapshot's reference.
    ///
    /// # Panics
    ///
    /// Panics if the images exceed the configured tolerances, a reference is created or
    /// replaced, or image decoding, filesystem access or browser reporting fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn check(&self, encoded_image: &[u8]) {
        let ref_path = self
            .snapshots_dir
            .join(format!("{}.png", self.reference_name));

        let write_ref_image = || {
            let optimized =
                oxipng::optimize_from_memory(encoded_image, &oxipng::Options::max_compression())
                    .unwrap();
            std::fs::write(&ref_path, optimized).unwrap();
        };

        if !ref_path.exists() {
            if self.is_reference {
                write_ref_image();
                panic!("new reference image was created");
            } else {
                panic!("no reference image exists");
            }
        }

        let ref_image = load_from_memory(&std::fs::read(&ref_path).unwrap())
            .unwrap()
            .into_rgba8();
        let actual = load_from_memory(encoded_image).unwrap().into_rgba8();

        let diff_result = get_diff(&ref_image, &actual, self.threshold, self.diff_pixels);

        if let Some((diff_image, diff_data)) = diff_result {
            if should_replace() && self.is_reference {
                write_ref_image();
                panic!("test was replaced");
            }

            let (diff_path, json_path) = diff::write_diff(
                &self.diffs_dir.join(self.run_name),
                &diff_image,
                &diff::DiffReport::new(diff_data),
            );

            panic!(
                "test didn't match reference image\n  diff image: {}\n  diff report: {}",
                diff_path.display(),
                json_path.display()
            );
        }
    }

    /// Compare an already-rendered PNG with this snapshot's embedded reference.
    #[cfg(target_arch = "wasm32")]
    pub fn check(&self, encoded_image: &[u8]) {
        assert!(
            !self.is_reference,
            "WASM cannot create new reference images"
        );
        let actual = load_from_memory(encoded_image).unwrap().into_rgba8();
        let ref_image = load_from_memory(self.reference_png).unwrap().into_rgba8();
        if let Some((diff_image, _)) =
            get_diff(&ref_image, &actual, self.threshold, self.diff_pixels)
        {
            append_diff_image_to_browser_document(self.run_name, &diff_image);
            panic!("test didn't match reference image. Scroll to bottom of browser to view diff.");
        }
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

    let uint8_array = Uint8Array::new_with_length(u32::try_from(diff_png.len()).unwrap());
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

#[cfg(not(target_arch = "wasm32"))]
fn should_replace() -> bool {
    match std::env::var("REPLACE") {
        Ok(value) => value == "1",
        Err(_) => false,
    }
}
