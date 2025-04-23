// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A cli

use std::{
    fs::{self, File},
    path::{self, Path},
};

use anyhow::bail;
use color::{AlphaColor, LinearSrgb, PremulColor, Rgba8, Srgb};
use png::Transformations;
use vello_filters_cpu::{Image, NaivePremulPixel};

fn main() -> anyhow::Result<()> {
    let in_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/linebender_logo.png");
    let file_data = fs::read(in_path)?;
    let mut decoder = png::Decoder::new(&*file_data);
    decoder
        // We treat all images as 8 bit per channel, for simplicity.
        .set_transformations(Transformations::normalize_to_color8() | Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let (png::ColorType::Rgba, png::BitDepth::Eight) = reader.output_color_type() else {
        bail!(
            "Only support some output types, got {:?}",
            reader.output_color_type()
        );
    };

    let mut buf = vec![
        Rgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        };
        reader.output_buffer_size() / 4
    ];
    let data = bytemuck::cast_slice_mut(&mut buf);
    let (width, height) = reader.info().size();
    reader.next_frame(data)?;
    let rgba_image = Image {
        width: width.try_into().unwrap(),
        height: height.try_into().unwrap(),
        pixels: buf,
    };

    let linear_pixel = rgba_image
        .pixels
        .iter()
        .map(|rgba| {
            AlphaColor::<Srgb>::from(*rgba)
                .convert::<LinearSrgb>()
                .premultiply()
        })
        .collect();
    let linear_srgb_image = Image {
        width: rgba_image.width,
        height: rgba_image.height,
        pixels: linear_pixel,
    };
    drop(rgba_image);
    let mut naive_input_image: Image<NaivePremulPixel> = Image {
        width: linear_srgb_image.width,
        height: linear_srgb_image.height,
        pixels: bytemuck::cast_vec(linear_srgb_image.pixels),
    };
    let mut scratch_image = Image::empty_scratch();

    vello_filters_cpu::blur::conventional_box::approx_gauss_box_blur(
        &mut naive_input_image,
        None,
        5.0,
        10.0,
        &mut scratch_image,
    );
    let rgba_out_image = Image {
        width: naive_input_image.width,
        height: naive_input_image.height,
        pixels: bytemuck::cast_vec::<_, PremulColor<LinearSrgb>>(naive_input_image.pixels)
            .iter()
            .map(|color| color.convert::<Srgb>().to_rgba8())
            .collect(),
    };

    let out_path = path::absolute("debug.png").unwrap();
    let mut file = File::create(&out_path)?;
    let mut png_encoder = png::Encoder::new(
        &mut file,
        rgba_out_image.width.into(),
        rgba_out_image.height.into(),
    );
    png_encoder.set_color(png::ColorType::Rgba);
    png_encoder.set_depth(png::BitDepth::Eight);
    let mut writer = png_encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&rgba_out_image.pixels))?;
    writer.finish()?;
    println!("Wrote result ({width}x{height}) to {out_path:?}");
    Ok(())
}
