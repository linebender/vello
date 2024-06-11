// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::fmt;
use std::{
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

use image::{DynamicImage, ImageError};
use nv_flip::FlipPool;
use vello::{
    peniko::{Format, Image},
    Scene,
};

use crate::{env_var_relates_to, render, write_png_to_file, TestParams};
use anyhow::{anyhow, bail, Result};

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots")
}

#[must_use]
pub struct Snapshot<'a> {
    pub pool: Option<FlipPool>,
    pub reference_path: PathBuf,
    pub update_path: PathBuf,
    pub raw_rendered: Image,
    pub params: &'a TestParams,
}

impl Snapshot<'_> {
    pub fn assert_mean_less_than(&mut self, value: f32) -> Result<()> {
        assert!(
            value < 0.1,
            "Mean should be less than 0.1 in almost all cases for a successful test"
        );
        if let Some(pool) = &self.pool {
            let mean = pool.mean();
            if mean > value {
                self.handle_failure(format_args!(
                    "Expected mean to be less than {value}, got {mean}"
                ))?;
            }
        } else {
            // The image is new, so assertion needed?
        }
        self.handle_success()?;
        Ok(())
    }

    fn handle_success(&mut self) -> Result<()> {
        match std::fs::remove_file(&self.update_path) {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            res => res.map_err(Into::into),
        }
    }

    fn handle_failure(&mut self, message: fmt::Arguments) -> Result<()> {
        if env_var_relates_to("VELLO_TEST_UPDATE", &self.params.name, self.params.use_cpu) {
            if !self.params.use_cpu {
                write_png_to_file(self.params, &self.reference_path, &self.raw_rendered)?;
                eprintln!(
                    "Updated result for updated test {} to {:?}",
                    self.params.name, &self.reference_path
                );
            } else {
                eprintln!(
                    "Skipped updating result for test {} as not GPU test",
                    self.params.name
                );
            }
        } else {
            write_png_to_file(self.params, &self.update_path, &self.raw_rendered)?;
            eprintln!(
                "Wrote result for failing test {} to {:?}\n\
                Use `VELLO_TEST_UPDATE=all` to update",
                self.params.name, &self.update_path
            );
        }
        bail!("{}", message);
    }
}

/// Run a snapshot test.
///
/// Try and keep the width and height small, to reduce the size of committed binary data
pub fn snapshot_test_sync(scene: Scene, params: &TestParams) -> Result<Snapshot<'_>> {
    pollster::block_on(snapshot_test(scene, params))
}

pub async fn snapshot_test(scene: Scene, params: &TestParams) -> Result<Snapshot> {
    let raw_rendered = render(scene, params).await?;

    // TODO: A different file for GPU and CPU?
    let reference_path = snapshot_dir().join(&params.name).with_extension("png");
    let update_extension = if params.use_cpu {
        "cpu.new.png"
    } else {
        "gpu.new.png"
    };
    let update_path = snapshot_dir()
        .join(&params.name)
        .with_extension(update_extension);

    let expected_data = match image::open(&reference_path) {
        Ok(contents) => contents.into_rgb8(),
        Err(ImageError::IoError(e)) if e.kind() == io::ErrorKind::NotFound => {
            if env_var_relates_to("VELLO_TEST_CREATE", &params.name, params.use_cpu) {
                if params.use_cpu {
                    write_png_to_file(params, &reference_path, &raw_rendered)?;
                    eprintln!(
                        "Wrote result for new test {} to {:?}",
                        params.name, &reference_path
                    );
                } else {
                    eprintln!(
                        "Skipped writing result for new test {} as not GPU test",
                        params.name
                    );
                }
                return Ok(Snapshot {
                    pool: None,
                    reference_path,
                    update_path,
                    raw_rendered,
                    params,
                });
            } else {
                write_png_to_file(params, &update_path, &raw_rendered)?;
                bail!(
                    "Couldn't find snapshot for test {}. Searched at {:?}\n\
                    Test result written to {:?}\n\
                    Use `VELLO_TEST_CREATE=all` to update",
                    params.name,
                    reference_path,
                    update_path
                );
            }
        }
        Err(e) => return Err(e.into()),
    };

    if expected_data.width() != raw_rendered.width || expected_data.height() != raw_rendered.height
    {
        let mut snapshot = Snapshot {
            pool: None,
            reference_path,
            update_path,
            raw_rendered,
            params,
        };
        snapshot.handle_failure(format_args!(
            "Got wrong size. Expected ({expected_width}x{expected_height}), found ({actual_width}x{actual_height})",
            expected_width = expected_data.width(),
            expected_height = expected_data.height(),
            actual_width = params.width,
            actual_height = params.height
        ))?;
        unreachable!();
    }
    // Compare the images using nv-flip
    assert_eq!(raw_rendered.format, Format::Rgba8);
    let rendered_data: DynamicImage = image::RgbaImage::from_raw(
        raw_rendered.width,
        raw_rendered.height,
        raw_rendered.data.as_ref().to_vec(),
    )
    .ok_or(anyhow!("Couldn't create image"))?
    .into();
    let rendered_data = rendered_data.to_rgb8();
    let expected = nv_flip::FlipImageRgb8::with_data(
        expected_data.width(),
        expected_data.height(),
        &expected_data,
    );
    let rendered = nv_flip::FlipImageRgb8::with_data(
        rendered_data.width(),
        rendered_data.height(),
        &rendered_data,
    );

    let error_map = nv_flip::flip(expected, rendered, nv_flip::DEFAULT_PIXELS_PER_DEGREE);

    let pool = nv_flip::FlipPool::from_image(&error_map);

    Ok(Snapshot {
        pool: Some(pool),
        reference_path,
        update_path,
        raw_rendered,
        params,
    })
}
