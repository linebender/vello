// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::fmt;
use std::{
    env,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use image::{DynamicImage, ImageError};
use nv_flip::FlipPool;
use vello::{
    Scene,
    peniko::{ImageData, ImageFormat},
};

use crate::{TestParams, env_var_relates_to, render_then_debug, write_png_to_file};
use anyhow::{Result, anyhow, bail};

fn current_dir(directory: SnapshotDirectory, use_cpu: bool) -> PathBuf {
    let mut path_buf = Path::new(env!("CARGO_MANIFEST_DIR")).join("current");
    path_buf.push(if use_cpu { "cpu" } else { "gpu" });
    match directory {
        SnapshotDirectory::Smoke => path_buf.push("smoke"),
        SnapshotDirectory::Lfs => { /* Do nothing */ }
    };
    path_buf
}

fn snapshot_dir(directory: SnapshotDirectory) -> PathBuf {
    let mut path_buf = Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots");
    match directory {
        SnapshotDirectory::Smoke => path_buf.push("smoke"),
        SnapshotDirectory::Lfs => { /* Do nothing */ }
    };
    path_buf
}

#[must_use = "A snapshot test doesn't do anything unless an assertion method is called on it"]
/// The result of a scene render, and the difference between that and a stored snapshot.
pub struct Snapshot<'a> {
    pub statistics: Option<FlipPool>,
    pub reference_path: PathBuf,
    pub update_path: PathBuf,
    pub raw_rendered: ImageData,
    pub params: &'a TestParams,
    pub directory: SnapshotDirectory,
}

impl Snapshot<'_> {
    /// Assert that that mean value stored in `statistics` is less than `value`.
    ///
    /// This is a high-level measure of how different the newly rendered result and
    /// the existing snapshot are.
    /// This should be expected to be small, as a large value would represent the
    /// renderer not matching the previous snapshot.
    ///
    /// However, this value could potentially be non-zero (i.e. there is a slight difference
    /// between the new and previous results) due to e.g. fast math on the GPU or other
    /// platform specific factors.
    pub fn assert_mean_less_than(&mut self, value: f32) -> &mut Self {
        assert!(
            value < 0.1,
            "Mean should be less than 0.1 in almost all cases for a successful test"
        );
        if let Some(stats) = &self.statistics {
            let mean = stats.mean();
            if mean > value {
                self.handle_failure(format_args!(
                    "Expected mean to be less than {value}, got {mean}"
                ))
                .unwrap();
            }
        } else {
            // The result image was newly created, and so we know the test will pass
        }
        self.handle_success().unwrap();
        self
    }

    fn handle_success(&mut self) -> Result<()> {
        match std::fs::remove_file(&self.update_path) {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            res => res.map_err(Into::into),
        }
    }

    fn handle_failure(&mut self, message: fmt::Arguments<'_>) -> Result<()> {
        if env_var_relates_to("VELLO_TEST_UPDATE", &self.params.name, self.params.use_cpu) {
            if !self.params.use_cpu {
                write_png_to_file(
                    self.params,
                    &self.reference_path,
                    &self.raw_rendered,
                    Some(self.directory.max_size_in_bytes()),
                    true,
                )?;
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
            write_png_to_file(
                self.params,
                &self.update_path,
                &self.raw_rendered,
                None,
                false,
            )?;
            eprintln!(
                "Wrote result for failing test {} to {:?}\n\
                Use `VELLO_TEST_UPDATE=all` to update",
                self.params.name, &self.update_path
            );
        }
        bail!("{}", message);
    }
}

/// The directory to store a snapshot test within.
#[derive(Clone, Copy)]
pub enum SnapshotDirectory {
    /// Run in a smoke test directory.
    ///
    /// Snapshots in this directory should be small, because they commit binary data directly to the repository.
    /// This test will ensure that files produced are no more than 4KiB.
    Smoke,
    /// Run the test in a git LFS managed directory.
    ///
    /// This test will ensure that files produced are no larger than 64KiB.
    Lfs,
}

impl SnapshotDirectory {
    fn max_size_in_bytes(self) -> u64 {
        match self {
            Self::Smoke => 4 * 1024, /* 4KiB */
            Self::Lfs => 64 * 1024,  /* 64KiB */
        }
    }
}

/// Run a snapshot test of the given scene.
///
/// This will store the files in an LFS managed directory, and has a larger limit on file size.
///
/// This test will ensure that files produced are no larger than 64KiB.
pub fn snapshot_test_sync(scene: Scene, params: &TestParams) -> Result<Snapshot<'_>> {
    pollster::block_on(snapshot_test(scene, params, SnapshotDirectory::Lfs))
}

/// Run a snapshot test of the given scene.
///
/// Try and keep the width and height small, to reduce the size of committed binary data.
///
/// This test will ensure that files produced are no more than 4KiB.
pub fn smoke_snapshot_test_sync(scene: Scene, params: &TestParams) -> Result<Snapshot<'_>> {
    pollster::block_on(snapshot_test(scene, params, SnapshotDirectory::Smoke))
}

/// Run an snapshot test of the given scene.
///
/// In most cases, you should use [`snapshot_test_sync`] or [`smoke_snapshot_test_sync`].
pub async fn snapshot_test(
    scene: Scene,
    params: &TestParams,
    directory: SnapshotDirectory,
) -> Result<Snapshot<'_>> {
    let raw_rendered = render_then_debug(&scene, params).await?;
    snapshot_test_image(raw_rendered, params, directory)
}

/// Evaluate a snapshot test on the given image.
///
/// This is useful if a post-processing step needs to happen
/// in-between running Vello and the image.
pub fn snapshot_test_image(
    raw_rendered: ImageData,
    params: &TestParams,
    directory: SnapshotDirectory,
) -> Result<Snapshot<'_>> {
    let c_dir = current_dir(directory, params.use_cpu);
    std::fs::create_dir_all(&c_dir)?;
    let update_path = c_dir.join(&params.name).with_extension("png");

    let reference_path = snapshot_dir(directory)
        .join(&params.name)
        .with_extension("png");

    if env::var("VELLO_TEST_GENERATE_ALL").is_ok() {
        write_png_to_file(
            params,
            &update_path,
            &raw_rendered,
            Some(directory.max_size_in_bytes()),
            true,
        )?;
    }

    let expected_data = match image::open(&reference_path) {
        Ok(contents) => {
            let size = std::fs::metadata(&reference_path).map(|it| it.len())?;
            if size > directory.max_size_in_bytes()
                // If we expect to be updating the test, there's no need to fail here.
                && !env_var_relates_to("VELLO_TEST_UPDATE", &params.name, params.use_cpu)
            {
                bail!(
                    "Stored result for {test_name} is too large.\n\
                    Expected {max} bytes, got {size} bytes in {reference_path}",
                    max = directory.max_size_in_bytes(),
                    test_name = params.name,
                    reference_path = reference_path.display()
                );
            }

            contents.into_rgb8()
        }
        Err(ImageError::IoError(e)) if e.kind() == ErrorKind::NotFound => {
            if env_var_relates_to("VELLO_TEST_CREATE", &params.name, params.use_cpu) {
                if !params.use_cpu {
                    write_png_to_file(
                        params,
                        &reference_path,
                        &raw_rendered,
                        Some(directory.max_size_in_bytes()),
                        true,
                    )?;
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
                    statistics: None,
                    reference_path,
                    update_path,
                    raw_rendered,
                    directory,
                    params,
                });
            } else {
                write_png_to_file(
                    params,
                    &update_path,
                    &raw_rendered,
                    Some(directory.max_size_in_bytes()),
                    // This file could get copied by the user, so optimise as we're writing it out anyway.
                    true,
                )?;
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
        Err(ImageError::Decoding(d)) => {
            if env_var_relates_to("VELLO_SKIP_LFS_SNAPSHOTS", &params.name, params.use_cpu) {
                return Ok(Snapshot {
                    statistics: None,
                    reference_path,
                    update_path,
                    raw_rendered,
                    directory,
                    params,
                });
            } else {
                bail!(
                    "Decoding error: {d}\n\
                    in image file {reference_path:?}.\n\
                    If this file is an LFS file, install git lfs (https://git-lfs.com/) and run `git lfs pull`.\n\
                    If that fails (due to e.g. a lack of bandwidth), rerun tests with `VELLO_SKIP_LFS_SNAPSHOTS=all` to skip this test."
                )
            }
        }
        Err(e) => return Err(e.into()),
    };

    if expected_data.width() != raw_rendered.width || expected_data.height() != raw_rendered.height
    {
        let mut snapshot = Snapshot {
            statistics: None,
            reference_path,
            update_path,
            raw_rendered,
            directory,
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
    assert_eq!(raw_rendered.format, ImageFormat::Rgba8);
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

    let pool = FlipPool::from_image(&error_map);

    Ok(Snapshot {
        statistics: Some(pool),
        reference_path,
        update_path,
        raw_rendered,
        directory,
        params,
    })
}
