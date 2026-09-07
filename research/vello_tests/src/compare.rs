// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use image::DynamicImage;
use nv_flip::FlipPool;
use vello::{
    Scene,
    peniko::{ImageData, ImageFormat},
};

use crate::{TestParams, render_then_debug, write_png_to_file};

fn comparison_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("comparisons")
}

#[must_use = "A snapshot test doesn't do anything unless an assertion method is called on it"]
/// A scene rendered on the CPU and the same scene rendered on the GPU, and information about their differences.
///
/// Use an assertion method or access `statistics` to make a determination based on the result of this test.
pub struct GpuCpuComparison {
    pub statistics: Option<FlipPool>,
    pub cpu_path: PathBuf,
    pub gpu_path: PathBuf,
    pub cpu_rendered: ImageData,
    pub gpu_rendered: ImageData,
    pub params: TestParams,
}

impl GpuCpuComparison {
    /// Assert that that mean value stored in `statistics` is less than `value`.
    ///
    /// This is a high-level measure of how different the GPU and CPU results are.
    /// This should be expected to be small, a large value would represent a
    /// significant difference between the two results.
    ///
    /// However, this value could potentially be non-zero (i.e. there is a slight difference
    /// between the GPU and CPU results) due to fast math on the GPU or different precisions
    /// used in the renderers.
    pub fn assert_mean_less_than(&mut self, value: f32) -> &mut Self {
        assert!(
            value < 0.1,
            "Mean should be less than 0.1 in almost all cases for a successful test"
        );
        if let Some(stats) = &self.statistics {
            let mean = stats.mean();
            if mean > value {
                self.handle_failure(format!("Expected mean to be less than {value}, got {mean}"))
                    .unwrap();
            }
        } else {
            // The result image was newly created, and so we know the test will pass
        }
        self.handle_success().unwrap();
        self
    }

    fn handle_success(&mut self) -> Result<()> {
        match std::fs::remove_file(&self.cpu_path) {
            Err(e) if e.kind() == ErrorKind::NotFound => (),
            res => return res.map_err(Into::into),
        }
        match std::fs::remove_file(&self.gpu_path) {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            res => res.map_err(Into::into),
        }
    }

    fn handle_failure(&mut self, message: String) -> Result<()> {
        write_png_to_file(
            &self.params,
            &self.cpu_path,
            &self.cpu_rendered,
            None,
            false,
        )?;
        write_png_to_file(
            &self.params,
            &self.gpu_path,
            &self.gpu_rendered,
            None,
            false,
        )?;
        eprintln!(
            "Wrote CPU result from test {} to {:?}\n\
            Wrote GPU result to {:?}\n",
            self.params.name, &self.cpu_path, &self.gpu_path
        );

        bail!("{}", message);
    }
}

/// Run a scene comparing the outputs from the CPU and GPU renderers
pub fn compare_gpu_cpu_sync(scene: Scene, params: TestParams) -> Result<GpuCpuComparison> {
    pollster::block_on(compare_gpu_cpu(scene, params))
}

/// Run a scene comparing the outputs from the CPU and GPU renderers
pub async fn compare_gpu_cpu(scene: Scene, mut params: TestParams) -> Result<GpuCpuComparison> {
    params.use_cpu = false;
    // TODO: Reuse the same RenderContext?
    let gpu_rendered = render_then_debug(&scene, &params).await?;
    params.use_cpu = true;
    let cpu_rendered = render_then_debug(&scene, &params).await?;

    let path_root = &comparison_dir();
    let cpu_dir = path_root.join("cpu");
    std::fs::create_dir_all(&cpu_dir)?;
    let gpu_dir = path_root.join("gpu");
    std::fs::create_dir_all(&gpu_dir)?;
    let cpu_path = cpu_dir.join(&params.name).with_extension(".png");
    let gpu_path = path_root.join(&params.name).with_extension(".png");

    assert!(gpu_rendered.width == cpu_rendered.width && gpu_rendered.height == cpu_rendered.height,);

    // Compare the images using nv-flip
    assert_eq!(cpu_rendered.format, ImageFormat::Rgba8);
    assert_eq!(gpu_rendered.format, ImageFormat::Rgba8);
    let gpu_rendered_data: DynamicImage = image::RgbaImage::from_raw(
        cpu_rendered.width,
        cpu_rendered.height,
        cpu_rendered.data.as_ref().to_vec(),
    )
    .ok_or(anyhow!("Couldn't create image for cpu result"))?
    .into();
    let gpu_rendered_data = gpu_rendered_data.to_rgb8();

    let cpu_rendered_data: DynamicImage = image::RgbaImage::from_raw(
        cpu_rendered.width,
        cpu_rendered.height,
        cpu_rendered.data.as_ref().to_vec(),
    )
    .ok_or(anyhow!("Couldn't create image for cpu result"))?
    .into();
    let cpu_rendered_data = cpu_rendered_data.to_rgb8();

    let cpu_flip = nv_flip::FlipImageRgb8::with_data(
        cpu_rendered_data.width(),
        cpu_rendered_data.height(),
        &cpu_rendered_data,
    );
    let gpu_flip = nv_flip::FlipImageRgb8::with_data(
        gpu_rendered_data.width(),
        gpu_rendered_data.height(),
        &gpu_rendered_data,
    );

    let error_map = nv_flip::flip(cpu_flip, gpu_flip, nv_flip::DEFAULT_PIXELS_PER_DEGREE);

    let pool = FlipPool::from_image(&error_map);

    Ok(GpuCpuComparison {
        statistics: Some(pool),
        cpu_path,
        gpu_path,
        cpu_rendered,
        gpu_rendered,
        params,
    })
}
