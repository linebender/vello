// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for probing the wgpu renderer.

use vello_common::probe;
use vello_gpu::{RenderTargetConfig, Renderer, WgpuProbeStatus};

use crate::renderer::{lock_wgpu_tests, shared_wgpu_device_queue};

#[test]
fn wgpu_probe_succeeds_with_depth_buffer() {
    run_wgpu_probe(true);
}

#[test]
fn wgpu_probe_succeeds_without_depth_buffer() {
    run_wgpu_probe(false);
}

fn run_wgpu_probe(use_depth_buffer: bool) {
    let _gpu_guard = lock_wgpu_tests();
    let (device, queue) = shared_wgpu_device_queue();

    let (width, height) = probe::canvas_size(probe::ALL_PROBE_ELEMENTS);
    let config = RenderTargetConfig {
        format: wgpu::TextureFormat::Rgba8Unorm,
        width,
        height,
    };
    let (mut renderer, _) = Renderer::new(&device, &queue, &config);
    let mut pending = renderer
        .probe(probe::ALL_PROBE_ELEMENTS, use_depth_buffer)
        .unwrap_or_else(|error| panic!("Renderer::probe() failed to render: {error:?}"));

    loop {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        match pending.try_finish() {
            Ok(WgpuProbeStatus::Complete(result)) => {
                assert!(result.is_success(), "probe failed unexpectedly");
                return;
            }
            Ok(WgpuProbeStatus::Pending(next_pending)) => pending = next_pending,
            Err(error) => panic!("Renderer::probe() readback failed: {error:?}"),
        }
    }
}
