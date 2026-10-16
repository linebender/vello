// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for probing the wgpu renderer.

use vello_common::probe::{self, Probe, ProbeFeature};
use vello_gpu::{
    RenderError, RenderTargetConfig, Renderer, WgpuPendingProbe, WgpuProbeError, WgpuProbeStatus,
};

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("Failed to create device")
}

fn renderer(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
    let (width, height) = probe::canvas_size(probe::ALL_PROBE_ELEMENTS);
    let config = RenderTargetConfig {
        format,
        width,
        height,
    };
    Renderer::new(device, &config).0
}

fn finish(device: &wgpu::Device, mut pending: WgpuPendingProbe) -> Probe<RenderError> {
    loop {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        match pending.try_finish(device).unwrap() {
            WgpuProbeStatus::Pending(still_pending) => pending = still_pending,
            WgpuProbeStatus::Complete(result) => return result,
        }
    }
}

fn assert_probe_succeeds(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    elements: &[ProbeFeature],
    use_depth_buffer: bool,
) {
    let pending = renderer
        .probe(device, queue, elements, use_depth_buffer)
        .unwrap();
    match finish(device, pending) {
        Probe::Success => {}
        Probe::Error(result) => panic!(
            "probe of {elements:?} (depth buffer: {use_depth_buffer}) did not match the \
             reference: {:#?}",
            result.statistics
        ),
        Probe::RenderError(error) => panic!("probe failed to render: {error}"),
    }
}

#[test]
fn wgpu_probe_succeeds() {
    let (device, queue) = device();
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ] {
        let mut renderer = renderer(&device, format);
        for use_depth_buffer in [false, true] {
            assert_probe_succeeds(
                &device,
                &queue,
                &mut renderer,
                probe::ALL_PROBE_ELEMENTS,
                use_depth_buffer,
            );
        }
    }
}

#[test]
fn wgpu_probe_succeeds_for_individual_elements() {
    let (device, queue) = device();
    let mut renderer = renderer(&device, wgpu::TextureFormat::Rgba8Unorm);
    for element in probe::ALL_PROBE_ELEMENTS {
        assert_probe_succeeds(&device, &queue, &mut renderer, &[*element], true);
    }
}

#[test]
fn wgpu_probe_rejects_invalid_requests() {
    let (device, queue) = device();
    let mut rgba8 = renderer(&device, wgpu::TextureFormat::Rgba8Unorm);
    assert!(matches!(
        rgba8.probe(&device, &queue, &[], false),
        Err(WgpuProbeError::NoElements)
    ));

    let mut rgba16 = renderer(&device, wgpu::TextureFormat::Rgba16Float);
    assert!(matches!(
        rgba16.probe(&device, &queue, probe::ALL_PROBE_ELEMENTS, false),
        Err(WgpuProbeError::UnsupportedFormat(
            wgpu::TextureFormat::Rgba16Float
        ))
    ));
}
