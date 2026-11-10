// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for rendering into a texture with the wgpu renderer.

use vello_common::kurbo::Rect;
use vello_common::paint::{Image, ImageSource, TextureRegion};
use vello_common::peniko::ImageSampler;
use vello_cpu::{Level, RenderMode};
use vello_gpu::{
    RenderError, RenderOptions, RenderTarget, RenderTargetConfig, Scene, TextureBindings, TextureId,
};

use crate::renderer::{GpuRenderer, Renderer};

#[test]
fn texture_target_rejects_sampling_from_itself() {
    let ctx = GpuRenderer::new(1, 1, 0, Level::baseline(), RenderMode::OptimizeSpeed);
    let device = ctx.device();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (mut renderer, mut resources) = vello_gpu::Renderer::new(
        device,
        ctx.queue(),
        &RenderTargetConfig {
            format: texture.format(),
            width: 1,
            height: 1,
        },
    );
    let texture_id = TextureId(0);
    let mut bindings = TextureBindings::new();
    bindings.insert(texture_id, view.clone());

    let mut scene = Scene::new(1, 1);
    scene.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let result = renderer.render(
        &scene,
        &mut resources,
        &mut encoder,
        RenderTarget {
            view: &view,
            depth_view: None,
        },
        &bindings,
        RenderOptions::default(),
    );
    assert!(
        matches!(result, Err(RenderError::TextureFeedbackLoop(id)) if id == texture_id),
        "sampling from the render target should fail, got {result:?}"
    );
}
