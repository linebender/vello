// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for scenes whose packed data doesn't fit into the renderer's resource textures.

use smallvec::smallvec;
use vello_common::color::palette::css::{BLUE, WHITE};
use vello_common::color::{AlphaColor, DynamicColor};
use vello_common::kurbo::Rect;
use vello_common::peniko::{ColorStop, ColorStops, Gradient, LinearGradientPosition};
use vello_gpu::{
    RenderError, RenderOptions, RenderTarget, RenderTargetConfig, Renderer, ResourceTexture, Scene,
    TextureBindings,
};

#[test]
fn renderer_recovers_after_gradient_overflow() {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 64,
            ..adapter.limits()
        },
        ..Default::default()
    }))
    .expect("Failed to create device");

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let config = RenderTargetConfig {
        format,
        width: 32,
        height: 32,
    };
    let (mut renderer, mut resources) = Renderer::new(&device, &queue, &config);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut render = |scene: &Scene| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let target = RenderTarget {
            view: &view,
            depth_view: None,
        };
        let result = renderer.render(
            scene,
            &mut resources,
            &mut encoder,
            target,
            &TextureBindings::new(),
            RenderOptions::default(),
        );
        queue.submit([encoder.finish()]);
        result
    };
    let rect = Rect::new(0.0, 0.0, 32.0, 32.0);

    let mut gradients = Scene::new(32, 32);
    for i in 0..32_u8 {
        let start = DynamicColor::from_alpha_color(AlphaColor::from_rgba8(i * 8, 0, 0, 255));
        let end = DynamicColor::from_alpha_color(WHITE);
        gradients.set_paint(Gradient {
            kind: LinearGradientPosition::new((0.0, 0.0), (32.0, 0.0)).into(),
            stops: ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: start,
                },
                ColorStop {
                    offset: 1.0,
                    color: end,
                },
            ]),
            ..Default::default()
        });
        gradients.fill_rect(&rect);
    }
    assert!(matches!(
        render(&gradients),
        Err(RenderError::ResourceTextureTooLarge {
            texture: ResourceTexture::Gradients,
            ..
        })
    ));

    let mut solid = Scene::new(32, 32);
    solid.set_paint(BLUE);
    solid.fill_rect(&rect);
    render(&solid).expect("a scene without gradients should render after a gradient overflow");
}
