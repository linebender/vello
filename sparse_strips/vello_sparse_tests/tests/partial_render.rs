// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Correctness pins for `vello_hybrid`'s partial (damage-region) rendering via
//! [`RenderRegion::Rects`] and [`ClearSettings::Rects`].

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
mod tests {
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Rect;
    use vello_common::peniko::Color;
    use vello_common::pixmap::Pixmap;
    use vello_hybrid::{ClearSettings, RenderRegion};

    use crate::renderer::{HybridRenderer, Renderer};
    use crate::util::{get_ctx, render_pixmap};
    use vello_cpu::RenderMode;

    const W: u16 = 128;
    const H: u16 = 64;

    fn hybrid() -> HybridRenderer {
        get_ctx::<HybridRenderer>(W, H, false, 0, "fallback", RenderMode::OptimizeQuality)
    }

    fn paint_scene(ctx: &mut HybridRenderer) {
        ctx.set_paint(Color::new([0.086, 0.106, 0.133, 1.0]));
        ctx.fill_rect(&Rect::new(0.0, 0.0, W as f64, H as f64));
        ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
        ctx.fill_rect(&Rect::new(10.3, 8.7, 60.6, 40.2));
        ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.7]));
        ctx.fill_rect(&Rect::new(70.5, 12.25, 115.75, 50.5));
    }

    fn pixel(p: &Pixmap, x: u16, y: u16) -> [u8; 4] {
        let i = (y as usize * W as usize + x as usize) * 4;
        let d = p.data_as_u8_slice();
        [d[i], d[i + 1], d[i + 2], d[i + 3]]
    }

    /// A viewport-clear render confined to a damage rect must reproduce the
    /// full render inside the rect and leave everything outside cleared to
    /// transparent — i.e. the region actually confines drawing.
    #[test]
    fn partial_render_clear_confines_to_scissor() {
        let mut full_ctx = hybrid();
        paint_scene(&mut full_ctx);
        let full = render_pixmap(&mut full_ctx);

        let damage = RectU16::new(16, 8, 96, 48);
        let mut scissored_ctx = hybrid();
        paint_scene(&mut scissored_ctx);
        let mut scissored = Pixmap::new(W, H);
        scissored_ctx.render_region_to_pixmap(
            ClearSettings::default(),
            RenderRegion::Rects(&[damage]),
            false,
            &mut scissored,
        );

        for y in 0..H {
            for x in 0..W {
                let inside = x >= damage.x0 && x < damage.x1 && y >= damage.y0 && y < damage.y1;
                if inside {
                    assert_eq!(
                        pixel(&scissored, x, y),
                        pixel(&full, x, y),
                        "in-scissor pixel ({x}, {y}) must match the full render"
                    );
                } else {
                    assert_eq!(
                        pixel(&scissored, x, y),
                        [0, 0, 0, 0],
                        "out-of-scissor pixel ({x}, {y}) must stay cleared"
                    );
                }
            }
        }
    }

    /// A full render followed by a damage render (clear the damaged rect, redraw
    /// only it) reproduces the full render everywhere: the damaged region is
    /// redrawn identically and the rest of the target is preserved.
    #[test]
    fn partial_render_damage_preserves_and_matches_full() {
        let mut full_ctx = hybrid();
        paint_scene(&mut full_ctx);
        let full = render_pixmap(&mut full_ctx);

        let damage = RectU16::new(16, 8, 96, 48);
        let mut damaged_ctx = hybrid();
        paint_scene(&mut damaged_ctx);
        let mut damaged = Pixmap::new(W, H);
        damaged_ctx.render_region_to_pixmap(
            ClearSettings::Rects {
                color: Color::TRANSPARENT,
                rects: &[damage],
            },
            RenderRegion::Rects(&[damage]),
            true,
            &mut damaged,
        );

        let diffs = damaged
            .data_as_u8_slice()
            .iter()
            .zip(full.data_as_u8_slice())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "damage render must be byte-identical to the full render ({diffs} differing bytes)"
        );
    }

    /// Two disjoint damage rects clear and redraw independently, still matching
    /// the full render (exercises the multi-rect scissor loop and the rect clear).
    #[test]
    fn partial_render_disjoint_damage_rects_match_full() {
        let mut full_ctx = hybrid();
        paint_scene(&mut full_ctx);
        let full = render_pixmap(&mut full_ctx);

        let damage = [RectU16::new(8, 8, 40, 40), RectU16::new(80, 16, 120, 56)];
        let mut damaged_ctx = hybrid();
        paint_scene(&mut damaged_ctx);
        let mut damaged = Pixmap::new(W, H);
        damaged_ctx.render_region_to_pixmap(
            ClearSettings::Rects {
                color: Color::TRANSPARENT,
                rects: &damage,
            },
            RenderRegion::Rects(&damage),
            true,
            &mut damaged,
        );

        let diffs = damaged
            .data_as_u8_slice()
            .iter()
            .zip(full.data_as_u8_slice())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "disjoint-damage render must be byte-identical to the full render ({diffs} differing bytes)"
        );

        // The region render is the partial one; the preceding full render is
        // not counted.
        assert_eq!(
            damaged_ctx.partial_renders(),
            1,
            "the region render must be counted as a partial render"
        );
        // The two scenes' edge quads that fall in the gap between the disjoint
        // damage rects have no scissored draw, so the cull must skip some.
        assert!(
            damaged_ctx.culled_strips() > 0,
            "the damage-region cull must skip strips in the gap between disjoint rects"
        );
    }

    /// Overlapping damage rects must composite translucent content exactly
    /// once. The root draw sequence is replayed once per scissor rect, so
    /// without normalizing the region into a disjoint union, the overlap
    /// would blend the translucent fill twice and diverge from a full render.
    /// The scene deliberately has no opaque backdrop: an opaque first draw
    /// would reset the overlap on every replay and mask the double blend.
    #[test]
    fn partial_render_overlapping_damage_rects_match_full() {
        let paint_translucent = |ctx: &mut HybridRenderer| {
            ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.7]));
            ctx.fill_rect(&Rect::new(4.0, 4.0, 124.0, 60.0));
        };

        let mut full_ctx = hybrid();
        paint_translucent(&mut full_ctx);
        let full = render_pixmap(&mut full_ctx);

        // Two rects overlapping over x 40..88, y 16..48.
        let damage = [RectU16::new(8, 8, 88, 48), RectU16::new(40, 16, 120, 56)];
        let mut damaged_ctx = hybrid();
        paint_translucent(&mut damaged_ctx);
        let mut damaged = Pixmap::new(W, H);
        damaged_ctx.render_region_to_pixmap(
            ClearSettings::default(),
            RenderRegion::Rects(&damage),
            false,
            &mut damaged,
        );

        for y in 0..H {
            for x in 0..W {
                let inside = damage
                    .iter()
                    .any(|r| x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1);
                if inside {
                    assert_eq!(
                        pixel(&damaged, x, y),
                        pixel(&full, x, y),
                        "in-region pixel ({x}, {y}) must match the full render \
                         (a mismatch inside the overlap means it was composited twice)"
                    );
                } else {
                    assert_eq!(
                        pixel(&damaged, x, y),
                        [0, 0, 0, 0],
                        "out-of-region pixel ({x}, {y}) must stay cleared"
                    );
                }
            }
        }
    }

    /// [`ClearSettings::Rects`] renders into the caller's target, so its
    /// pipeline must be built for the render target's format — not for the
    /// `Rgba8Unorm` the renderer's own layer textures use.
    ///
    /// The pixel tests above all run on an `Rgba8Unorm` target, where the two
    /// formats coincide. A real swapchain is typically `Bgra8Unorm`, and there
    /// a layer-format clear pipeline is incompatible with the render pass:
    /// wgpu rejects the `set_pipeline`, which invalidates the encoder and drops
    /// the whole frame. Assert the damage-region recipe raises no validation
    /// error on a target whose format differs from the layer format.
    #[test]
    fn clear_rects_matches_the_render_target_format() {
        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

        let instance = wgpu::Instance::default();
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
        else {
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("clear_rects target format"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width: W.into(),
                height: H.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut renderer = vello_hybrid::Renderer::new(
            &device,
            &vello_hybrid::RenderTargetConfig {
                format: FORMAT,
                width: W.into(),
                height: H.into(),
            },
        );
        let mut resources = vello_hybrid::Resources::new();
        let texture_bindings = vello_hybrid::TextureBindings::new();
        let render_size = vello_hybrid::RenderSize {
            width: W.into(),
            height: H.into(),
        };

        let mut scene = vello_hybrid::Scene::new(W, H);
        scene.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
        scene.fill_rect(&Rect::new(0.0, 0.0, W as f64, H as f64));

        let damage = [RectU16::new(16, 8, 96, 48)];
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Damage Render"),
        });
        renderer
            .render(
                &scene,
                &mut resources,
                &device,
                &queue,
                &mut encoder,
                &render_size,
                &view,
                &texture_bindings,
                ClearSettings::Rects {
                    color: Color::TRANSPARENT,
                    rects: &damage,
                },
                RenderRegion::Rects(&damage),
            )
            .unwrap();
        queue.submit([encoder.finish()]);

        let error = pollster::block_on(scope.pop());
        assert!(
            error.is_none(),
            "a rect clear on a {FORMAT:?} target must not raise a validation error: {error:?}"
        );
    }
}
