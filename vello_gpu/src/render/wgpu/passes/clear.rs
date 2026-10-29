    fn clear_full_target(encoder: &mut CommandEncoder, view: &TextureView, color: wgpu::Color) {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Target"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }

    fn clear_pass_inner(&mut self, target: DrawPassTarget, settings: ClearSettings<'_>) {
        let color = settings.clear_color();

        let view = match target {
            DrawPassTarget::Root(_) => self.view,
            DrawPassTarget::Layer(target) => self.programs.resources.layer_view(target),
        };
        let rects = match settings {
            ClearSettings::Viewport { .. } => {
                Self::clear_full_target(self.encoder, view, clear_color(color));

                return;
            }
            ClearSettings::Rects { rects, .. } => rects,
        };

        let (target_size, pipeline) = match target {
            DrawPassTarget::Root(_) => (
                [
                    self.programs.render_size.width,
                    self.programs.render_size.height,
                ],
                &self.programs.root_clear_pipeline,
            ),
            DrawPassTarget::Layer(_) => {
                let texture_size = self.texture_size();
                (
                    [texture_size.width(), texture_size.height()],
                    &self.programs.clear_pipeline,
                )
            }
        };
        let bounds = RectU16::new(0, 0, target_size[0], target_size[1]);
        let target_size = target_size.map(u32::from);
        let color = color.premultiply().to_rgba8().to_u32();
        self.scratch_buffers.clear_instances.clear();
        self.scratch_buffers.clear_instances.extend(
            rects
                .iter()
                .map(|rect| rect.intersect(bounds))
                .filter(|rect| !rect.is_empty())
                .map(|rect| GpuClearInstance {
                    origin: [u32::from(rect.x0), u32::from(rect.y0)],
                    size: [u32::from(rect.width()), u32::from(rect.height())],
                    target_size,
                    color,
                }),
        );

        if self.scratch_buffers.clear_instances.is_empty() {
            return;
        }

        let clear_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Clear Buffer"),
                contents: bytemuck::cast_slice(&self.scratch_buffers.clear_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Rects"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_vertex_buffer(0, clear_buffer.slice(..));
        render_pass.draw(
            0..4,
            0..u32::try_from(self.scratch_buffers.clear_instances.len()).unwrap(),
        );
    }
fn clear_color(color: AlphaColor<Srgb>) -> wgpu::Color {
    let [r, g, b, a] = color.premultiply().components;
    wgpu::Color {
        r: f64::from(r),
        g: f64::from(g),
        b: f64::from(b),
        a: f64::from(a),
    }
}
