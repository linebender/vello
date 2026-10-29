    fn create_blend_layer_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        layer_texture_views: [&TextureView; 2],
        alphas_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Layer Textures Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
            ],
        })
    }

    fn blend_pass_inner(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) {
        let texture_size = self.texture_size();
        let resources = &self.programs.resources;
        let blend_layer_bind_group = match self.programs.blend_layer_bind_groups.entry(bindings) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let alphas_texture_view = resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default());
                entry.insert(Programs::create_blend_layer_bind_group(
                    self.device,
                    &self.programs.blend_layer_bind_group_layout,
                    resources.layer_views([
                        bindings.layer_id(TextureParity::Even),
                        bindings.layer_id(TextureParity::Odd),
                    ]),
                    &alphas_texture_view,
                ))
            }
        };

        self.scratch_buffers.blend_instances.clear();

        for blend in blends.iter() {
            if let Some(range) = &blend.clip_strips {
                self.scratch_buffers.blend_instances.extend(
                    blend_strips[usize::try_from(range.start).unwrap()
                        ..usize::try_from(range.end).unwrap()]
                        .iter()
                        .copied()
                        .map(|strip| GpuBlendInstance::new(blend, Some(strip), texture_size)),
                );
            } else {
                self.scratch_buffers
                    .blend_instances
                    .push(GpuBlendInstance::new(blend, None, texture_size));
            }
        }

        let instance_count = u32::try_from(self.scratch_buffers.blend_instances.len()).unwrap();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Blend Instances Buffer"),
                contents: bytemuck::cast_slice(&self.scratch_buffers.blend_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Blend To Scratch"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: self.programs.resources.scratch_view(),
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
            render_pass.set_pipeline(&self.programs.blend_pipeline);
            render_pass.set_bind_group(0, &*blend_layer_bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }

        self.scratch_buffers.copy_instances.clear();
        self.scratch_buffers.copy_instances.extend(
            self.scratch_buffers
                .blend_instances
                .iter()
                .copied()
                .map(GpuBlendInstance::copy_from_scratch),
        );
        encode_copy_pass(
            self.device,
            self.encoder,
            &self.programs.copy_pipeline,
            &self.scratch_buffers.copy_instances,
            &self.programs.resources.scratch_copy_bind_group,
            self.programs.resources.layer_view(bindings.target()),
            "Copy Blend Scratch To Layer",
        );
    }

