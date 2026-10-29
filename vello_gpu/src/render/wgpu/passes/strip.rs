#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StripTargetKind {
    Root,
    Layer,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 6] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
            4 => Uint32,
            5 => Uint32,
        ]
    }
}

    /// `wgpu_external_texture` must be `Some` exactly if the layout has a `texture_external`
    /// binding.
    fn create_external_texture_bind_group(
        device: &Device,
        external_texture_bind_group_layout: &BindGroupLayout,
        texture_views: [&TextureView; EXTERNAL_TEXTURE_SLOT_COUNT],
        samplers: [&Sampler; EXTERNAL_TEXTURE_SLOT_COUNT],
        wgpu_external_texture: Option<wgpu::BindingResource<'_>>,
    ) -> BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(samplers[0]),
            },
        ];
        if let Some(resource) = wgpu_external_texture {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource,
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("External Texture Bind Group"),
            layout: external_texture_bind_group_layout,
            entries: &entries,
        })
    }

    fn create_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        config_buffer: &Buffer,
        strip_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Strip Bind Group"),
            layout: strip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(strip_texture_view),
                },
            ],
        })
    }

    /// Bind a [`wgpu::ExternalTexture`] to the `texture_external` binding and any other texture to
    /// the regular texture binding, with the placeholder in the unused one.
    fn create_run_external_texture_bind_group(
        &self,
        device: &Device,
        bindings: &[TextureBinding; EXTERNAL_TEXTURE_SLOT_COUNT],
        samplers: [&Sampler; EXTERNAL_TEXTURE_SLOT_COUNT],
    ) -> BindGroup {
        let placeholder = &self.resources.placeholder_external_texture_view;
        let [binding] = bindings;
        let (texture_view, wgpu_external_texture) = match binding {
            TextureBinding::TextureView(view) => (
                view,
                self.wgpu_external_textures
                    .then_some(wgpu::BindingResource::TextureView(placeholder)),
            ),
            #[cfg(feature = "wgpu_external_texture")]
            TextureBinding::WgpuExternalTexture { texture, .. } => (
                placeholder,
                Some(wgpu::BindingResource::ExternalTexture(texture)),
            ),
        };
        Self::create_external_texture_bind_group(
            device,
            &self.external_texture_bind_group_layout,
            [texture_view],
            samplers,
            wgpu_external_texture,
        )
    }

    /// Uploads two strip slices (opaque then alpha) into the frame arena and
    /// returns the byte range of the uploaded data.
    fn upload_strip_pair(
        &mut self,
        encoder: &mut CommandEncoder,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
    ) -> Range<u64> {
        let opaque_len = size_of_val(opaque_strips) as u64;
        let alpha_len = (alpha_strips.len() * size_of::<GpuStrip>()) as u64;
        let total_len = opaque_len + alpha_len;
        if total_len == 0 {
            return 0..0;
        }

        let arena = &mut self.strips_arena;
        let offset = arena.alloc(total_len);
        let size = NonZeroU64::new(total_len).expect("total length is non-zero");
        let mut view = self
            .staging_belt
            .write_buffer(encoder, &arena.buffer, offset, size);
        if opaque_len > 0 {
            view.slice(..opaque_len as usize)
                .copy_from_slice(bytemuck::cast_slice(opaque_strips));
        }
        let mut pos = opaque_len as usize;
        for strips in alpha_strips.slices() {
            let bytes = bytemuck::cast_slice(strips);
            view.slice(pos..pos + bytes.len()).copy_from_slice(bytes);
            pos += bytes.len();
        }

        offset..offset + total_len
    }
    fn external_texture_bind_group_for_textures(
        &mut self,
        bindings: ExternalTextureBindings,
    ) -> &BindGroup {
        let atlas_texture_views = &self.programs.resources.atlas_texture_views;
        let placeholder = &self.programs.resources.placeholder_external_texture_view;
        let texture_bindings = self.texture_bindings;
        match self.external_texture_bind_groups.entry(bindings) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let texture_sources = bindings.as_array();
                let slot_bindings = core::array::from_fn(|slot| match texture_sources[slot] {
                    Some(TextureSourceId::Atlas(atlas_id)) => TextureBinding::TextureView(
                        atlas_texture_views[atlas_id.as_u32() as usize].clone(),
                    ),
                    Some(TextureSourceId::External(texture_id, _)) => texture_bindings
                        .get(texture_id)
                        .expect("external texture binding was validated during paint preparation")
                        .clone(),
                    None => TextureBinding::TextureView(placeholder.clone()),
                });
                let sampler_keys = texture_sources.map(|source| match source {
                    Some(TextureSourceId::External(_, sampler)) => sampler,
                    _ => ExternalSampler::DEFAULT,
                });
                let samplers = sampler_keys.map(|key| {
                    self.programs
                        .external_samplers
                        .entry(key)
                        .or_insert_with(|| Programs::create_external_sampler(self.device, key))
                        .clone()
                });
                let bind_group = self.programs.create_run_external_texture_bind_group(
                    self.device,
                    &slot_bindings,
                    samplers.each_ref(),
                );
                entry.insert(bind_group)
            }
        }
    }

    /// Render the strips to the specified render target.
    fn strip_pass_inner(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        target: DrawPassTarget,
        child_layer_texture: Option<LayerTextureId>,
    ) {
        let opaque_count = opaque_strips.len();
        let alpha_count = alpha_strips.len();

        if opaque_count == 0 && alpha_count == 0 {
            return;
        }
        // Create bind groups for all external textures used by this pass.
        for run in external_texture_runs {
            self.external_texture_bind_group_for_textures(run.bindings);
        }

        let strips_range =
            self.programs
                .upload_strip_pair(self.encoder, opaque_strips, alpha_strips);
        let opaque_count = opaque_count as u32;
        let alpha_count = alpha_count as u32;

        let color_load_op = if matches!(target, DrawPassTarget::Root(_)) {
            self.take_root_load_op()
        } else {
            // For layer targets we always want to load.
            wgpu::LoadOp::Load
        };
        let (view, config_buffer, bind_group_target) = match target {
            DrawPassTarget::Root(_) => (
                self.view,
                &self.programs.resources.view_config_buffer,
                StripTargetKind::Root,
            ),
            DrawPassTarget::Layer(id) => (
                self.programs.resources.layer_view(id),
                &self.programs.resources.layer_config_buffer,
                StripTargetKind::Layer,
            ),
        };
        let bind_group_key = (bind_group_target, child_layer_texture);
        if !self
            .programs
            .strip_layer_bind_groups
            .contains_key(&bind_group_key)
        {
            let child_layer_view = child_layer_texture.map_or(
                &self.programs.resources.placeholder_external_texture_view,
                |id| self.programs.resources.layer_view(id),
            );
            let alphas_texture_view = self
                .programs
                .resources
                .alphas_texture
                .create_view(&TextureViewDescriptor::default());
            let bind_group = Programs::create_strip_bind_group(
                self.device,
                &self.programs.strip_bind_group_layout,
                &alphas_texture_view,
                config_buffer,
                child_layer_view,
            );
            self.programs
                .strip_layer_bind_groups
                .insert(bind_group_key, bind_group);
        }

        let bind_group = &self.programs.strip_layer_bind_groups[&bind_group_key];

        let enable_opaque = self.depth_view.is_some() && target.enable_opaque();

        let depth_stencil_attachment = if enable_opaque {
            let depth_load = if self.programs.depth_cleared_this_frame {
                wgpu::LoadOp::Load
            } else {
                self.programs.depth_cleared_this_frame = true;
                wgpu::LoadOp::Clear(1.0)
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: self
                    .depth_view
                    .expect("enabled depth buffering requires a depth texture view"),
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load_op,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(2, &self.programs.resources.encoded_paints_bind_group, &[]);
        render_pass.set_bind_group(3, &self.programs.resources.gradient_bind_group, &[]);
        render_pass.set_vertex_buffer(
            0,
            self.programs
                .strips_arena
                .buffer
                .slice(strips_range.clone()),
        );

        let draw_strip_runs = |render_pass: &mut wgpu::RenderPass<'_>, first_instance, count| {
            if external_texture_runs.is_empty() {
                render_pass.set_bind_group(
                    1,
                    &self.programs.resources.empty_external_texture_bind_group,
                    &[],
                );
                render_pass.draw(0..4, first_instance..first_instance + count);

                return;
            }

            // Each run is drawn with different external texture bindings. Runs go from
            // `run.strips_start` to the next run's `strips_start`; the last run goes to the end of
            // the strips buffer.
            for (i, run) in external_texture_runs.iter().enumerate() {
                let external_texture_bind_group = self
                    .external_texture_bind_groups
                    .get(&run.bindings)
                    .unwrap();
                render_pass.set_bind_group(1, external_texture_bind_group, &[]);
                let start = u32::try_from(run.strips_start).unwrap();
                let end = external_texture_runs
                    .get(i + 1)
                    .map_or(count, |next| u32::try_from(next.strips_start).unwrap());
                render_pass.draw(0..4, first_instance + start..first_instance + end);
            }
        };

        if opaque_count > 0 {
            // Opaque pass
            debug_assert!(
                enable_opaque,
                "opaque strips require the final view depth attachment"
            );
            render_pass.set_pipeline(&self.programs.opaque_strip_pipeline);
            draw_strip_runs(&mut render_pass, 0, opaque_count);
        }

        if alpha_count > 0 {
            // Alpha pass
            if matches!(target, DrawPassTarget::Root(RootTarget::UserSurface)) {
                let pipeline = if enable_opaque {
                    &self.programs.depth_alpha_strip_pipeline
                } else {
                    &self.programs.alpha_strip_pipeline
                };
                render_pass.set_pipeline(pipeline);
            } else {
                render_pass.set_pipeline(&self.programs.intermediate_strip_pipeline);
            }

            draw_strip_runs(&mut render_pass, opaque_count, alpha_count);
        }
    }

