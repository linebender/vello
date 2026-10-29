    fn create_copy_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        source_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Copy Source Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_texture_view),
            }],
        })
    }

fn encode_copy_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    copy_pipeline: &RenderPipeline,
    instances: &[GpuCopyInstance],
    source_bind_group: &BindGroup,
    output_view: &TextureView,
    label: &'static str,
) {
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: output_view,
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
    render_pass.set_pipeline(copy_pipeline);
    render_pass.set_bind_group(0, source_bind_group, &[]);
    render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
    render_pass.draw(0..4, 0..u32::try_from(instances.len()).unwrap());
}

