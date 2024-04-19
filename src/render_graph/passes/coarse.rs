use vello_encoding::{RenderConfig, Resolver, WorkgroupSize};

use crate::{
    render_graph::{Handle, PassContext},
    BufferProxy, ImageFormat, ImageProxy, Recording, ResourceProxy,
};

use super::RenderPass;

pub struct VelloCoarse {}

#[derive(Clone, Copy)]
pub struct CoarseOutput {
    pub config_buf: Handle<BufferProxy>,
    pub tile_buf: Handle<BufferProxy>,
    pub segments_buf: Handle<BufferProxy>,
    pub ptcl_buf: Handle<BufferProxy>,
    pub gradient_image: Handle<ImageProxy>,
    pub info_bin_data_buf: Handle<BufferProxy>,
    pub image_atlas: Handle<ImageProxy>,

    pub out_image: Handle<ImageProxy>,

    pub fine_workgroup_size: WorkgroupSize,
}

impl RenderPass for VelloCoarse {
    type Output = CoarseOutput;

    fn record(self, cx: PassContext<'_>) -> (Recording, Self::Output)
    where
        Self: Sized,
    {
        let mut recording = Recording::default();

        let mut resolver = Resolver::new();
        let mut packed = vec![];
        let (layout, ramps, images) = resolver.resolve(cx.encoding, &mut packed);
        let gradient_image = if ramps.height == 0 {
            ImageProxy::new(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            recording.upload_image(ramps.width, ramps.height, ImageFormat::Rgba8, data)
        };
        let image_atlas = if images.images.is_empty() {
            ImageProxy::new(1, 1, ImageFormat::Rgba8)
        } else {
            ImageProxy::new(images.width, images.height, ImageFormat::Rgba8)
        };
        for image in images.images {
            recording.write_image(
                image_atlas,
                image.1,
                image.2,
                image.0.width,
                image.0.height,
                image.0.data.data(),
            );
        }

        let cpu_config = RenderConfig::new(
            &layout,
            cx.params.width,
            cx.params.height,
            &cx.params.base_color,
        );
        let buffer_sizes = &cpu_config.buffer_sizes;
        let wg_counts = &cpu_config.workgroup_counts;

        let scene_buf = recording.upload("scene", packed);
        let config_buf = recording.upload_uniform("config", bytemuck::bytes_of(&cpu_config.gpu));
        let info_bin_data_buf = BufferProxy::new(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            "info_bin_data_buf",
        );
        let tile_buf = BufferProxy::new(buffer_sizes.tiles.size_in_bytes().into(), "tile_buf");
        let segments_buf =
            BufferProxy::new(buffer_sizes.segments.size_in_bytes().into(), "segments_buf");
        let ptcl_buf = BufferProxy::new(buffer_sizes.ptcl.size_in_bytes().into(), "ptcl_buf");
        let reduced_buf = BufferProxy::new(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            "reduced_buf",
        );
        // TODO: really only need pathtag_wgs - 1
        recording.dispatch(
            cx.shaders.pathtag_reduce,
            wg_counts.path_reduce,
            (config_buf, scene_buf, reduced_buf),
        );
        let mut pathtag_parent = reduced_buf;
        let mut large_pathtag_bufs = None;
        let use_large_path_scan = wg_counts.use_large_path_scan && !cx.shaders.pathtag_is_cpu;
        if use_large_path_scan {
            let reduced2_buf = BufferProxy::new(
                buffer_sizes.path_reduced2.size_in_bytes().into(),
                "reduced2_buf",
            );
            recording.dispatch(
                cx.shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                (reduced_buf, reduced2_buf),
            );
            let reduced_scan_buf = BufferProxy::new(
                buffer_sizes.path_reduced_scan.size_in_bytes().into(),
                "reduced_scan_buf",
            );
            recording.dispatch(
                cx.shaders.pathtag_scan1,
                wg_counts.path_scan1,
                (reduced_buf, reduced2_buf, reduced_scan_buf),
            );
            pathtag_parent = reduced_scan_buf;
            large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
        }

        let tagmonoid_buf = BufferProxy::new(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            "tagmonoid_buf",
        );
        let pathtag_scan = if use_large_path_scan {
            cx.shaders.pathtag_scan_large
        } else {
            cx.shaders.pathtag_scan
        };
        recording.dispatch(
            pathtag_scan,
            wg_counts.path_scan,
            (config_buf, scene_buf, pathtag_parent, tagmonoid_buf),
        );
        recording.free_resource(reduced_buf.into());
        if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
            recording.free_resource(reduced2.into());
            recording.free_resource(reduced_scan.into());
        }
        let path_bbox_buf = BufferProxy::new(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            "path_bbox_buf",
        );
        recording.dispatch(
            cx.shaders.bbox_clear,
            wg_counts.bbox_clear,
            (config_buf, path_bbox_buf),
        );
        let bump_buf = BufferProxy::new(buffer_sizes.bump_alloc.size_in_bytes().into(), "bump_buf");
        recording.clear_all(bump_buf);
        let lines_buf = BufferProxy::new(buffer_sizes.lines.size_in_bytes().into(), "lines_buf");
        recording.dispatch(
            cx.shaders.flatten,
            wg_counts.flatten,
            (
                config_buf,
                scene_buf,
                tagmonoid_buf,
                path_bbox_buf,
                bump_buf,
                lines_buf,
            ),
        );
        let draw_reduced_buf = BufferProxy::new(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            "draw_reduced_buf",
        );
        recording.dispatch(
            cx.shaders.draw_reduce,
            wg_counts.draw_reduce,
            (config_buf, scene_buf, draw_reduced_buf),
        );
        let draw_monoid_buf = BufferProxy::new(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            "draw_monoid_buf",
        );
        let clip_inp_buf = BufferProxy::new(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            "clip_inp_buf",
        );
        recording.dispatch(
            cx.shaders.draw_leaf,
            wg_counts.draw_leaf,
            (
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ),
        );
        recording.free_resource(draw_reduced_buf.into());
        let clip_el_buf =
            BufferProxy::new(buffer_sizes.clip_els.size_in_bytes().into(), "clip_el_buf");
        let clip_bic_buf = BufferProxy::new(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            "clip_bic_buf",
        );
        if wg_counts.clip_reduce.0 > 0 {
            recording.dispatch(
                cx.shaders.clip_reduce,
                wg_counts.clip_reduce,
                (clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf),
            );
        }
        let clip_bbox_buf = BufferProxy::new(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            "clip_bbox_buf",
        );
        if wg_counts.clip_leaf.0 > 0 {
            recording.dispatch(
                cx.shaders.clip_leaf,
                wg_counts.clip_leaf,
                (
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ),
            );
        }
        recording.free_resource(clip_inp_buf.into());
        recording.free_resource(clip_bic_buf.into());
        recording.free_resource(clip_el_buf.into());
        let draw_bbox_buf = BufferProxy::new(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            "draw_bbox_buf",
        );
        let bin_header_buf = BufferProxy::new(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            "bin_header_buf",
        );
        recording.dispatch(
            cx.shaders.binning,
            wg_counts.binning,
            (
                config_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ),
        );
        recording.free_resource(draw_monoid_buf.into());
        recording.free_resource(path_bbox_buf.into());
        recording.free_resource(clip_bbox_buf.into());
        // Note: this only needs to be rounded up because of the workaround to store the tile_offset
        // in storage rather than workgroup memory.
        let path_buf =
            ResourceProxy::new_buf(buffer_sizes.paths.size_in_bytes().into(), "path_buf");
        recording.dispatch(
            cx.shaders.tile_alloc,
            wg_counts.tile_alloc,
            (
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ),
        );
        recording.free_resource(draw_bbox_buf.into());
        recording.free_resource(tagmonoid_buf.into());
        let indirect_count_buf = BufferProxy::new(
            buffer_sizes.indirect_count.size_in_bytes().into(),
            "indirect_count",
        );
        recording.dispatch(
            cx.shaders.path_count_setup,
            wg_counts.path_count_setup,
            (bump_buf, indirect_count_buf),
        );
        let seg_counts_buf = BufferProxy::new(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            "seg_counts_buf",
        );
        recording.dispatch_indirect(
            cx.shaders.path_count,
            indirect_count_buf,
            0,
            (
                config_buf,
                bump_buf,
                lines_buf,
                path_buf,
                tile_buf,
                seg_counts_buf,
            ),
        );
        recording.dispatch(
            cx.shaders.backdrop,
            wg_counts.backdrop,
            (config_buf, path_buf, tile_buf),
        );
        recording.dispatch(
            cx.shaders.coarse,
            wg_counts.coarse,
            (
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
            ),
        );
        recording.dispatch(
            cx.shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            (bump_buf, indirect_count_buf, ptcl_buf),
        );
        recording.dispatch_indirect(
            cx.shaders.path_tiling,
            indirect_count_buf,
            0,
            (
                bump_buf,
                seg_counts_buf,
                lines_buf,
                path_buf,
                tile_buf,
                segments_buf,
            ),
        );
        recording.free_buffer(indirect_count_buf);
        recording.free_resource(seg_counts_buf.into());
        recording.free_resource(lines_buf.into());
        recording.free_resource(scene_buf.into());
        recording.free_resource(draw_monoid_buf.into());
        recording.free_resource(bin_header_buf.into());
        recording.free_resource(path_buf);

        if cx.robust {
            recording.download(bump_buf);
        }
        recording.free_resource(bump_buf.into());

        let out_image = ImageProxy::new(cx.params.width, cx.params.height, ImageFormat::Rgba8);
        (
            recording,
            CoarseOutput {
                config_buf: cx.resources.import_buffer(config_buf),
                tile_buf: cx.resources.import_buffer(tile_buf),
                segments_buf: cx.resources.import_buffer(seg_counts_buf),
                ptcl_buf: cx.resources.import_buffer(ptcl_buf),
                gradient_image: cx.resources.import_image(gradient_image),
                info_bin_data_buf: cx.resources.import_buffer(info_bin_data_buf),
                image_atlas: cx.resources.import_image(image_atlas),
                out_image: cx.resources.import_image(out_image),
                fine_workgroup_size: wg_counts.fine,
            },
        )
    }
}
