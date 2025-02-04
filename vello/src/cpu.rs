use vello_encoding::{
    BinHeader, BufferSize, BumpAllocators, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox,
    DrawMonoid, Encoding, IndirectCount, LineSoup, Path, PathBbox, PathMonoid, PathSegment,
    RenderConfig, Resolver, SegmentCount, Tile,
};
use vello_shaders::cpu::{
    backdrop_main, bbox_clear_main, binning_main, clip_leaf_main, clip_reduce_main, coarse_main,
    draw_leaf_main, draw_reduce_main, flatten_main, path_count_main, path_count_setup_main,
    path_tiling_main, path_tiling_setup, path_tiling_setup_main, pathtag_reduce_main,
    pathtag_scan_main, tile_alloc_main,
};

use crate::RenderParams;

#[derive(Default)]
pub struct Buffer<T: bytemuck::Zeroable> {
    inner: Vec<T>,
}

impl<T: bytemuck::Zeroable> Buffer<T> {
    fn to_fit(&mut self, size: BufferSize<T>) -> &mut [T] {
        self.inner
            .resize_with(size.len().try_into().expect("32 bit platform"), || {
                T::zeroed()
            });
        &mut self.inner
    }
    fn to_fit_zeroed(&mut self, size: BufferSize<T>) -> &mut [T] {
        self.inner.clear();
        self.to_fit(size)
    }
}

#[derive(Default)]
pub struct CoarseBuffers {
    path_reduced: Buffer<PathMonoid>,
    path_reduced2: Buffer<PathMonoid>,
    path_reduced_scan: Buffer<PathMonoid>,
    path_monoids: Buffer<PathMonoid>,
    path_bboxes: Buffer<PathBbox>,
    draw_reduced: Buffer<DrawMonoid>,
    draw_monoids: Buffer<DrawMonoid>,
    info: Buffer<u32>,
    clip_inps: Buffer<Clip>,
    clip_els: Buffer<ClipElement>,
    clip_bics: Buffer<ClipBic>,
    clip_bboxes: Buffer<ClipBbox>,
    draw_bboxes: Buffer<DrawBbox>,
    bump_alloc: BumpAllocators,
    indirect_count: Buffer<IndirectCount>,
    bin_headers: Buffer<BinHeader>,
    paths: Buffer<Path>,
    // Bump allocated buffers
    lines: Buffer<LineSoup>,
    bin_data: Buffer<u32>,
    tiles: Buffer<Tile>,
    seg_counts: Buffer<SegmentCount>,
    segments: Buffer<PathSegment>,
    blend_spill: Buffer<u32>,
    ptcl: Buffer<u32>,
}

pub fn run_coarse_cpu(
    encoding: &Encoding,
    resolver: &mut Resolver,
    params: &RenderParams,
    buffers: &mut CoarseBuffers,
) {
    let mut packed = vec![];

    let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
    let cpu_config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);
    // HACK: The coarse workgroup counts is the number of active bins.
    if (cpu_config.workgroup_counts.coarse.0
        * cpu_config.workgroup_counts.coarse.1
        * cpu_config.workgroup_counts.coarse.2)
        > 256
    {
        log::warn!(
            "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
            params.width,
            params.height
        );
    }
    let buffer_sizes = &cpu_config.buffer_sizes;
    let wg_counts = &cpu_config.workgroup_counts;

    // TODO: This is an alignment hazard, which just happens to work on mainstream platforms
    // Maybe don't merge as-is?
    let scene_buf = bytemuck::cast_slice(&packed);
    let config_buf = cpu_config.gpu;
    let info_bin_data_buf = buffers.bin_data.to_fit(buffer_sizes.bin_data);
    let tile_buf = buffers.tiles.to_fit(buffer_sizes.tiles);
    let segments_buf = buffers.segments.to_fit(buffer_sizes.segments);

    let ptcl_buf = buffers.ptcl.to_fit(buffer_sizes.ptcl);
    let reduced_buf = buffers.path_reduced.to_fit(buffer_sizes.path_reduced);

    pathtag_reduce_main(wg_counts.path_reduce.0, &config_buf, scene_buf, reduced_buf);

    let tagmonoid_buf = buffers.path_monoids.to_fit(buffer_sizes.path_monoids);

    pathtag_scan_main(
        wg_counts.path_scan.0,
        &config_buf,
        scene_buf,
        reduced_buf,
        tagmonoid_buf,
    );

    // Could re-use `reduced_buf` from this point

    let path_bbox_buf = buffers.path_bboxes.to_fit(buffer_sizes.path_bboxes);

    bbox_clear_main(&config_buf, path_bbox_buf);
    let bump_buf = &mut buffers.bump_alloc;
    let lines_buf = buffers.lines.to_fit(buffer_sizes.lines);
    flatten_main(
        wg_counts.flatten.0,
        &config_buf,
        scene_buf,
        tagmonoid_buf,
        path_bbox_buf,
        bump_buf,
        lines_buf,
    );

    let draw_reduced_buf = buffers.draw_reduced.to_fit(buffer_sizes.draw_reduced);

    draw_reduce_main(
        wg_counts.draw_reduce.0,
        &config_buf,
        scene_buf,
        draw_reduced_buf,
    );

    let draw_monoid_buf = buffers.draw_monoids.to_fit(buffer_sizes.draw_monoids);
    let clip_inp_buf = buffers.clip_inps.to_fit(buffer_sizes.clip_inps);
    draw_leaf_main(
        wg_counts.draw_leaf.0,
        &config_buf,
        scene_buf,
        draw_reduced_buf,
        path_bbox_buf,
        draw_monoid_buf,
        info_bin_data_buf,
        clip_inp_buf,
    );

    // Could re-use `draw_reduced_buf` from this point

    let clip_el_buf = buffers.clip_els.to_fit(buffer_sizes.clip_els);

    let clip_bic_buf = buffers.clip_bics.to_fit(buffer_sizes.clip_bics);

    if wg_counts.clip_reduce.0 > 0 {
        clip_reduce_main(
            wg_counts.clip_reduce.0,
            clip_inp_buf,
            path_bbox_buf,
            clip_bic_buf,
            clip_el_buf,
        );
    }
    let clip_bbox_buf = buffers.clip_bboxes.to_fit(buffer_sizes.clip_bboxes);

    if wg_counts.clip_leaf.0 > 0 {
        clip_leaf_main(
            &config_buf,
            clip_inp_buf,
            path_bbox_buf,
            draw_monoid_buf,
            clip_bbox_buf,
        );
    }

    // Could re-use `clip_inp_buf`, `clip_bic_buf`, and `clip_el_buf` from this point

    let draw_bbox_buf = buffers.draw_bboxes.to_fit(buffer_sizes.draw_bboxes);

    let bin_header_buf = buffers.bin_headers.to_fit(buffer_sizes.bin_headers);

    binning_main(
        wg_counts.binning.0,
        &config_buf,
        draw_monoid_buf,
        path_bbox_buf,
        clip_bbox_buf,
        draw_bbox_buf,
        bump_buf,
        info_bin_data_buf,
        bin_header_buf,
    );

    // Could re-use `draw_monoid_buf` and `clip_bbox_buf` from this point

    // TODO: What does this comment mean?
    // Note: this only needs to be rounded up because of the workaround to store the tile_offset
    // in storage rather than workgroup memory.
    let path_buf = buffers.paths.to_fit(buffer_sizes.paths);
    tile_alloc_main(
        &config_buf,
        scene_buf,
        draw_bbox_buf,
        bump_buf,
        path_buf,
        tile_buf,
    );

    // Could re-use `draw_bbox_buf` and `tagmonoid_buf` from this point

    let mut indirect_count_buf = IndirectCount::default();

    path_count_setup_main(bump_buf, &mut indirect_count_buf);

    let seg_counts_buf = buffers.seg_counts.to_fit(buffer_sizes.seg_counts);
    path_count_main(bump_buf, lines_buf, path_buf, tile_buf, seg_counts_buf);

    backdrop_main(&config_buf, bump_buf, path_buf, tile_buf);

    coarse_main(
        &config_buf,
        scene_buf,
        draw_monoid_buf,
        bin_header_buf,
        info_bin_data_buf,
        path_buf,
        tile_buf,
        bump_buf,
        ptcl_buf,
    );

    path_tiling_setup_main(
        bump_buf,
        &mut indirect_count_buf, /* ptcl_buf (for forwarding errors to fine)*/
    );

    path_tiling_main(
        bump_buf,
        seg_counts_buf,
        lines_buf,
        path_buf,
        tile_buf,
        segments_buf,
    );
}
