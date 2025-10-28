// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::layer_manager::LayerManager;
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use vello_common::coarse::{Bbox, Cmd, LayerKind, MODE_CPU, Wide};
use vello_common::color::palette::css::TRANSPARENT;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::filter_effects::Filter;
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PremulColor};
use vello_common::render_graph::{RenderGraph, RenderNodeKind};
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    wide: Wide,
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
    layer_id_next: u32,
    render_graph: RenderGraph,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);
        let strip_storage = StripStorage::default();
        let mut render_graph = RenderGraph::new();

        // Create root node (layer_id 0) as the first node (will be node 0)
        let wtile_bbox = Bbox::new([0, 0, wide.width_tiles(), wide.height_tiles()]);
        let root_node = render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox,
        });
        assert_eq!(root_node, 0, "Root node must be node 0");

        Self {
            wide,
            strip_generator,
            strip_storage,
            level,
            layer_id_next: 0,
            render_graph,
        }
    }

    fn rasterize_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height, encoded_paints));
    }

    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height, encoded_paints));
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut layer_manager = LayerManager::<F::Numeric>::new();

        // if self.has_filters() {
        self.rasterize_with_filters::<S, F>(
            simd,
            buffer,
            width,
            height,
            encoded_paints,
            &mut layer_manager,
        );
        // } else {
        //     self.rasterize_simple::<S, F>(
        //         simd,
        //         buffer,
        //         width,
        //         height,
        //         encoded_paints,
        //     );
        // }
    }

    fn rasterize_with_filters<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        layer_manager: &mut LayerManager<F::Numeric>,
    ) {
        let mut regions = Regions::new(width, height, buffer);
        let mut fine = Fine::<S, F>::new(simd);

        // Get topologically sorted nodes
        let sorted_nodes = self.render_graph.topological_sort();
        for node_id in sorted_nodes {
            let node = &self.render_graph.nodes[node_id];

            match &node.kind {
                RenderNodeKind::FilterLayer {
                    layer_id,
                    filter,
                    wtile_bbox,
                } => {
                    layer_manager.allocate_layer_with_id(*layer_id, *wtile_bbox);

                    // Render geometry commands for this layer - iterate only over tiles in bbox
                    for y in wtile_bbox.y0()..wtile_bbox.y1() {
                        for x in wtile_bbox.x0()..wtile_bbox.x1() {
                            self.process_layer_tile(
                                &mut fine,
                                x,
                                y,
                                *layer_id,
                                PremulColor::from_alpha_color(TRANSPARENT),
                                layer_manager,
                                encoded_paints,
                            );

                            assert_eq!(
                                fine.blend_buf.len(),
                                1,
                                "blend buffer should contain exactly one layer after tile processing"
                            );

                            if let Some(layer_tile) =
                                layer_manager.get_layer_tile_mut(*layer_id, x, y)
                            {
                                fine.pack_into_layer(layer_tile);
                            }
                        }
                    }

                    // Apply filter to the entire layer
                    if let Some(layer) = layer_manager.get_layer_mut(*layer_id) {
                        fine.apply_filter_to_layer(layer, *wtile_bbox, filter);
                    }
                }
                RenderNodeKind::RootLayer {
                    layer_id,
                    wtile_bbox,
                } => {
                    regions.update_regions(|region| {
                        let bg = self.wide.get(region.x, region.y).bg;
                        self.process_layer_tile(
                            &mut fine,
                            region.x,
                            region.y,
                            *layer_id,
                            bg,
                            layer_manager,
                            encoded_paints,
                        );

                        assert_eq!(
                            fine.blend_buf.len(),
                            1,
                            "blend buffer should contain exactly one layer after tile processing"
                        );

                        fine.pack(region);
                    });
                }
            }
        }
    }

    fn process_layer_tile<S: Simd, F: FineKernel<S>>(
        &self,
        fine: &mut Fine<S, F>,
        x: u16,
        y: u16,
        layer_id: u32,
        clear_color: PremulColor,
        layer_manager: &mut LayerManager<F::Numeric>,
        encoded_paints: &[EncodedPaint],
    ) {
        let wtile = self.wide.get(x, y);
        fine.set_coords(x, y);

        fine.clear(clear_color);

        if let Some(ranges) = wtile.layer_cmd_ranges.get(&layer_id) {
            let mut cmd_idx = ranges.render_range.start;
            while cmd_idx < ranges.render_range.end {
                let cmd = &wtile.cmds[cmd_idx];
                fine.run_cmd(cmd, &self.strip_storage.alphas, encoded_paints);

                if let Cmd::PushBuf(LayerKind::Filtered(filtered_layer_id)) = cmd {
                    let next_cmd = &wtile.cmds[cmd_idx + 1];
                    if let Cmd::PushBuf(LayerKind::Clip(_)) = next_cmd {
                        fine.run_cmd(next_cmd, &self.strip_storage.alphas, encoded_paints);
                        cmd_idx += 1;
                    }

                    if let Some(layer_tile) =
                        layer_manager.get_layer_tile_mut(*filtered_layer_id, x, y)
                    {
                        fine.copy_from_layer_buffer(layer_tile);
                    }
                    if let Some(filtered_ranges) = wtile.layer_cmd_ranges.get(filtered_layer_id) {
                        cmd_idx = filtered_ranges.render_range.end.max(cmd_idx + 1);
                    }
                } else {
                    cmd_idx += 1;
                }
            }
        }
    }

    fn rasterize_simple<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut buffer = Regions::new(width, height, buffer);
        let mut fine = Fine::<S, F>::new(simd);

        buffer.update_regions(|region| {
            let x = region.x;
            let y = region.y;

            let wtile = self.wide.get(x, y);
            fine.set_coords(x, y);

            fine.clear(wtile.bg);
            for cmd in &wtile.cmds {
                fine.run_cmd(cmd, &self.strip_storage.alphas, encoded_paints);
            }

            fine.pack(region);
        });
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn wide(&self) -> &Wide {
        &self.wide
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;

        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
        );

        wide.generate(&self.strip_storage.strips, paint, 0);
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let wide = &mut self.wide;

        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
        );

        wide.generate(&self.strip_storage.strips, paint, 0);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.layer_id_next += 1;

        let clip = if let Some(c) = clip_path {
            self.strip_generator.generate_filled_path(
                c,
                fill_rule,
                clip_transform,
                aliasing_threshold,
                &mut self.strip_storage,
            );

            Some(self.strip_storage.strips.as_slice())
        } else {
            None
        };

        self.wide.push_layer(
            self.layer_id_next,
            clip,
            blend_mode,
            mask,
            opacity,
            filter,
            &mut self.render_graph,
            0,
        );
    }

    fn pop_layer(&mut self) {
        self.wide.pop_layer(&mut self.render_graph);
    }

    fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator.reset();
        self.strip_storage.clear();
        self.render_graph = RenderGraph::new();

        // Recreate root node as node 0
        let root_node = self.render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox: Bbox::new([0, 0, self.wide.width_tiles(), self.wide.height_tiles()]),
        });
        assert_eq!(root_node, 0, "Root node must be node 0");

        self.layer_id_next = 0;
    }

    fn flush(&mut self) {}

    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        match render_mode {
            RenderMode::OptimizeSpeed => {
                self.rasterize_u8(buffer, width, height, encoded_paints);
            }
            RenderMode::OptimizeQuality => {
                self.rasterize_f32(buffer, width, height, encoded_paints);
            }
        }
    }

    fn generate_wide_cmd(&mut self, strip_buf: &[Strip], paint: Paint) {
        self.wide.generate(strip_buf, paint, 0);
    }

    fn strip_storage_mut(&mut self) -> &mut StripStorage {
        &mut self.strip_storage
    }

    fn has_filters(&self) -> bool {
        self.render_graph.has_filters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Rect;
    use vello_common::color::palette::css::BLUE;
    use vello_common::kurbo::Shape;
    use vello_common::paint::PremulColor;

    #[test]
    fn buffers_cleared_on_reset() {
        let mut dispatcher = SingleThreadedDispatcher::new(100, 100, Level::new());

        dispatcher.fill_path(
            &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            None,
        );

        // Ensure there is data to clear.
        assert!(!dispatcher.strip_storage.alphas.is_empty());
        assert!(!dispatcher.wide.get(0, 0).cmds.is_empty());

        dispatcher.reset();

        // Verify buffers are cleared.
        assert!(dispatcher.strip_storage.alphas.is_empty());
        assert!(dispatcher.wide.get(0, 0).cmds.is_empty());
    }
}
