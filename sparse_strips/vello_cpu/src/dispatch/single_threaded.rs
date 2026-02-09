// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{Fine, FineKernel};
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::layer_manager::LayerManager;
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use vello_common::clip::ClipContext;
use vello_common::coarse::{Cmd, LayerKind, MODE_CPU, Wide, WideTilesBbox};
use vello_common::color::palette::css::TRANSPARENT;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter_effects::Filter;
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PremulColor};
use vello_common::pixmap::Pixmap;
use vello_common::render_graph::{RenderGraph, RenderNodeKind};
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

/// Single-threaded implementation of the rendering dispatcher.
///
/// This dispatcher handles the entire rendering pipeline on a single thread,
/// including path rasterization, layer composition, and filter effects.
/// It maintains the coarse tile grid (`Wide`), strip generation for paths,
/// and the render graph for managing layer dependencies and filter effects.
#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    /// Coarse tile grid containing rendering commands for each wide tile.
    wide: Wide,
    /// Clip context for managing non-isolated clipping.
    clip_context: ClipContext,
    /// Generator for converting paths into coverage strips.
    strip_generator: StripGenerator,
    /// Storage for alpha coverage data from strip generation.
    strip_storage: StripStorage,
    /// SIMD level for fearless SIMD dispatch.
    level: Level,
    /// Counter for generating unique layer IDs.
    layer_id_next: u32,
    /// Dependency graph tracking layer relationships and filter effects.
    render_graph: RenderGraph,
}

impl SingleThreadedDispatcher {
    /// Creates a new single-threaded dispatcher for the given dimensions.
    ///
    /// # Arguments
    /// * `width` - Width of the rendering surface in pixels.
    /// * `height` - Height of the rendering surface in pixels.
    /// * `level` - SIMD level to use for rasterization.
    ///
    /// # Notes
    /// The root layer (`layer_id` 0) is created immediately and must be node 0
    /// in the render graph for proper rendering order.
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let strip_generator = StripGenerator::new(width, height, level);
        let clip_context = ClipContext::new();
        let strip_storage = StripStorage::default();
        let mut render_graph = RenderGraph::new();

        // Create root node (layer_id 0) as the first node (will be node 0).
        // This ensures the root layer is always rendered last in the execution order.
        let wtile_bbox = WideTilesBbox::new([0, 0, wide.width_tiles(), wide.height_tiles()]);
        let root_node = render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox,
        });
        assert_eq!(root_node, 0, "Root node must be node 0");

        Self {
            wide,
            clip_context,
            strip_generator,
            strip_storage,
            level,
            layer_id_next: 0,
            render_graph,
        }
    }

    /// Rasterizes the scene using f32 precision (high quality).
    ///
    /// This dispatches to the appropriate SIMD implementation based on the
    /// configured level, using f32 for intermediate calculations.
    #[cfg(feature = "f32_pipeline")]
    fn rasterize_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height, encoded_paints));
    }

    /// Rasterizes the scene using u8 precision (fast).
    ///
    /// This dispatches to the appropriate SIMD implementation based on the
    /// configured level, using u8 for intermediate calculations to maximize speed.
    #[cfg(feature = "u8_pipeline")]
    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height, encoded_paints));
    }

    /// Core rasterization dispatcher that chooses between simple and filter-aware paths.
    ///
    /// # Type Parameters
    /// * `S` - SIMD implementation to use.
    /// * `F` - Fine rasterization kernel (determines precision).
    ///
    /// If the scene contains filter effects, uses the filter-aware path which maintains
    /// intermediate layer buffers. Otherwise, uses the simpler direct rasterization path.
    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut layer_manager = LayerManager::new();

        if self.has_filters() {
            // Use filter-aware path that maintains layer buffers for filter effects.
            self.rasterize_with_filters::<S, F>(
                simd,
                buffer,
                width,
                height,
                encoded_paints,
                &mut layer_manager,
            );
        } else {
            // Use simple direct rasterization for scenes without filters.
            self.rasterize_simple::<S, F>(simd, buffer, width, height, encoded_paints);
        }
    }

    /// Rasterizes a scene with filter effects using dependency-ordered execution.
    ///
    /// This processes the render graph in topological order, ensuring that filtered
    /// layers are rendered into intermediate buffers before being composed. Each
    /// filter layer is rendered to its own pixmap, the filter is applied, and then
    /// the result is stored in the layer manager for use by dependent layers.
    ///
    /// # Render Graph Execution
    /// - `FilterLayer` nodes: Render to intermediate buffer, apply filter, store result.
    /// - `RootLayer` node: Final composition to output buffer.
    fn rasterize_with_filters<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
        layer_manager: &mut LayerManager,
    ) {
        let mut fine = Fine::<S, F>::new(simd);

        // Process nodes in dependency order (filtered layers before their consumers).
        for node_id in self.render_graph.execution_order() {
            let node = &self.render_graph.nodes[node_id];

            match &node.kind {
                RenderNodeKind::FilterLayer {
                    layer_id,
                    filter,
                    wtile_bbox,
                    transform,
                } => {
                    // Allocate intermediate buffer for this filtered layer.
                    let bbox_width = wtile_bbox.width_px();
                    let bbox_height = wtile_bbox.height_px();
                    let mut pixmap = Pixmap::new(bbox_width, bbox_height);
                    // TODO: Re-use this allocation by adding a .configure() or similar method
                    // to avoid allocating the internal Vec<Region> on every filtered layer.
                    let mut regions =
                        Regions::new(bbox_width, bbox_height, pixmap.data_as_u8_slice_mut());

                    // Render each tile in the layer's bounding box.
                    regions.update_regions(|region| {
                        // Convert region-local coords to global wtile coords.
                        let x = wtile_bbox.x0() + region.x;
                        let y = wtile_bbox.y0() + region.y;

                        self.process_layer_tile(
                            &mut fine,
                            x,
                            y,
                            *layer_id,
                            PremulColor::from_alpha_color(TRANSPARENT),
                            layer_manager,
                            encoded_paints,
                        );

                        debug_assert_eq!(
                            fine.blend_buf.len(),
                            1,
                            "blend buffer should contain exactly one layer after tile processing"
                        );

                        fine.pack(region);
                    });

                    // Apply the filter effect to the completed layer.
                    fine.filter_layer(&mut pixmap, filter, layer_manager, *transform);

                    // Save the filtered pixmap to disk for debugging.
                    // #[cfg(all(debug_assertions, feature = "std", feature = "png"))]
                    // save_filtered_layer_debug(&pixmap, *layer_id);

                    // Store the filtered result for use by dependent layers.
                    layer_manager.register_layer(*layer_id, *wtile_bbox, pixmap);
                }
                RenderNodeKind::RootLayer {
                    layer_id,
                    wtile_bbox: _,
                } => {
                    // Final composition directly to output buffer.
                    let mut regions = Regions::new(width, height, buffer);
                    regions.update_regions(|region| {
                        // Use the background color from the wide tile.
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

                        debug_assert_eq!(
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

    /// Processes all rendering commands for a single layer within a specific tile.
    ///
    /// This handles the complex logic of composing filtered layers by:
    /// 1. Running normal rendering commands in sequence.
    /// 2. When encountering a filtered layer reference, compositing its pre-rendered
    ///    content from the layer manager.
    /// 3. Skipping the filtered layer's internal commands (already rendered separately).
    ///
    /// # Arguments
    /// * `fine` - The fine rasterizer instance.
    /// * `x`, `y` - Wide tile coordinates.
    /// * `layer_id` - The layer being processed.
    /// * `clear_color` - Initial color for the tile.
    /// * `layer_manager` - Storage for filtered layer buffers.
    /// * `encoded_paints` - Paint definitions for the scene.
    fn process_layer_tile<S: Simd, F: FineKernel<S>>(
        &self,
        fine: &mut Fine<S, F>,
        x: u16,
        y: u16,
        layer_id: u32,
        clear_color: PremulColor,
        layer_manager: &mut LayerManager,
        encoded_paints: &[EncodedPaint],
    ) {
        let wtile = &self.wide.get(x, y);
        fine.set_coords(x, y);
        fine.clear(clear_color);

        // Process all commands in this layer's render range.
        // Invariant: tiles within a layer's bbox must have commands for that layer.
        let ranges = wtile.layer_cmd_ranges.get(&layer_id).unwrap();

        let mut cmd_idx = ranges.render_range.start;
        while cmd_idx < ranges.render_range.end {
            let cmd: &Cmd = &wtile.cmds[cmd_idx];

            fine.run_cmd(
                cmd,
                &self.strip_storage.alphas,
                encoded_paints,
                &self.wide.attrs,
            );

            // Special handling for filtered layer composition.
            // Filtered layers have already been rendered and stored in layer_manager.
            // Here we composite them into the current buffer, with special handling for clipping.
            if let Cmd::PushBuf(LayerKind::Filtered(child_layer_id)) = cmd {
                // Invariant: PushBuf(Filtered) command must have corresponding layer_cmd_ranges entry.
                let filtered_ranges = wtile.layer_cmd_ranges.get(child_layer_id).unwrap();

                // Check what comes after the filtered layer push to determine clipping state
                match wtile.cmds.get(cmd_idx + 1) {
                    // Zero-clip region: tile is completely outside the clip path.
                    // The layer was already rendered for filtering, but we skip compositing
                    // since this tile is entirely clipped out.
                    // (PushZeroClip only appears for clipped filter layers)
                    Some(Cmd::PushZeroClip(id)) if *id == *child_layer_id => {
                        cmd_idx += 1; // Skip the PushZeroClip command
                    }

                    // Partial clip: push the clip buffer, then composite the filtered layer
                    Some(Cmd::PushBuf(LayerKind::Clip(_))) => {
                        fine.run_cmd(
                            &wtile.cmds[cmd_idx + 1],
                            &self.strip_storage.alphas,
                            encoded_paints,
                            &self.wide.attrs,
                        );
                        cmd_idx += 1;

                        if let Some(mut region) =
                            layer_manager.layer_tile_region_mut(*child_layer_id, x, y)
                        {
                            fine.unpack(&mut region);
                        }
                    }

                    // No clip or fully inside clip: composite the filtered layer directly
                    _ => {
                        if let Some(mut region) =
                            layer_manager.layer_tile_region_mut(*child_layer_id, x, y)
                        {
                            fine.unpack(&mut region);
                        }
                    }
                }

                // Skip past the filtered layer's internal commands, as they were already
                // rendered when the FilterLayer node was processed earlier.
                cmd_idx = filtered_ranges.render_range.end.max(cmd_idx + 1);
            } else {
                cmd_idx += 1;
            }
        }
    }

    /// Simple rasterization path for scenes without filter effects.
    ///
    /// This directly processes each tile's commands without maintaining intermediate
    /// layer buffers. All rendering happens in a single pass directly to the output buffer.
    /// This is more efficient than the filter-aware path when no filters are present.
    fn rasterize_simple<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut regions = Regions::new(width, height, buffer);
        let mut fine = Fine::<S, F>::new(simd);

        regions.update_regions(|region| {
            let x = region.x;
            let y = region.y;

            let wtile = self.wide.get(x, y);
            fine.set_coords(x, y);

            // Clear to background and process all commands in order.
            fine.clear(wtile.bg);
            for cmd in &wtile.cmds {
                fine.run_cmd(
                    cmd,
                    &self.strip_storage.alphas,
                    encoded_paints,
                    &self.wide.attrs,
                );
            }

            fine.pack(region);
        });
    }

    /// Returns true if the scene contains any filter effects.
    fn has_filters(&self) -> bool {
        self.render_graph.has_filters()
    }

    /// Composites at an offset using f32 precision (high quality).
    #[cfg(feature = "f32_pipeline")]
    fn composite_at_offset_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        use crate::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.composite_at_offset_with::<_, F32Kernel>(
            simd, buffer, width, height, dst_x, dst_y, dst_buffer_width, dst_buffer_height, encoded_paints
        ));
    }

    /// Composites at an offset using u8 precision (fast).
    #[cfg(feature = "u8_pipeline")]
    fn composite_at_offset_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        use crate::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.composite_at_offset_with::<_, U8Kernel>(
            simd, buffer, width, height, dst_x, dst_y, dst_buffer_width, dst_buffer_height, encoded_paints
        ));
    }

    /// Core implementation for compositing at an offset.
    ///
    /// Composites tiles sequentially, writing directly to the destination buffer
    /// at the specified offset.
    fn composite_at_offset_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut regions = Regions::new_at_offset(
            width,
            height,
            dst_x,
            dst_y,
            dst_buffer_width,
            dst_buffer_height,
            buffer,
        );
        let mut fine = Fine::<S, F>::new(simd);

        regions.update_regions(|region| {
            let x = region.x;
            let y = region.y;

            let wtile = self.wide.get(x, y);
            fine.set_coords(x, y);

            // Unpack existing pixel data from the region instead of clearing,
            // so that rendering composites onto the existing pixmap contents.
            fine.unpack(region);
            for cmd in &wtile.cmds {
                fine.run_cmd(
                    cmd,
                    &self.strip_storage.alphas,
                    encoded_paints,
                    &self.wide.attrs,
                );
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
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        let wide = &mut self.wide;

        // Convert path to coverage strips.
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            self.clip_context.get(),
        );

        // Generate coarse-level commands from strips (layer_id 0 = root layer).
        wide.generate(
            &self.strip_storage.strips,
            paint,
            blend_mode,
            0,
            mask,
            encoded_paints,
        );
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        let wide = &mut self.wide;

        // Convert stroked path to coverage strips.
        self.strip_generator.generate_stroked_path(
            path,
            stroke,
            transform,
            aliasing_threshold,
            &mut self.strip_storage,
            self.clip_context.get(),
        );

        // Generate coarse-level commands from strips (layer_id 0 = root layer).
        wide.generate(
            &self.strip_storage.strips,
            paint,
            blend_mode,
            0,
            mask,
            encoded_paints,
        );
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        // Allocate a new unique layer ID.
        self.layer_id_next += 1;

        // Generate clip coverage if a clip path is provided.
        let clip = if let Some(c) = clip_path {
            self.strip_generator.generate_filled_path(
                c,
                fill_rule,
                transform,
                aliasing_threshold,
                &mut self.strip_storage,
                self.clip_context.get(),
            );

            Some(self.strip_storage.strips.as_slice())
        } else {
            None
        };

        // Push the layer onto the coarse tile stack and update render graph.
        self.wide.push_layer(
            self.layer_id_next,
            clip,
            blend_mode,
            mask,
            opacity,
            filter,
            transform,
            &mut self.render_graph,
            0,
        );
    }

    fn pop_layer(&mut self) {
        // Pop the current layer and update render graph.
        self.wide.pop_layer(&mut self.render_graph);
    }

    fn reset(&mut self) {
        // Clear all rendering state to prepare for a new scene.
        self.wide.reset();
        self.clip_context.reset();
        self.strip_generator.reset();
        self.strip_storage.clear();
        self.render_graph.clear();
        self.layer_id_next = 0;

        // Recreate root node as node 0 (required for proper execution order).
        let root_node = self.render_graph.add_node(RenderNodeKind::RootLayer {
            layer_id: 0,
            wtile_bbox: WideTilesBbox::new([
                0,
                0,
                self.wide.width_tiles(),
                self.wide.height_tiles(),
            ]),
        });
        debug_assert_eq!(root_node, 0, "Root node must be node 0");

        // Reset layer ID counter.
        self.layer_id_next = 0;
    }

    fn flush(&mut self, _encoded_paints: &[EncodedPaint]) {
        // No-op for single-threaded dispatcher (no work queue to flush).
    }

    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        // If only the u8 pipeline is enabled, then use it
        #[cfg(all(feature = "u8_pipeline", not(feature = "f32_pipeline")))]
        {
            let _ = render_mode;
            self.rasterize_u8(buffer, width, height, encoded_paints);
        }

        // If only the f32 pipeline is enabled, then use it
        #[cfg(all(feature = "f32_pipeline", not(feature = "u8_pipeline")))]
        {
            let _ = render_mode;
            self.rasterize_f32(buffer, width, height, encoded_paints);
        }

        // If both pipelines are enabled, select precision based on render mode parameter.
        #[cfg(all(feature = "u8_pipeline", feature = "f32_pipeline"))]
        match render_mode {
            RenderMode::OptimizeSpeed => {
                // Use u8 precision for faster rendering.
                self.rasterize_u8(buffer, width, height, encoded_paints);
            }
            RenderMode::OptimizeQuality => {
                // Use f32 precision for higher quality.
                self.rasterize_f32(buffer, width, height, encoded_paints);
            }
        }

        #[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
        {
            // This case never gets hit because there is a compile_error in the root.
            // But have this code disables some warnings and makes the compile error easier to read
            let _ = (buffer, render_mode, width, height, encoded_paints);
        }
    }

    fn composite_at_offset(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        dst_x: u16,
        dst_y: u16,
        dst_buffer_width: u16,
        dst_buffer_height: u16,
        render_mode: RenderMode,
        encoded_paints: &[EncodedPaint],
    ) {
        #[cfg(all(feature = "u8_pipeline", not(feature = "f32_pipeline")))]
        {
            let _ = render_mode;
            self.composite_at_offset_u8(
                buffer,
                width,
                height,
                dst_x,
                dst_y,
                dst_buffer_width,
                dst_buffer_height,
                encoded_paints,
            );
        }

        #[cfg(all(feature = "f32_pipeline", not(feature = "u8_pipeline")))]
        {
            let _ = render_mode;
            self.composite_at_offset_f32(
                buffer,
                width,
                height,
                dst_x,
                dst_y,
                dst_buffer_width,
                dst_buffer_height,
                encoded_paints,
            );
        }

        #[cfg(all(feature = "u8_pipeline", feature = "f32_pipeline"))]
        match render_mode {
            RenderMode::OptimizeSpeed => {
                self.composite_at_offset_u8(
                    buffer,
                    width,
                    height,
                    dst_x,
                    dst_y,
                    dst_buffer_width,
                    dst_buffer_height,
                    encoded_paints,
                );
            }
            RenderMode::OptimizeQuality => {
                self.composite_at_offset_f32(
                    buffer,
                    width,
                    height,
                    dst_x,
                    dst_y,
                    dst_buffer_width,
                    dst_buffer_height,
                    encoded_paints,
                );
            }
        }

        #[cfg(all(not(feature = "u8_pipeline"), not(feature = "f32_pipeline")))]
        {
            let _ = (
                buffer,
                width,
                height,
                dst_x,
                dst_y,
                dst_buffer_width,
                dst_buffer_height,
                render_mode,
                encoded_paints,
            );
        }
    }

    fn generate_wide_cmd(
        &mut self,
        strip_buf: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        encoded_paints: &[EncodedPaint],
    ) {
        // Generate coarse-level commands from pre-computed strips (layer_id 0 = root layer).
        self.wide
            .generate(strip_buf, paint, blend_mode, 0, None, encoded_paints);
    }

    fn strip_storage_mut(&mut self) -> &mut StripStorage {
        &mut self.strip_storage
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            fill_rule,
            transform,
            aliasing_threshold,
        );
    }

    fn pop_clip_path(&mut self) {
        self.clip_context.pop_clip();
    }
}

/// Saves a filtered pixmap to disk for debugging purposes.
/// Only available in debug builds with `std` and `png` features enabled.
#[allow(
    dead_code,
    reason = "useful debug utility, can be enabled by uncommenting the call site"
)]
#[cfg(all(debug_assertions, feature = "std", feature = "png"))]
fn save_filtered_layer_debug(pixmap: &Pixmap, layer_id: u32) {
    use std::path::PathBuf;

    let diffs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_sparse_tests/diffs");
    let _ = std::fs::create_dir_all(&diffs_path);
    let filename = diffs_path.join(alloc::format!("filtered_layer_{}.png", layer_id));

    if let Ok(png_data) = pixmap.clone().into_png() {
        let _ = std::fs::write(&filename, &png_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kurbo::Rect;
    use vello_common::color::palette::css::BLUE;
    use vello_common::kurbo::Shape;
    use vello_common::paint::PremulColor;

    /// Verifies that `reset()` properly clears all internal buffers and state.
    ///
    /// This is important to ensure that a dispatcher can be reused for multiple
    /// rendering passes without accumulating stale data from previous frames.
    #[test]
    fn buffers_cleared_on_reset() {
        let mut dispatcher = SingleThreadedDispatcher::new(100, 100, Level::new());

        // Render a simple shape to populate internal buffers.
        dispatcher.fill_path(
            &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            BlendMode::default(),
            None,
            None,
            &[],
        );

        // Ensure there is data to clear.
        assert!(!dispatcher.strip_storage.alphas.is_empty());
        assert!(!dispatcher.wide.get(0, 0).cmds.is_empty());

        dispatcher.reset();

        // Verify all buffers are cleared.
        assert!(dispatcher.strip_storage.alphas.is_empty());
        assert!(dispatcher.wide.get(0, 0).cmds.is_empty());
        assert_eq!(dispatcher.layer_id_next, 0);
    }
}
