// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use crate::color::palette::css::TRANSPARENT;
use crate::encode::EncodedPaint;
use crate::filter_effects::Filter;
use crate::kurbo::{Affine, Rect};
use crate::mask::Mask;
use crate::paint::{Paint, PremulColor};
use crate::peniko::{BlendMode, Compose, Mix};
use crate::render_graph::{DependencyKind, LayerId, RenderGraph, RenderNodeKind};
use crate::{strip::Strip, tile::Tile};
use alloc::vec;
use alloc::{boxed::Box, vec::Vec};
#[cfg(debug_assertions)]
use alloc::{format, string::String};
use core::ops::Range;
use hashbrown::HashMap;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

#[derive(Debug)]
struct Layer {
    /// The layer's ID.
    layer_id: LayerId,
    /// Whether the layer has a clip associated with it.
    clip: bool,
    /// The blend mode with which this layer should be blended into
    /// the previous layer.
    blend_mode: BlendMode,
    /// An opacity to apply to the whole layer before blending it
    /// into the backdrop.
    opacity: f32,
    /// A mask to apply to the layer before blending it back into
    /// the backdrop.
    mask: Option<Mask>,
    /// A filter effect to apply to the layer before other operations.
    filter: Option<Filter>,
    /// Bounding box of wide tiles containing geometry.
    /// Starts with inverted bounds, shrinks to actual content during drawing.
    wtile_bbox: WideTilesBbox,
}

impl Layer {
    /// Whether the layer actually requires allocating a new scratch buffer
    /// for drawing its contents.
    fn needs_buf(&self) -> bool {
        self.blend_mode.mix != Mix::Normal
            || self.blend_mode.compose != Compose::SrcOver
            || self.opacity != 1.0
            || self.mask.is_some()
            || self.filter.is_some()
            || !self.clip
    }
}

/// `MODE_CPU` allows compile time optimizations to be applied to wide tile draw command generation
/// specific to `vello_cpu`.
pub const MODE_CPU: u8 = 0;
/// `MODE_HYBRID` allows compile time optimizations to be applied to wide tile draw command
/// generation specific for `vello_hybrid`.
pub const MODE_HYBRID: u8 = 1;

/// A container for wide tiles.
#[derive(Debug)]
pub struct Wide<const MODE: u8 = MODE_CPU> {
    /// The width of the container.
    pub width: u16,
    /// The height of the container.
    pub height: u16,
    /// The wide tiles in the container.
    pub tiles: Vec<WideTile<MODE>>,
    /// Shared command properties, referenced by index from fill and clip commands.
    pub attrs: CommandAttrs,
    /// The stack of layers.
    layer_stack: Vec<Layer>,
    /// The stack of active clip regions.
    clip_stack: Vec<Clip>,
    /// Stack of filter layer node IDs for render graph dependency tracking.
    /// Initialized with node 0 (the root node representing the final output).
    /// As layers with filters are pushed, their node IDs are added to this stack.
    filter_node_stack: Vec<usize>,
    /// Count of nested filtered layers with clip paths.
    /// When > 0, command generation uses full viewport bounds instead of clip bounds
    /// to ensure filter effects can process the full layer before applying the clip.
    clipped_filter_layer_depth: u32,
}

/// A clip region.
#[derive(Debug)]
struct Clip {
    /// The intersected bounding box after clip
    pub clip_bbox: WideTilesBbox,
    /// The rendered path in sparse strip representation
    pub strips: Box<[Strip]>,
    /// The index of the thread that owns the alpha buffer.
    /// Always 0 in single-threaded mode.
    pub thread_idx: u8,
}

/// An axis-aligned bounding box represented by top-left and bottom-right corners.
///
/// The coordinates are stored as `[x0, y0, x1, y1]` in wide tile coordinates,
/// where `(x0, y0)` is the top-left corner and `(x1, y1)` is the bottom-right corner.
#[derive(Debug, Clone, Copy)]
pub struct WideTilesBbox {
    /// The bounding box coordinates.
    pub bbox: [u16; 4],
}

impl WideTilesBbox {
    /// Create a new bounding box.
    pub fn new(bbox: [u16; 4]) -> Self {
        Self { bbox }
    }

    /// Get the x0 coordinate of the bounding box.
    #[inline(always)]
    pub fn x0(&self) -> u16 {
        self.bbox[0]
    }

    /// Get the y0 coordinate of the bounding box.
    #[inline(always)]
    pub fn y0(&self) -> u16 {
        self.bbox[1]
    }

    /// Get the x1 coordinate of the bounding box.
    #[inline(always)]
    pub fn x1(&self) -> u16 {
        self.bbox[2]
    }

    /// Get the y1 coordinate of the bounding box.
    #[inline(always)]
    pub fn y1(&self) -> u16 {
        self.bbox[3]
    }

    /// Get the width of the bounding box (x1 - x0).
    #[inline(always)]
    pub fn width_tiles(&self) -> u16 {
        self.x1().saturating_sub(self.x0())
    }

    /// Get the width of the bounding box in pixels.
    #[inline(always)]
    pub fn width_px(&self) -> u16 {
        self.width_tiles() * WideTile::WIDTH
    }

    /// Get the height of the bounding box (y1 - y0).
    #[inline(always)]
    pub fn height_tiles(&self) -> u16 {
        self.y1().saturating_sub(self.y0())
    }

    /// Get the height of the bounding box in pixels.
    #[inline(always)]
    pub fn height_px(&self) -> u16 {
        self.height_tiles() * Tile::HEIGHT
    }

    /// Check if a point (x, y) is contained within this bounding box.
    ///
    /// Returns `true` if x0 <= x < x1 and y0 <= y < y1.
    #[inline(always)]
    pub fn contains(&self, x: u16, y: u16) -> bool {
        let [x0, y0, x1, y1] = self.bbox;
        (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    }

    /// Create an empty bounding box (zero area).
    #[inline(always)]
    pub(crate) fn empty() -> Self {
        Self::new([0, 0, 0, 0])
    }

    /// Calculate the intersection of this bounding box with another.
    #[inline(always)]
    pub(crate) fn intersect(self, other: Self) -> Self {
        Self::new([
            self.x0().max(other.x0()),
            self.y0().max(other.y0()),
            self.x1().min(other.x1()),
            self.y1().min(other.y1()),
        ])
    }

    /// Update this bounding box to include another bounding box (union in place).
    #[inline(always)]
    pub(crate) fn union(&mut self, other: Self) {
        if !other.is_inverted() {
            if self.is_inverted() {
                // If self is empty, just copy other
                self.bbox = other.bbox;
            } else {
                // Otherwise compute the union
                self.bbox[0] = self.bbox[0].min(other.x0());
                self.bbox[1] = self.bbox[1].min(other.y0());
                self.bbox[2] = self.bbox[2].max(other.x1());
                self.bbox[3] = self.bbox[3].max(other.y1());
            }
        }
    }

    /// Create an inverted bounding box for incremental updates.
    #[inline(always)]
    pub(crate) fn inverted() -> Self {
        Self::new([u16::MAX, u16::MAX, 0, 0])
    }

    /// Check if the bbox is still in its inverted state (no updates yet).
    #[inline(always)]
    pub(crate) fn is_inverted(self) -> bool {
        self.bbox[0] == u16::MAX && self.bbox[1] == u16::MAX
    }

    /// Update the bbox to include the given tile coordinates.
    #[inline(always)]
    pub(crate) fn include_tile(&mut self, wtile_x: u16, wtile_y: u16) {
        self.bbox[0] = self.bbox[0].min(wtile_x);
        self.bbox[1] = self.bbox[1].min(wtile_y);
        self.bbox[2] = self.bbox[2].max(wtile_x + 1);
        self.bbox[3] = self.bbox[3].max(wtile_y + 1);
    }

    /// Scale this bounding box by the given scale factors.
    ///
    /// Multiplies each coordinate by the corresponding scale factor to convert
    /// from one coordinate system to another.
    #[inline(always)]
    pub fn scale(&self, scale_x: u16, scale_y: u16) -> [u32; 4] {
        [
            u32::from(self.x0()) * u32::from(scale_x),
            u32::from(self.y0()) * u32::from(scale_y),
            u32::from(self.x1()) * u32::from(scale_x),
            u32::from(self.y1()) * u32::from(scale_y),
        ]
    }

    /// Expands the bounding box outward by the given pixel amounts in each direction.
    ///
    /// Pixel values are converted to tile coordinates (rounding up) and clamped to the
    /// valid range `[0, max_x)` × `[0, max_y)`. The result is a new bounding box in
    /// wide tile coordinates.
    pub fn expand_by_pixels(&self, expansion: Rect, max_x: u16, max_y: u16) -> Self {
        // The expansion rect is centered at origin:
        // - Negative coordinates (x0, y0) represent left/top expansion
        // - Positive coordinates (x1, y1) represent right/bottom expansion
        let left_px = (-expansion.x0).max(0.0).ceil() as u16;
        let top_px = (-expansion.y0).max(0.0).ceil() as u16;
        let right_px = expansion.x1.max(0.0).ceil() as u16;
        let bottom_px = expansion.y1.max(0.0).ceil() as u16;

        // Convert pixel expansion to tile expansion (round up)
        let left_tiles = left_px.div_ceil(WideTile::WIDTH);
        let top_tiles = top_px.div_ceil(Tile::HEIGHT);
        let right_tiles = right_px.div_ceil(WideTile::WIDTH);
        let bottom_tiles = bottom_px.div_ceil(Tile::HEIGHT);

        Self::new([
            self.x0().saturating_sub(left_tiles),
            self.y0().saturating_sub(top_tiles),
            (self.x1() + right_tiles).min(max_x),
            (self.y1() + bottom_tiles).min(max_y),
        ])
    }
}

impl Wide<MODE_CPU> {
    /// Create a new container for wide tiles.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_internal(width, height)
    }
}

impl Wide<MODE_HYBRID> {
    /// Create a new container for wide tiles.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_internal(width, height)
    }
}

impl<const MODE: u8> Wide<MODE> {
    /// Create a new container for wide tiles.
    fn new_internal(width: u16, height: u16) -> Self {
        let width_tiles = width.div_ceil(WideTile::WIDTH);
        let height_tiles = height.div_ceil(Tile::HEIGHT);
        let mut tiles = Vec::with_capacity(usize::from(width_tiles) * usize::from(height_tiles));

        for h in 0..height_tiles {
            for w in 0..width_tiles {
                tiles.push(WideTile::<MODE>::new_internal(
                    w * WideTile::WIDTH,
                    h * Tile::HEIGHT,
                ));
            }
        }

        Self {
            tiles,
            width,
            height,
            attrs: CommandAttrs::default(),
            layer_stack: vec![],
            clip_stack: vec![],
            // Start with root node 0.
            filter_node_stack: vec![0],
            clipped_filter_layer_depth: 0,
        }
    }

    /// Whether there are any existing layers that haven't been popped yet.
    pub fn has_layers(&self) -> bool {
        !self.layer_stack.is_empty()
    }

    /// Reset all tiles in the container.
    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.bg = PremulColor::from_alpha_color(TRANSPARENT);
            tile.cmds.clear();
            tile.layer_ids.truncate(1);
            tile.layer_cmd_ranges.clear();
            tile.layer_cmd_ranges
                .insert(0, LayerCommandRanges::default());
        }
        self.attrs.clear();
        self.layer_stack.clear();
        self.clip_stack.clear();
        self.filter_node_stack.truncate(1);
        self.clipped_filter_layer_depth = 0;
    }

    /// Return the number of horizontal tiles.
    pub fn width_tiles(&self) -> u16 {
        self.width.div_ceil(WideTile::WIDTH)
    }

    /// Return the number of vertical tiles.
    pub fn height_tiles(&self) -> u16 {
        self.height.div_ceil(Tile::HEIGHT)
    }

    /// Get the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get(&self, x: u16, y: u16) -> &WideTile<MODE> {
        assert!(
            x < self.width_tiles() && y < self.height_tiles(),
            "attempted to access out-of-bounds wide tile"
        );

        &self.tiles[usize::from(y) * usize::from(self.width_tiles()) + usize::from(x)]
    }

    /// Get mutable access to the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get_mut(&mut self, x: u16, y: u16) -> &mut WideTile<MODE> {
        assert!(
            x < self.width_tiles() && y < self.height_tiles(),
            "attempted to access out-of-bounds wide tile"
        );

        let idx = usize::from(y) * usize::from(self.width_tiles()) + usize::from(x);
        &mut self.tiles[idx]
    }

    /// Return a reference to all wide tiles.
    pub fn tiles(&self) -> &[WideTile<MODE>] {
        self.tiles.as_slice()
    }

    /// Get the current layer Id.
    #[inline(always)]
    pub fn get_current_layer_id(&self) -> LayerId {
        self.layer_stack.last().map_or(0, |l| l.layer_id)
    }

    /// Update the bounding box of the current layer to include the given tile.
    /// Should be called whenever a command is generated for a tile.
    #[inline]
    fn update_current_layer_bbox(&mut self, wtile_x: u16, wtile_y: u16) {
        if let Some(layer) = self.layer_stack.last_mut() {
            layer.wtile_bbox.include_tile(wtile_x, wtile_y);
        }
    }

    /// Generate wide tile commands from the strip buffer.
    ///
    /// This method processes a buffer of strips that represent a path, applies the fill rule,
    /// and generates appropriate drawing commands for each affected wide tile.
    ///
    /// # Algorithm overview:
    /// 1. For each strip in the buffer:
    ///    - Calculate its position and width in pixels
    ///    - Determine which wide tiles the strip intersects
    ///    - Generate alpha fill commands for the intersected wide tiles
    /// 2. For active fill regions (determined by fill rule):
    ///    - Generate solid fill commands for the regions between strips
    pub fn generate(
        &mut self,
        strip_buf: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        thread_idx: u8,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        let alpha_base_idx = strip_buf[0].alpha_idx();

        // Create shared attributes for all commands from this path
        let attrs_idx = self.attrs.fill.len() as u32;
        self.attrs.fill.push(FillAttrs {
            thread_idx,
            paint,
            blend_mode,
            mask,
            alpha_base_idx,
        });

        // Get current clip bounding box or full viewport if no clip is active
        let bbox = self.active_bbox();

        // Save current_layer_id to avoid borrowing issues
        let current_layer_id = self.get_current_layer_id();

        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];

            debug_assert!(
                strip.y < self.height,
                "Strips below the viewport should have been culled prior to this stage."
            );

            // Don't render strips that are outside the viewport width
            if strip.x >= self.width {
                continue;
            }

            let next_strip = &strip_buf[i + 1];
            let x0 = strip.x;
            let strip_y = strip.strip_y();

            // Skip strips outside the current clip bounding box
            if strip_y < bbox.y0() {
                continue;
            }
            if strip_y >= bbox.y1() {
                // The rest of our strips must be outside the clip, so we can break early.
                break;
            }

            // Calculate the width of the strip in columns
            let mut col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            // Can potentially be 0 if strip only changes winding without covering pixels
            let strip_width = next_col.saturating_sub(col) as u16;
            let x1 = x0.saturating_add(strip_width);

            // Calculate which wide tiles this strip intersects
            let wtile_x0 = (x0 / WideTile::WIDTH).max(bbox.x0());
            // It's possible that a strip extends into a new wide tile, but we don't actually
            // have as many wide tiles (e.g. because the pixmap width is only 512, but
            // strip ends at 513), so take the minimum between the rounded values and `width_tiles`.
            let wtile_x1 = x1
                .div_ceil(WideTile::WIDTH)
                .min(bbox.x1())
                .min(WideTile::MAX_WIDE_TILE_COORD);

            // Adjust column starting position if needed to respect clip boundaries
            let mut x = x0;
            let clip_x = bbox.x0() * WideTile::WIDTH;
            if clip_x > x {
                col += u32::from(clip_x - x);
                x = clip_x;
            }

            // Generate alpha fill commands for each wide tile intersected by this strip
            for wtile_x in wtile_x0..wtile_x1 {
                let x_wtile_rel = x % WideTile::WIDTH;
                // Restrict the width of the fill to the width of the wide tile
                let width = x1.min((wtile_x + 1) * WideTile::WIDTH) - x;
                let cmd = CmdAlphaFill {
                    x: x_wtile_rel,
                    width,
                    alpha_offset: col * u32::from(Tile::HEIGHT) - alpha_base_idx,
                    attrs_idx,
                };
                x += width;
                col += u32::from(width);
                self.get_mut(wtile_x, strip_y).strip(cmd, current_layer_id);
                self.update_current_layer_bbox(wtile_x, strip_y);
            }

            // Determine if the region between this strip and the next should be filled.
            let active_fill = next_strip.fill_gap();

            // If region should be filled and both strips are on the same row,
            // generate fill commands for the region between them
            if active_fill && strip_y == next_strip.strip_y() {
                // Clamp the fill to the clip bounding box
                x = x1.max(bbox.x0() * WideTile::WIDTH);
                let x2 = next_strip.x.min(
                    self.width
                        .checked_next_multiple_of(WideTile::WIDTH)
                        .unwrap_or(u16::MAX),
                );
                let wfxt0 = (x1 / WideTile::WIDTH).max(bbox.x0());
                let wfxt1 = x2
                    .div_ceil(WideTile::WIDTH)
                    .min(bbox.x1())
                    .min(WideTile::MAX_WIDE_TILE_COORD);

                // Compute fill hint based on paint type
                let fill_attrs = &self.attrs.fill[attrs_idx as usize];
                let fill_hint = if fill_attrs.mask.is_none() {
                    match &fill_attrs.paint {
                        Paint::Solid(s) if s.is_opaque() => FillHint::OpaqueSolid(*s),
                        Paint::Indexed(idx) => {
                            if let Some(EncodedPaint::Image(img)) = encoded_paints.get(idx.index())
                                && !img.may_have_opacities
                                && img.sampler.alpha == 1.0
                            {
                                FillHint::OpaqueImage
                            } else {
                                FillHint::None
                            }
                        }
                        _ => FillHint::None,
                    }
                } else {
                    FillHint::None
                };

                // Generate fill commands for each wide tile in the fill region
                for wtile_x in wfxt0..wfxt1 {
                    let x_wtile_rel = x % WideTile::WIDTH;
                    let width = x2.min(
                        (wtile_x
                            .checked_add(1)
                            .unwrap_or(WideTile::MAX_WIDE_TILE_COORD))
                            * WideTile::WIDTH,
                    ) - x;
                    x += width;
                    self.get_mut(wtile_x, strip_y).fill(
                        x_wtile_rel,
                        width,
                        attrs_idx,
                        current_layer_id,
                        fill_hint,
                    );
                    // TODO: This bbox update might be redundant since filled regions are always
                    // bounded by strip regions (which already update the bbox). Consider removing
                    // this in a follow-up with proper benchmarks to verify correctness.
                    self.update_current_layer_bbox(wtile_x, strip_y);
                }
            }
        }
    }

    /// Push a new layer with the given properties.
    ///
    /// Rendering will be directed to the layer storage identified by `layer_id`.
    /// This is used for filter effects that require access to a fully-rendered layer.
    ///
    /// If `filter` is Some, builds render graph nodes for filter effects.
    pub fn push_layer(
        &mut self,
        layer_id: LayerId,
        clip_path: Option<impl Into<Box<[Strip]>>>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
        filter: Option<Filter>,
        transform: Affine,
        render_graph: &mut RenderGraph,
        thread_idx: u8,
    ) {
        // Some explanations about what is going on here: We support the concept of
        // layers, where a user can push a new layer (with certain properties), draw some
        // stuff, and finally pop the layer, as part of which the layer as a whole will be
        // blended into the previous layer.
        // There are 3 "straightforward" properties that can be set for each layer:
        // 1) The blend mode that should be used to blend the layer into the backdrop.
        // 2) A mask that will be applied to the whole layer in the very end before blending.
        // 3) An optional opacity that will be applied to the whole layer before blending (this
        //    could in theory be simulated with an alpha mask, but since it's a common operation and
        //    we only have a single opacity, this can easily be optimized.
        //
        // Finally, you can also add a clip path to the layer. However, clipping has its own
        // more complicated logic for pushing/popping buffers where drawing is also suppressed
        // in clipped-out wide tiles. Because of this, in case we have one of the above properties
        // AND a clipping path, we will actually end up pushing two buffers, the first one handles
        // the three properties and the second one is just for clip paths. That is a bit wasteful
        // and I believe it should be possible to process them all in just one go, but for now
        // this is good enough, and it allows us to implement blending without too deep changes to
        // the original clipping implementation.

        // Build render graph node ONLY if we have a filter.
        // The render graph tracks dependencies and execution order for filter effects.
        if let Some(filter) = &filter {
            // Create a FilterLayer node that combines render + filter + other operations
            let child_node = render_graph.add_node(RenderNodeKind::FilterLayer {
                layer_id,
                filter: filter.clone(),
                // Bounding box starts inverted and will be updated in pop_layer with actual bounds
                wtile_bbox: WideTilesBbox::inverted(),
                transform,
            });

            // Connect to parent node if there is one
            if let Some(&parent_node) = self.filter_node_stack.last() {
                render_graph.add_edge(
                    child_node,
                    parent_node,
                    DependencyKind::Sequential { layer_id },
                );
            }

            // Push this filter node onto the stack so subsequent filters depend on it
            self.filter_node_stack.push(child_node);
        }

        let has_filter = filter.is_some();
        let has_clip = clip_path.is_some();
        let layer_kind = if has_filter {
            LayerKind::Filtered(layer_id)
        } else {
            LayerKind::Regular(layer_id)
        };

        // Filtered layers with clipping require special handling: normally, tiles with
        // zero-winding clips suppress all drawing. However, filters need the full layer
        // content rendered (including zero-clipped areas) before applying the clip as a mask.
        // When this flag is true, we generate explicit drawing commands instead of just counters.
        let in_clipped_filter_layer = has_filter && has_clip;

        // Increment the depth counter so that active_bbox() returns the full viewport
        // instead of the clipped bbox. This ensures command generation covers all tiles,
        // allowing the filter to process the entire layer before the clip is applied.
        if in_clipped_filter_layer {
            self.clipped_filter_layer_depth += 1;
        }

        let layer = Layer {
            layer_id,
            clip: has_clip,
            blend_mode,
            opacity,
            mask,
            filter,
            wtile_bbox: WideTilesBbox::inverted(),
        };

        // In case we do blending, masking, opacity, or filtering, push one buffer per wide tile.
        //
        // Layers require buffers for different reasons:
        // - Blending, opacity, and masking: Need to composite results with the backdrop
        // - Filtering: Content must be rendered to a buffer before applying filter effects
        //
        // The layer_kind parameter distinguishes how buffers are managed:
        // - Regular layers use the local blend_buf stack
        // - Filtered layers are materialized in persistent layer storage for filter processing
        // - Clip layers have special handling for clipping operations
        if layer.needs_buf() {
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    let tile = self.get_mut(x, y);
                    tile.push_buf(layer_kind);
                    // Mark tiles that are in a clipped filter layer so they generate
                    // explicit clip commands for proper filter processing.
                    tile.in_clipped_filter_layer = in_clipped_filter_layer;
                }
            }
        }

        // If we have a clip path, push another buffer in the affected wide tiles.
        // Note that it is important that we FIRST push the buffer for blending etc. and
        // only then for clipping, otherwise we will use the empty clip buffer as the backdrop
        // for blending!
        if let Some(clip) = clip_path {
            self.push_clip(clip, layer_id, thread_idx);
        }

        self.layer_stack.push(layer);
    }

    /// Pop a previously pushed layer.
    ///
    /// This method finalizes the layer by:
    /// - Expanding the bounding box if filter effects are present
    /// - Updating the parent layer's bounding box to include this layer's bounds
    /// - Completing render graph nodes for filter effects
    /// - Generating filter commands for each tile
    /// - Popping any associated clip
    /// - Applying mask, opacity, and blend mode operations if needed
    pub fn pop_layer(&mut self, render_graph: &mut RenderGraph) {
        // This method basically unwinds everything we did in `push_layer`.
        let mut layer = self.layer_stack.pop().unwrap();

        if let Some(filter) = &layer.filter {
            // Update render graph node with final bounding box
            if let Some(node_id) = self.filter_node_stack.pop() {
                // Get the transform from the FilterLayer node and scale the expansion by it
                if let Some(node) = render_graph.nodes.get_mut(node_id)
                    && let RenderNodeKind::FilterLayer {
                        wtile_bbox,
                        transform,
                        ..
                    } = &mut node.kind
                {
                    // Calculate expansion in device/pixel space, accounting for the full transform.
                    // This ensures that rotated filters (e.g., drop shadows) have correct bounds.
                    let expansion = filter.bounds_expansion(transform);
                    let expanded_bbox = layer.wtile_bbox.expand_by_pixels(
                        expansion,
                        self.width_tiles(),
                        self.height_tiles(),
                    );
                    let clip_bbox = self.active_bbox();
                    let final_bbox = expanded_bbox.intersect(clip_bbox);

                    // Update both the local layer and the render graph node
                    layer.wtile_bbox = final_bbox;
                    *wtile_bbox = final_bbox;
                }
                // Record this node in execution order (children before parents)
                render_graph.record_node_for_execution(node_id);
            }

            // Generate filter commands for each tile (used for non-graph path rendering)
            // Apply filter BEFORE clipping (per SVG spec: filter → clip → mask → opacity → blend)
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    self.get_mut(x, y).filter(layer.layer_id, filter.clone());
                }
            }
        }

        // Union this layer's bbox into the parent layer's bbox.
        // This ensures the parent knows about all tiles used by this child layer,
        // which is important for filter effects that may expand beyond the original content bounds.
        if let Some(parent_layer) = self.layer_stack.last_mut() {
            parent_layer.wtile_bbox.union(layer.wtile_bbox);
        }

        if layer.clip {
            self.pop_clip();
        }

        let needs_buf = layer.needs_buf();

        if needs_buf {
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    let t = self.get_mut(x, y);

                    // Optimization: If no drawing happened since the last `PushBuf`, then we don't
                    // need to do any masking or buffer-wide opacity work. The same holds for
                    // blending, unless it is destructive blending.
                    let has_draw_commands = !matches!(t.cmds.last().unwrap(), &Cmd::PushBuf(_));
                    if has_draw_commands {
                        if let Some(mask) = layer.mask.clone() {
                            t.mask(mask);
                        }
                        t.opacity(layer.opacity);
                    }
                    if has_draw_commands || layer.blend_mode.is_destructive() {
                        t.blend(layer.blend_mode);
                    }

                    t.pop_buf();
                }
            }
        }

        let in_clipped_filter_layer = layer.filter.is_some() && layer.clip;
        // Decrement the depth counter after popping a filtered layer with clip
        if in_clipped_filter_layer {
            self.clipped_filter_layer_depth -= 1;
        }
    }

    /// Adds a clipping region defined by the provided strips.
    ///
    /// This method takes a vector of strips representing a clip path, calculates the
    /// intersection with the current clip region, and updates the clip stack.
    ///
    /// # Algorithm overview:
    /// 1. Calculate bounding box of the clip path
    /// 2. Intersect with current clip bounding box
    /// 3. For each tile in the intersected bounding box:
    ///    - If covered by zero winding: `push_zero_clip`
    ///    - If fully covered by non-zero winding: do nothing (clip is a no-op)
    ///    - If partially covered: `push_clip`
    pub fn push_clip(
        &mut self,
        strips: impl Into<Box<[Strip]>>,
        layer_id: LayerId,
        thread_idx: u8,
    ) {
        let strips = strips.into();
        let n_strips = strips.len();

        // Calculate the bounding box of the clip path in strip coordinates
        let path_bbox = if n_strips <= 1 {
            WideTilesBbox::empty()
        } else {
            // Calculate the y range from first to last strip in wide tile coordinates
            let wtile_y0 = strips[0].strip_y();
            let wtile_y1 = strips[n_strips.saturating_sub(1)].strip_y() + 1;

            // Calculate the x range by examining all strips in wide tile coordinates
            let mut wtile_x0 = strips[0].x / WideTile::WIDTH;
            let mut wtile_x1 = wtile_x0;
            for i in 0..n_strips.saturating_sub(1) {
                let strip = &strips[i];
                let next_strip = &strips[i + 1];
                let width =
                    ((next_strip.alpha_idx() - strip.alpha_idx()) / u32::from(Tile::HEIGHT)) as u16;
                let x = strip.x;
                wtile_x0 = wtile_x0.min(x / WideTile::WIDTH);
                wtile_x1 = wtile_x1.max((x + width).div_ceil(WideTile::WIDTH));
            }
            WideTilesBbox::new([wtile_x0, wtile_y0, wtile_x1, wtile_y1])
        };

        let parent_bbox = self.active_bbox();
        // Determine which tiles need clip processing:
        // - For clipped filter layers: active_bbox() returns the full viewport, so parent_bbox
        //   already covers all tiles. We need to process all of them because the filter needs
        //   the entire layer rendered, and tiles outside the clip path must get `PushZeroClip`
        //   commands to properly suppress their content after filtering.
        // - For normal clips: Intersect with the path bounds to only process tiles that are
        //   actually affected by the clip path, avoiding unnecessary work.
        let clip_bbox = if self.clipped_filter_layer_depth > 0 {
            // Use parent_bbox as-is (full viewport) to process all tiles
            parent_bbox
        } else {
            // Optimize by processing only the intersection of parent and path bounds
            parent_bbox.intersect(path_bbox)
        };

        let mut cur_wtile_x = clip_bbox.x0();
        let mut cur_wtile_y = clip_bbox.y0();

        // Process strips to determine the clipping state for each wide tile
        for i in 0..n_strips.saturating_sub(1) {
            let strip = &strips[i];
            let strip_y = strip.strip_y();

            // Skip strips before current wide tile row
            if strip_y < cur_wtile_y {
                continue;
            }

            // Process wide tiles in rows before this strip's row
            // These wide tiles are all zero-winding (outside the path)
            while cur_wtile_y < strip_y.min(clip_bbox.y1()) {
                for wtile_x in cur_wtile_x..clip_bbox.x1() {
                    self.get_mut(wtile_x, cur_wtile_y).push_zero_clip(layer_id);
                }
                // Reset x to the left edge of the clip bounding box
                cur_wtile_x = clip_bbox.x0();
                // Move to the next row
                cur_wtile_y += 1;
            }

            // If we've reached the bottom of the clip bounding box, stop processing.
            // Note that we are explicitly checking >= instead of ==, so that we abort if the clipping box
            // is zero-area (see issue 1072).
            if cur_wtile_y >= clip_bbox.y1() {
                break;
            }

            // Process wide tiles to the left of this strip in the same row
            let x = strip.x;
            let wtile_x_clamped = (x / WideTile::WIDTH).min(clip_bbox.x1());
            if cur_wtile_x < wtile_x_clamped {
                // If winding is zero or doesn't match fill rule, these wide tiles are outside the path
                let is_inside = strip.fill_gap();
                if !is_inside {
                    for wtile_x in cur_wtile_x..wtile_x_clamped {
                        self.get_mut(wtile_x, cur_wtile_y).push_zero_clip(layer_id);
                    }
                }
                // If winding is nonzero, then wide tiles covered entirely
                // by sparse fill are no-op (no clipping is applied).
                cur_wtile_x = wtile_x_clamped;
            }

            // Process wide tiles covered by the strip - these need actual clipping
            let next_strip = &strips[i + 1];
            let width =
                ((next_strip.alpha_idx() - strip.alpha_idx()) / u32::from(Tile::HEIGHT)) as u16;
            let wtile_x1 = (x + width).div_ceil(WideTile::WIDTH).min(clip_bbox.x1());
            if cur_wtile_x < wtile_x1 {
                for wtile_x in cur_wtile_x..wtile_x1 {
                    self.get_mut(wtile_x, cur_wtile_y).push_clip(layer_id);
                }
                cur_wtile_x = wtile_x1;
            }
        }

        // Process any remaining wide tiles in the bounding box (all zero-winding)
        while cur_wtile_y < clip_bbox.y1() {
            for wtile_x in cur_wtile_x..clip_bbox.x1() {
                self.get_mut(wtile_x, cur_wtile_y).push_zero_clip(layer_id);
            }
            cur_wtile_x = clip_bbox.x0();
            cur_wtile_y += 1;
        }

        self.clip_stack.push(Clip {
            clip_bbox,
            strips,
            thread_idx,
        });
    }

    /// Get the bounding box of the current clip region or the entire viewport if no clip regions are active.
    fn active_bbox(&self) -> WideTilesBbox {
        // When in a clipped filter layer, use full viewport to allow
        // filter to process the complete layer before applying clip as mask
        if self.clipped_filter_layer_depth > 0 {
            return self.full_viewport_bbox();
        }

        self.clip_stack
            .last()
            .map(|top| top.clip_bbox)
            .unwrap_or_else(|| self.full_viewport_bbox())
    }

    /// Returns the bounding box covering the entire viewport in wide tile coordinates.
    fn full_viewport_bbox(&self) -> WideTilesBbox {
        WideTilesBbox::new([0, 0, self.width_tiles(), self.height_tiles()])
    }

    /// Removes the most recently added clip region.
    ///
    /// This is the inverse operation of `push_clip`, carefully undoing all the clipping
    /// operations while also handling any rendering needed for the clip region itself.
    ///
    /// # Algorithm overview:
    /// 1. Retrieve the top clip from the stack
    /// 2. For each wide tile in the clip's bounding box:
    ///    - If covered by zero winding: `pop_zero_clip`
    ///    - If fully covered by non-zero winding: do nothing (was no-op)
    ///    - If partially covered: render the clip and `pop_clip`
    ///
    /// This operation must be symmetric with `push_clip` to maintain a balanced clip stack.
    fn pop_clip(&mut self) {
        let Clip {
            clip_bbox,
            strips,
            thread_idx,
        } = self.clip_stack.pop().unwrap();
        let n_strips = strips.len();

        if n_strips == 0 {
            return;
        }

        // Compute base alpha index and create shared clip attributes
        let alpha_base_idx = strips[0].alpha_idx();
        let clip_attrs_idx = self.attrs.clip.len() as u32;
        self.attrs.clip.push(ClipAttrs {
            thread_idx,
            alpha_base_idx,
        });

        let mut cur_wtile_x = clip_bbox.x0();
        let mut cur_wtile_y = clip_bbox.y0();
        let mut pop_pending = false;

        // Process each strip to determine the clipping state for each tile
        for i in 0..n_strips.saturating_sub(1) {
            let strip = &strips[i];
            let strip_y = strip.strip_y();

            // Skip strips before current tile row
            if strip_y < cur_wtile_y {
                continue;
            }

            // Process tiles in rows before this strip's row
            // These tiles all had zero-winding clips
            while cur_wtile_y < strip_y.min(clip_bbox.y1()) {
                // Handle any pending clip pop from previous iteration
                if core::mem::take(&mut pop_pending) {
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                    cur_wtile_x += 1;
                }

                // Pop zero clips for all remaining tiles in this row
                for wtile_x in cur_wtile_x..clip_bbox.x1() {
                    self.get_mut(wtile_x, cur_wtile_y).pop_zero_clip();
                }
                cur_wtile_x = clip_bbox.x0();
                cur_wtile_y += 1;
            }

            // If we've reached the bottom of the clip bounding box, stop processing
            // Note that we are explicitly checking >= instead of ==, so that we abort if the clipping box
            // is zero-area (see issue 1072).
            if cur_wtile_y >= clip_bbox.y1() {
                break;
            }

            // Process tiles to the left of this strip in the same row
            let x0 = strip.x;
            let wtile_x_clamped = (x0 / WideTile::WIDTH).min(clip_bbox.x1());
            if cur_wtile_x < wtile_x_clamped {
                // Handle any pending clip pop from previous iteration
                if core::mem::take(&mut pop_pending) {
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                    cur_wtile_x += 1;
                }

                // Pop zero clips for tiles that had zero winding or didn't match fill rule
                // TODO: The winding check is probably not needed; if there was a fill,
                // the logic below should have advanced wtile_x.
                let is_inside = strip.fill_gap();
                if !is_inside {
                    for wtile_x in cur_wtile_x..wtile_x_clamped {
                        self.get_mut(wtile_x, cur_wtile_y).pop_zero_clip();
                    }
                }
                cur_wtile_x = wtile_x_clamped;
            }

            // Process tiles covered by the strip - render clip content and pop
            let next_strip = &strips[i + 1];
            let strip_width =
                ((next_strip.alpha_idx() - strip.alpha_idx()) / u32::from(Tile::HEIGHT)) as u16;
            let mut clipped_x1 = x0 + strip_width;
            let wtile_x0 = (x0 / WideTile::WIDTH).max(clip_bbox.x0());
            let wtile_x1 = clipped_x1.div_ceil(WideTile::WIDTH).min(clip_bbox.x1());

            // Calculate starting position and column for alpha mask
            let mut x = x0;
            let mut col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let clip_x = clip_bbox.x0() * WideTile::WIDTH;
            if clip_x > x {
                col += u32::from(clip_x - x);
                x = clip_x;
                clipped_x1 = clip_x.max(clipped_x1);
            }

            // Render clip strips for each affected tile and mark for popping
            for wtile_x in wtile_x0..wtile_x1 {
                // If we've moved past tile_x and have a pending pop, do it now
                if cur_wtile_x < wtile_x && core::mem::take(&mut pop_pending) {
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                }

                // Calculate the portion of the strip that affects this tile
                let x_rel = x % WideTile::WIDTH;
                let width = clipped_x1.min((wtile_x + 1) * WideTile::WIDTH) - x;

                // Create clip strip command for rendering the partial coverage
                let cmd = CmdClipAlphaFill {
                    x: x_rel,
                    width,
                    alpha_offset: col * u32::from(Tile::HEIGHT) - alpha_base_idx,
                    attrs_idx: clip_attrs_idx,
                };
                x += width;
                col += u32::from(width);

                // Apply the clip strip command and update state
                self.get_mut(wtile_x, cur_wtile_y).clip_strip(cmd);
                cur_wtile_x = wtile_x;

                // Only request a pop if the x coordinate is actually inside the bounds.
                if cur_wtile_x < clip_bbox.x1() {
                    pop_pending = true;
                }
            }

            // Handle fill regions between strips based on fill rule
            let is_inside = next_strip.fill_gap();
            if is_inside && strip_y == next_strip.strip_y() {
                if cur_wtile_x >= clip_bbox.x1() {
                    continue;
                }

                let x2 = next_strip.x;
                let clipped_x2 = x2.min((cur_wtile_x + 1) * WideTile::WIDTH);
                let width = clipped_x2.saturating_sub(clipped_x1);

                // If there's a gap, fill it. Only do this if the fill wouldn't cover the
                // whole tile, as such clips are skipped by the `push_clip` function. See
                // <https://github.com/linebender/vello/blob/de0659e4df9842c8857153841a2b4ba6f1020bb0/sparse_strips/vello_common/src/coarse.rs#L504-L516>
                if width > 0 && width < WideTile::WIDTH {
                    let x_rel = clipped_x1 % WideTile::WIDTH;
                    self.get_mut(cur_wtile_x, cur_wtile_y)
                        .clip_fill(x_rel, width);
                }

                // If the next strip is a sentinel, skip the fill
                // It's a sentinel in the row if there is non-zero winding for the sparse fill
                // Look more into this in the strip.rs render function
                if x2 == u16::MAX {
                    continue;
                }

                // If fill extends to next tile, pop current and handle next
                if x2 > (cur_wtile_x + 1) * WideTile::WIDTH {
                    if core::mem::take(&mut pop_pending) {
                        self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                    }

                    let width2 = x2 % WideTile::WIDTH;
                    cur_wtile_x = x2 / WideTile::WIDTH;

                    // If the strip is outside the clipping box, we don't need to do any
                    // filling, so we continue (also to prevent out-of-bounds access).
                    if cur_wtile_x >= clip_bbox.x1() {
                        continue;
                    }

                    if width2 > 0 {
                        // An important thing to note: Note that we are only applying
                        // `clip_fill` to the wide tile that is actually covered by the next
                        // strip, and not the ones in-between! For example, if the first strip
                        // is in wide tile 1 and the second in wide tile 4, we will do a clip
                        // fill in wide tile 1 and 4, but not in 2 and 3. The reason for this is
                        // that any tile in-between is fully covered and thus no clipping is
                        // necessary at all. See also the `push_clip` function, where we don't
                        // push a new buffer for such tiles.
                        self.get_mut(cur_wtile_x, cur_wtile_y).clip_fill(0, width2);
                    }
                }
            }
        }

        // Handle any pending clip pop from the last iteration
        if core::mem::take(&mut pop_pending) {
            self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
            cur_wtile_x += 1;
        }

        // Process any remaining tiles in the bounding box (all zero-winding)
        while cur_wtile_y < clip_bbox.y1() {
            for wtile_x in cur_wtile_x..clip_bbox.x1() {
                self.get_mut(wtile_x, cur_wtile_y).pop_zero_clip();
            }
            cur_wtile_x = clip_bbox.x0();
            cur_wtile_y += 1;
        }
    }
}

/// A wide tile.
#[derive(Debug)]
pub struct WideTile<const MODE: u8 = MODE_CPU> {
    /// The x coordinate of the wide tile.
    pub x: u16,
    /// The y coordinate of the wide tile.
    pub y: u16,
    /// The background of the tile.
    pub bg: PremulColor,
    /// The draw commands of the tile.
    pub cmds: Vec<Cmd>,
    /// The number of zero-winding clips.
    pub n_zero_clip: usize,
    /// The number of non-zero-winding clips.
    pub n_clip: usize,
    /// The number of pushed buffers.
    pub n_bufs: usize,
    /// True when this tile is in a filtered layer with clipping applied.
    /// When set, clip operations generate explicit commands instead of just
    /// tracking counters, allowing filters to process clipped content correctly.
    pub in_clipped_filter_layer: bool,
    /// Maps layer Id to command ranges for this tile.
    pub layer_cmd_ranges: HashMap<LayerId, LayerCommandRanges>,
    /// Vector of layer IDs this tile participates in.
    pub layer_ids: Vec<LayerKind>,
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;
    /// The maximum coordinate of a wide tile.
    pub const MAX_WIDE_TILE_COORD: u16 = u16::MAX / Self::WIDTH;
}

impl WideTile<MODE_CPU> {
    /// Create a new wide tile.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_internal(width, height)
    }
}

impl WideTile<MODE_HYBRID> {
    /// Create a new wide tile.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_internal(width, height)
    }
}

impl<const MODE: u8> WideTile<MODE> {
    /// Create a new wide tile.
    fn new_internal(x: u16, y: u16) -> Self {
        let mut layer_cmd_ranges = HashMap::new();
        layer_cmd_ranges.insert(0, LayerCommandRanges::default());
        Self {
            x,
            y,
            bg: PremulColor::from_alpha_color(TRANSPARENT),
            cmds: vec![],
            n_zero_clip: 0,
            n_clip: 0,
            n_bufs: 0,
            in_clipped_filter_layer: false,
            layer_cmd_ranges,
            layer_ids: vec![LayerKind::Regular(0)],
        }
    }

    /// Fill a rectangular region with a paint.
    ///
    /// Generates fill commands unless the tile is in a zero-clip region (fully clipped out).
    /// For clipped filter layers, commands are always generated since filters need the full
    /// layer content rendered before applying the clip as a mask.
    ///
    /// The `fill_hint` parameter is pre-computed by the caller based on paint type:
    /// - `OpaqueSolid(color)`: Paint is an opaque solid color, can replace background
    /// - `OpaqueImage`: Paint is an opaque image, can clear previous commands
    /// - `None`: No optimization available
    pub(crate) fn fill(
        &mut self,
        x: u16,
        width: u16,
        attrs_idx: u32,
        current_layer_id: LayerId,
        fill_hint: FillHint,
    ) {
        if !self.is_zero_clip() || self.in_clipped_filter_layer {
            match MODE {
                MODE_CPU => {
                    // Check if we can apply overdraw elimination optimization.
                    // This requires filling the entire tile width with no clip/buffer stack.
                    //
                    // Note that we could be more aggressive in optimizing a whole-tile opaque fill
                    // even with a clip stack. It would be valid to elide all drawing commands from
                    // the enclosing clip push up to the fill. Further, we could extend the clip
                    // push command to include a background color, rather than always starting with
                    // a transparent buffer. Lastly, a sequence of push(bg); strip/fill; pop could
                    // be replaced with strip/fill with the color (the latter is true even with a
                    // non-opaque color).
                    //
                    // However, the extra cost of tracking such optimizations may outweigh the
                    // benefit, especially in hybrid mode with GPU painting.
                    let can_override =
                        x == 0 && width == WideTile::WIDTH && self.n_clip == 0 && self.n_bufs == 0;

                    if can_override {
                        match fill_hint {
                            FillHint::OpaqueSolid(color) => {
                                self.cmds.clear();
                                self.bg = color;
                                if let Some(ranges) =
                                    self.layer_cmd_ranges.get_mut(&current_layer_id)
                                {
                                    ranges.clear();
                                }
                                return;
                            }
                            FillHint::OpaqueImage => {
                                // Opaque image: clear previous commands but still emit the fill.
                                self.cmds.clear();
                                self.bg = PremulColor::from_alpha_color(TRANSPARENT);
                                if let Some(ranges) =
                                    self.layer_cmd_ranges.get_mut(&current_layer_id)
                                {
                                    ranges.clear();
                                }
                                // Fall through to emit the fill command below, as opposed to
                                // solid paints where we have a return statement.
                            }
                            FillHint::None => {}
                        }
                    }

                    self.record_fill_cmd(current_layer_id, self.cmds.len());
                    self.cmds.push(Cmd::Fill(CmdFill {
                        x,
                        width,
                        attrs_idx,
                    }));
                }
                MODE_HYBRID => {
                    self.record_fill_cmd(current_layer_id, self.cmds.len());
                    self.cmds.push(Cmd::Fill(CmdFill {
                        x,
                        width,
                        attrs_idx,
                    }));
                }
                _ => unreachable!(),
            }
        }
    }

    /// Fill a region using an alpha mask from a strip.
    ///
    /// Generates alpha fill commands unless the tile is in a zero-clip region (fully clipped out).
    /// For clipped filter layers, commands are always generated since filters need the full
    /// layer content rendered before applying the clip as a mask.
    pub(crate) fn strip(&mut self, cmd_strip: CmdAlphaFill, current_layer_id: LayerId) {
        if !self.is_zero_clip() || self.in_clipped_filter_layer {
            self.record_fill_cmd(current_layer_id, self.cmds.len());
            self.cmds.push(Cmd::AlphaFill(cmd_strip));
        }
    }

    /// Adds a new clip region to the current wide tile.
    ///
    /// Pushes a clip buffer unless the tile is in a zero-clip region (fully clipped out).
    /// For clipped filter layers, clip buffers are always pushed since filters need explicit
    /// clip state to process the full layer before applying the clip as a mask.
    pub fn push_clip(&mut self, layer_id: LayerId) {
        if !self.is_zero_clip() || self.in_clipped_filter_layer {
            self.push_buf(LayerKind::Clip(layer_id));
            self.n_clip += 1;
        }
    }

    /// Removes the most recently added clip region from the current wide tile.
    ///
    /// Pops a clip buffer unless the tile is in a zero-clip region (fully clipped out).
    /// For clipped filter layers, clip buffers are always popped since filters need explicit
    /// clip state to process the full layer before applying the clip as a mask.
    pub fn pop_clip(&mut self) {
        if !self.is_zero_clip() || self.in_clipped_filter_layer {
            self.pop_buf();
            self.n_clip -= 1;
        }
    }

    /// Adds a zero-winding clip region to the stack.
    ///
    /// Zero-winding clips represent tiles completely outside the clip path.
    /// Normally these just increment a counter to suppress drawing, but for
    /// clipped filter layers we generate explicit commands so filters can
    /// process the entire layer before applying the clip as a mask.
    pub fn push_zero_clip(&mut self, layer_id: LayerId) {
        if self.in_clipped_filter_layer {
            // Generate explicit command for filter processing
            self.cmds.push(Cmd::PushZeroClip(layer_id));
        }
        self.n_zero_clip += 1;
    }

    /// Removes the most recently added zero-winding clip region.
    pub fn pop_zero_clip(&mut self) {
        if self.in_clipped_filter_layer {
            // Generate explicit command for filter processing
            self.cmds.push(Cmd::PopZeroClip);
        }
        self.n_zero_clip -= 1;
    }

    /// Checks if the current clip region is a zero-winding clip.
    pub fn is_zero_clip(&mut self) -> bool {
        self.n_zero_clip > 0
    }

    /// Applies a clip strip operation with the given parameters.
    ///
    /// Note: Unlike content operations (`strip`, `push_clip`, etc.), clip operations don't need
    /// the `|| self.in_clipped_filter_layer` check. Filter effects need full layer *content*
    /// rendered (even in zero-clip areas).
    pub fn clip_strip(&mut self, cmd_clip_strip: CmdClipAlphaFill) {
        if (!self.is_zero_clip()) && !matches!(self.cmds.last(), Some(Cmd::PushBuf(_))) {
            self.cmds.push(Cmd::ClipStrip(cmd_clip_strip));
        }
    }

    /// Applies a clip fill operation at the specified position and width.
    pub fn clip_fill(&mut self, x: u16, width: u16) {
        if (!self.is_zero_clip()) && !matches!(self.cmds.last(), Some(Cmd::PushBuf(_))) {
            self.cmds.push(Cmd::ClipFill(CmdClipFill { x, width }));
        }
    }

    /// Records the fill command for a specific layer.
    pub fn record_fill_cmd(&mut self, layer_id: LayerId, cmd_idx: usize) {
        self.layer_cmd_ranges.entry(layer_id).and_modify(|ranges| {
            ranges.full_range.end = cmd_idx + 1;
            if ranges.render_range.is_empty() {
                ranges.render_range = cmd_idx..cmd_idx + 1;
            } else {
                ranges.render_range.end = cmd_idx + 1;
            }
        });
    }

    /// Push a buffer for a new layer.
    ///
    /// Different layer kinds are handled differently:
    /// - Regular layers: Use local `blend_buf` stack for temporary storage
    /// - Filtered layers: Materialized in persistent layer storage for filter processing
    /// - Clip layers: Special handling for clipping operations
    fn push_buf(&mut self, layer_kind: LayerKind) {
        let top_layer = layer_kind.id();
        if matches!(layer_kind, LayerKind::Filtered(_)) {
            self.layer_cmd_ranges.insert(
                top_layer,
                LayerCommandRanges {
                    full_range: self.cmds.len()..self.cmds.len() + 1,
                    // Start with empty render_range; will be updated by `record_fill_cmd` and `pop_buf`.
                    render_range: self.cmds.len() + 1..self.cmds.len() + 1,
                },
            );
        } else if matches!(layer_kind, LayerKind::Clip(_)) {
            self.layer_cmd_ranges.entry(top_layer).and_modify(|ranges| {
                ranges.full_range.end = self.cmds.len() + 1;
                // Start with empty render_range; will be updated by `record_fill_cmd` and `pop_buf`.
                ranges.render_range = self.cmds.len() + 1..self.cmds.len() + 1;
            });
        }
        self.cmds.push(Cmd::PushBuf(layer_kind));
        self.layer_ids.push(layer_kind);
        self.n_bufs += 1;
    }

    /// Pop the most recent buffer.
    fn pop_buf(&mut self) {
        let top_layer = self.layer_ids.pop().unwrap();
        let mut next_layer = *self.layer_ids.last().unwrap();

        if matches!(self.cmds.last(), Some(&Cmd::PushBuf(_))) {
            // Optimization: If no drawing happened between the last `PushBuf`,
            // we can just pop it instead.
            self.cmds.pop();
        } else {
            self.layer_cmd_ranges
                .entry(top_layer.id())
                .and_modify(|ranges| {
                    ranges.full_range.end = self.cmds.len() + 1;
                });
            if top_layer.id() == next_layer.id() {
                next_layer = *self
                    .layer_ids
                    .get(self.layer_ids.len().saturating_sub(2))
                    .unwrap();
            }

            self.layer_cmd_ranges
                .entry(next_layer.id())
                .and_modify(|ranges| {
                    ranges.full_range.end = self.cmds.len() + 1;
                    ranges.render_range.end = self.cmds.len() + 1;
                });
            self.cmds.push(Cmd::PopBuf);
        }
        self.n_bufs -= 1;
    }

    /// Apply an opacity to the whole buffer.
    fn opacity(&mut self, opacity: f32) {
        if opacity != 1.0 {
            self.cmds.push(Cmd::Opacity(opacity));
        }
    }

    /// Apply a filter effect to the whole buffer.
    pub fn filter(&mut self, layer_id: LayerId, filter: Filter) {
        self.cmds.push(Cmd::Filter(layer_id, filter));
    }

    /// Apply a mask to the whole buffer.
    fn mask(&mut self, mask: Mask) {
        self.cmds.push(Cmd::Mask(mask));
    }

    /// Blend the current buffer into the previous buffer in the stack.
    fn blend(&mut self, blend_mode: BlendMode) {
        self.cmds.push(Cmd::Blend(blend_mode));
    }
}

/// Debug utilities for wide tiles.
///
/// These methods are only available in debug builds (`debug_assertions`).
/// They provide introspection into the command buffer for debugging and logging purposes.
#[cfg(debug_assertions)]
impl<const MODE: u8> WideTile<MODE> {
    /// Lists all commands in this wide tile with their indices and names.
    ///
    /// Returns a formatted string with each command on a new line, showing its index
    /// and human-readable name. This is useful for debugging and understanding the
    /// command sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let commands = wide_tile.list_commands();
    /// println!("{}", commands);
    /// // Output:
    /// // 0: PushBuf(Regular)
    /// // 1: FillPath
    /// // 2: PushZeroClip
    /// // 3: FillPath
    /// // 4: PopBuf
    /// ```
    #[allow(dead_code, reason = "useful for debugging")]
    pub fn list_commands(&self) -> String {
        self.cmds
            .iter()
            .enumerate()
            .map(|(i, cmd)| format!("{}: {}", i, cmd.name()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Optimization hint for fill operations, computed in `Wide::generate` and passed to `WideTile::fill`.
///
/// This enum communicates whether a fill operation can benefit from overdraw elimination:
/// - For opaque solid colors: we can set the background color directly and skip the fill
/// - For opaque images: we can clear previous commands but still need to emit the fill
#[derive(Debug, Clone, Copy)]
pub enum FillHint {
    /// No optimization possible, emit fill command normally.
    None,
    /// Paint is an opaque solid color - can replace background if conditions are met.
    OpaqueSolid(PremulColor),
    /// Paint is an opaque image - can clear previous commands if conditions are met.
    OpaqueImage,
}

/// Distinguishes between different types of layers and their storage strategies.
///
/// Each layer kind determines how the layer's content is stored and processed:
/// - Regular layers are blended on-the-fly using a temporary buffer stack
/// - Filtered layers are materialized in persistent storage for filter processing
/// - Clip layers are special buffers used for clipping operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Regular layer using local `blend_buf` stack for temporary storage.
    Regular(LayerId),
    /// Filtered layer materialized in persistent `layer_manager` storage.
    Filtered(LayerId),
    /// Clip layer for clipping operations.
    Clip(LayerId),
}

impl LayerKind {
    /// Get the underlying layer ID.
    ///
    /// All layer kinds contain a layer ID that uniquely identifies the layer.
    pub fn id(&self) -> LayerId {
        match self {
            Self::Regular(id) | Self::Filtered(id) | Self::Clip(id) => *id,
        }
    }
}

/// A drawing command for wide tiles.
///
/// Commands are executed in order to render the final image. They include
/// drawing operations (`Fill`, `AlphaFill`), layer management (`PushBuf`, `PopBuf`),
/// clipping operations (`ClipFill`, `ClipStrip`), and post-processing effects
/// (`Filter`, `Blend`, `Opacity`, `Mask`).
#[derive(Debug, PartialEq)]
pub enum Cmd {
    /// Fill a rectangular region with a solid color or paint.
    Fill(CmdFill),
    /// Fill a region with a paint, modulated by an alpha mask.
    AlphaFill(CmdAlphaFill),
    /// Pushes a new buffer for drawing.
    /// Regular layers use the local `blend_buf` stack.
    /// Filtered layers are materialized in persistent layer storage.
    PushBuf(LayerKind),
    /// Pops the most recent buffer and blends it into the previous buffer.
    PopBuf,
    /// A fill command within a clipping region.
    ///
    /// This command will blend the contents of the current buffer within the clip fill region
    /// into the previous buffer in the stack.
    ClipFill(CmdClipFill),
    /// A fill command with alpha mask within a clipping region.
    ///
    /// This command will blend the contents of the current buffer within the clip fill region
    /// into the previous buffer in the stack, with an additional alpha mask.
    ClipStrip(CmdClipAlphaFill),
    /// Marks entry into a zero-winding clip region for a clipped filter layer.
    ///
    /// Zero-winding clips represent tiles completely outside the clip path. For clipped
    /// filter layers, this command allows the filter to process the full layer content
    /// before applying the clip as a mask (per SVG spec: filter → clip → mask → blend).
    PushZeroClip(LayerId),
    /// Marks exit from a zero-winding clip region for a clipped filter layer.
    PopZeroClip,
    /// Apply a filter effect to a layer's contents.
    ///
    /// This command applies a filter (e.g., blur, drop shadow) to the specified layer's
    /// rendered content. Per the SVG specification, filters are applied before clipping,
    /// masking, blending, and opacity operations.
    Filter(LayerId, Filter),
    /// Blend the current buffer into the previous buffer.
    ///
    /// This command blends the contents of the current buffer into the previous buffer
    /// using the specified blend mode (e.g., multiply, screen, overlay).
    Blend(BlendMode),
    /// Apply uniform opacity to the current buffer.
    ///
    /// Multiplies the alpha channel of all pixels in the buffer by the given opacity value.
    Opacity(f32),
    /// Apply a mask to the current buffer.
    ///
    /// Modulates the alpha channel of the buffer using the provided mask.
    Mask(Mask),
}

#[cfg(debug_assertions)]
impl Cmd {
    /// Returns a human-readable name for this command.
    ///
    /// This is useful for debugging, logging, and displaying command information
    /// in a user-friendly format. To get detailed paint information, use `name_with_attrs`
    /// which can look up the paint from the command attributes.
    ///
    /// **Note:** This method is only available in debug builds (`debug_assertions`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fill(_) => "FillPath",
            Self::AlphaFill(_) => "AlphaFillPath",
            Self::PushBuf(layer_kind) => match layer_kind {
                LayerKind::Regular(_) => "PushBuf(Regular)",
                LayerKind::Filtered(_) => "PushBuf(Filtered)",
                LayerKind::Clip(_) => "PushBuf(Clip)",
            },
            Self::PopBuf => "PopBuf",
            Self::ClipFill(_) => "ClipPathFill",
            Self::ClipStrip(_) => "ClipPathStrip",
            Self::PushZeroClip(_) => "PushZeroClip",
            Self::PopZeroClip => "PopZeroClip",
            Self::Filter(_, _) => "Filter",
            Self::Blend(_) => "Blend",
            Self::Opacity(_) => "Opacity",
            Self::Mask(_) => "Mask",
        }
    }

    /// Returns a human-readable name for this command with detailed paint information.
    ///
    /// This variant looks up paint details from the command attributes for fill commands.
    ///
    /// **Note:** This method is only available in debug builds (`debug_assertions`).
    pub fn name_with_attrs(
        &self,
        fill_attrs: &[FillAttrs],
        encoded_paints: &[EncodedPaint],
    ) -> String {
        match self {
            Self::Fill(cmd) => {
                if let Some(attrs) = fill_attrs.get(cmd.attrs_idx as usize) {
                    format!("FillPath({})", paint_name(&attrs.paint, encoded_paints))
                } else {
                    format!("FillPath(attrs_idx={})", cmd.attrs_idx)
                }
            }
            Self::AlphaFill(cmd) => {
                if let Some(attrs) = fill_attrs.get(cmd.attrs_idx as usize) {
                    format!(
                        "AlphaFillPath({})",
                        paint_name(&attrs.paint, encoded_paints)
                    )
                } else {
                    format!("AlphaFillPath(attrs_idx={})", cmd.attrs_idx)
                }
            }
            _ => self.name().into(),
        }
    }
}

/// Returns a human-readable description of a paint.
#[cfg(debug_assertions)]
fn paint_name(paint: &Paint, encoded_paints: &[EncodedPaint]) -> String {
    match paint {
        Paint::Solid(color) => {
            let rgba = color.as_premul_rgba8();
            format!(
                "Solid(#{:02x}{:02x}{:02x}{:02x})",
                rgba.r, rgba.g, rgba.b, rgba.a
            )
        }
        Paint::Indexed(idx) => {
            let index = idx.index();
            if let Some(encoded) = encoded_paints.get(index) {
                let kind = match encoded {
                    EncodedPaint::Gradient(g) => match &g.kind {
                        crate::encode::EncodedKind::Linear(_) => "LinearGradient",
                        crate::encode::EncodedKind::Radial(_) => "RadialGradient",
                        crate::encode::EncodedKind::Sweep(_) => "SweepGradient",
                    },
                    EncodedPaint::Image(_) => "Image",
                    EncodedPaint::BlurredRoundedRect(_) => "BlurredRoundedRect",
                };
                format!("{}[{}]", kind, index)
            } else {
                format!("Indexed({})", index)
            }
        }
    }
}

/// Shared attributes for alpha fill commands.
#[derive(Debug, Clone, PartialEq)]
pub struct FillAttrs {
    /// The index of the thread that owns the alpha buffer
    /// containing the mask values at `alpha_idx`.
    /// Always 0 in single-threaded mode.
    pub thread_idx: u8,
    /// The paint (color, gradient, etc.) to fill the region with.
    // TODO: Store premultiplied colors as indexed paints as well, to reduce
    // memory overhead? Or get rid of indexed paints and inline all paints?
    pub paint: Paint,
    /// The blend mode to apply before drawing the contents.
    pub blend_mode: BlendMode,
    /// A mask to apply to the command.
    pub mask: Option<Mask>,
    /// Base index into the alpha buffer for this path's commands.
    /// Commands store a relative offset that is added to this base.
    alpha_base_idx: u32,
}

impl FillAttrs {
    /// Compute the absolute alpha buffer index from a relative offset.
    pub fn alpha_idx(&self, offset: u32) -> u32 {
        self.alpha_base_idx + offset
    }
}

/// Shared attributes for clip alpha fill commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipAttrs {
    /// The index of the thread that owns the alpha buffer
    /// containing the mask values at `alpha_idx`.
    /// Always 0 in single-threaded mode.
    pub thread_idx: u8,
    /// Base index into the alpha buffer for this clip path's commands.
    /// Commands store a relative offset that is added to this base.
    alpha_base_idx: u32,
}

impl ClipAttrs {
    /// Compute the absolute alpha buffer index from a relative offset.
    pub fn alpha_idx(&self, offset: u32) -> u32 {
        self.alpha_base_idx + offset
    }
}

/// Container for shared command attributes.
///
/// This struct holds the shared attributes for fill and clip commands,
/// allowing them to be passed together to functions that need both.
#[derive(Debug, Default, Clone)]
pub struct CommandAttrs {
    /// Shared attributes for fill commands, indexed by `attrs_idx` in `CmdFill`/`CmdAlphaFill`.
    pub fill: Vec<FillAttrs>,
    /// Shared attributes for clip commands, indexed by `attrs_idx` in `CmdClipAlphaFill`.
    pub clip: Vec<ClipAttrs>,
}

impl CommandAttrs {
    /// Clear all attributes.
    pub fn clear(&mut self) {
        self.fill.clear();
        self.clip.clear();
    }
}

/// Fill a consecutive horizontal region of a wide tile.
///
/// This command fills a rectangular region with the specified paint.
/// The region starts at x-coordinate `x` and extends for `width` pixels
/// horizontally, spanning the full height of the wide tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmdFill {
    /// The horizontal start position relative to the wide tile's left edge, in pixels.
    pub x: u16,
    /// The width of the filled region in pixels.
    pub width: u16,
    /// Index into the command attributes array.
    pub attrs_idx: u32,
}

/// Fill a consecutive horizontal region with an alpha mask.
///
/// Similar to `CmdFill`, but modulates the paint by an alpha mask stored
/// in a separate buffer. This is used for anti-aliased edges and partial
/// coverage from path rasterization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmdAlphaFill {
    /// The horizontal start position relative to the wide tile's left edge, in pixels.
    pub x: u16,
    /// The width of the filled region in pixels.
    pub width: u16,
    /// Relative offset to the alpha buffer location.
    /// Use `FillAttrs::alpha_idx(alpha_offset)` to compute the absolute index.
    pub alpha_offset: u32,
    /// Index into the command attributes array.
    pub attrs_idx: u32,
}

/// Fill operation within a clipping region.
///
/// This command copies a horizontal region from the top of the clip buffer stack
/// to the next buffer on the stack, effectively rendering the clipped content.
/// Unlike `CmdFill`, this doesn't fill with a paint but transfers existing content.
#[derive(Debug, PartialEq, Eq)]
pub struct CmdClipFill {
    /// The horizontal start position relative to the wide tile's left edge, in pixels.
    pub x: u16,
    /// The width of the region to copy in pixels.
    pub width: u16,
}

/// Alpha-masked fill operation within a clipping region.
///
/// This command composites a horizontal region from the top of the clip buffer stack
/// to the next buffer, modulated by an alpha mask. This is used for anti-aliased
/// clip edges.
#[derive(Debug, PartialEq, Eq)]
pub struct CmdClipAlphaFill {
    /// The horizontal start position relative to the wide tile's left edge, in pixels.
    pub x: u16,
    /// The width of the region to composite in pixels.
    pub width: u16,
    /// Relative offset to the alpha buffer location.
    /// Use `ClipAttrs::alpha_idx(alpha_offset)` to compute the absolute index.
    pub alpha_offset: u32,
    /// Index into the clip attributes array.
    pub attrs_idx: u32,
}

trait BlendModeExt {
    /// Whether a blend mode might cause destructive changes in the backdrop.
    /// This disallows certain optimizations (like for example inlining a blend mode
    /// or only applying a blend mode to the current clipping area).
    fn is_destructive(&self) -> bool;
}

impl BlendModeExt for BlendMode {
    fn is_destructive(&self) -> bool {
        matches!(
            self.compose,
            Compose::Clear
                | Compose::Copy
                | Compose::SrcIn
                | Compose::DestIn
                | Compose::SrcOut
                | Compose::DestAtop
        )
    }
}

/// Ranges of commands for a specific layer in a specific tile.
///
/// This structure tracks two different ranges of commands:
/// - The full range includes all layer operations (push, draw, pop)
/// - The render range includes only the actual drawing commands
#[derive(Debug, Clone, Default)]
pub struct LayerCommandRanges {
    /// Full range including `PushBuf`, all commands, and `PopBuf`.
    pub full_range: Range<usize>,
    /// Range containing only fill commands (`Fill`, `AlphaFill`).
    /// This is the range to replace when sampling from a filtered layer.
    pub render_range: Range<usize>,
}

impl LayerCommandRanges {
    /// Clear the full range and render range.
    #[inline]
    pub fn clear(&mut self) {
        self.full_range = 0..0;
        self.render_range = 0..0;
    }
}

#[cfg(test)]
mod tests {
    use crate::coarse::{FillHint, LayerKind, MODE_CPU, Wide, WideTile};
    use crate::kurbo::Affine;
    use crate::peniko::{BlendMode, Compose, Mix};
    use crate::render_graph::RenderGraph;
    use crate::strip::Strip;
    use alloc::{boxed::Box, vec};

    #[test]
    fn optimize_empty_layers() {
        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.pop_buf();

        assert!(wide.cmds.is_empty());
    }

    #[test]
    fn basic_layer() {
        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.fill(0, 10, 0, 0, FillHint::None);
        wide.fill(10, 10, 0, 0, FillHint::None);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 4);
    }

    #[test]
    fn dont_inline_blend_with_two_fills() {
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::SrcOver);

        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.fill(0, 10, 0, 0, FillHint::None);
        wide.fill(10, 10, 0, 0, FillHint::None);
        wide.blend(blend_mode);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 5);
    }

    #[test]
    fn dont_inline_destructive_blend() {
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::Clear);

        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.fill(0, 10, 0, 0, FillHint::None);
        wide.blend(blend_mode);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 4);
    }

    #[test]
    fn tile_coordinates() {
        let wide = Wide::<MODE_CPU>::new(1000, 258);

        let tile_1 = wide.get(1, 3);
        assert_eq!(tile_1.x, 256);
        assert_eq!(tile_1.y, 12);

        let tile_2 = wide.get(2, 15);
        assert_eq!(tile_2.x, 512);
        assert_eq!(tile_2.y, 60);
    }

    #[test]
    fn reset_clears_layer_and_clip_stacks() {
        type ClipPath = Option<Box<[Strip]>>;

        let mut wide = Wide::<MODE_CPU>::new(1000, 258);
        let mut render_graph = RenderGraph::new();
        let no_clip_path: ClipPath = None;
        wide.push_layer(
            1,
            no_clip_path,
            BlendMode::default(),
            None,
            0.5,
            None,
            Affine::IDENTITY,
            &mut render_graph,
            0,
        );

        assert_eq!(wide.layer_stack.len(), 1);
        assert_eq!(wide.clip_stack.len(), 0);

        let strip = Strip::new(2, 2, 0, true);
        let clip_path = Some(vec![strip].into_boxed_slice());
        wide.push_layer(
            2,
            clip_path,
            BlendMode::default(),
            None,
            0.09,
            None,
            Affine::IDENTITY,
            &mut render_graph,
            0,
        );

        assert_eq!(wide.layer_stack.len(), 2);
        assert_eq!(wide.clip_stack.len(), 1);

        wide.reset();

        assert_eq!(wide.layer_stack.len(), 0);
        assert_eq!(wide.clip_stack.len(), 0);
    }
}
