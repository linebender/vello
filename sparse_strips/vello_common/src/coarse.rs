// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use core::ops::Range;
use std::collections::BTreeMap;

use crate::color::palette::css::TRANSPARENT;
use crate::filter_effects::Filter;
use crate::mask::Mask;
use crate::paint::{Paint, PremulColor};
use crate::peniko::{BlendMode, Compose, Mix};
use crate::render_graph::{DependencyKind, LayerId, RenderGraph, RenderNodeKind};
use crate::{strip::Strip, tile::Tile};
use alloc::vec;
use alloc::{boxed::Box, vec::Vec};

/// Ranges of commands for a specific layer in a specific tile.
#[derive(Debug, Clone, Default)]
pub struct LayerCommandRanges {
    /// Full range including PushBuf, all commands, and PopBuf
    pub full_range: Range<usize>,
    /// Range containing only fill commands (Fill, AlphaFill)
    /// This is the range to replace when sampling from filtered layer
    pub render_range: Range<usize>,
}

impl LayerCommandRanges {
    pub fn clear(&mut self) {
        self.full_range = 0..0;
        self.render_range = 0..0;
    }
}

#[derive(Debug)]
struct Layer {
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
    /// The layer ID of the layer.
    layer_id: LayerId,
    /// Bounding box of wide tiles containing geometry.
    /// Starts with inverted bounds, shrinks to actual content during drawing.
    wtile_bbox: Bbox,
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
    /// The stack of layers.
    layer_stack: Vec<Layer>,
    /// The stack of active clip regions.
    clip_stack: Vec<Clip>,
    /// Stack of filter layer node IDs for render graph dependency tracking.
    /// Initialized with node 0 (the root node).
    filter_node_stack: Vec<usize>,
}

/// A clip region.
#[derive(Debug)]
struct Clip {
    /// The intersected bounding box after clip
    pub clip_bbox: Bbox,
    /// The rendered path in sparse strip representation
    pub strips: Box<[Strip]>,
    #[cfg(feature = "multithreading")]
    pub thread_idx: u8,
}

/// A bounding box
///
/// The first two values represent the x0 and y0 coordinates, respectively.
/// The last two values represent the x1 and y1 coordinates, respectively.
///  x0, y0 — the top-left corner of the bounding box,
///  x1, y1 — the bottom-right corner of the bounding box.   
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub bbox: [u16; 4],
}

impl Bbox {
    pub fn new(bbox: [u16; 4]) -> Self {
        Self { bbox }
    }

    /// Get the x0 coordinate of the bounding box.
    #[inline]
    pub fn x0(&self) -> u16 {
        self.bbox[0]
    }

    /// Get the y0 coordinate of the bounding box.
    #[inline]
    pub fn y0(&self) -> u16 {
        self.bbox[1]
    }

    /// Get the x1 coordinate of the bounding box.
    #[inline]
    pub fn x1(&self) -> u16 {
        self.bbox[2]
    }

    /// Get the y1 coordinate of the bounding box.
    #[inline]
    pub fn y1(&self) -> u16 {
        self.bbox[3]
    }

    /// Get the width of the bounding box (x1 - x0).
    #[inline]
    pub fn width(&self) -> u16 {
        self.x1().saturating_sub(self.x0())
    }

    /// Get the height of the bounding box (y1 - y0).
    #[inline]
    pub fn height(&self) -> u16 {
        self.y1().saturating_sub(self.y0())
    }

    /// Check if a point (x, y) is contained within this bounding box.
    ///
    /// Returns `true` if x0 <= x < x1 and y0 <= y < y1.
    #[inline]
    pub fn contains(&self, x: u16, y: u16) -> bool {
        x >= self.x0() && x < self.x1() && y >= self.y0() && y < self.y1()
    }

    /// Create an empty bounding box (zero area).
    #[inline]
    pub(crate) fn empty() -> Self {
        Self::new([0, 0, 0, 0])
    }

    /// Calculate the intersection of this bounding box with another.
    #[inline]
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        Self::new([
            self.x0().max(other.x0()),
            self.y0().max(other.y0()),
            self.x1().min(other.x1()),
            self.y1().min(other.y1()),
        ])
    }

    /// Update this bounding box to include another bounding box (union in place).
    #[inline]
    pub(crate) fn include_bbox(&mut self, other: &Self) {
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
    /// Starts with max values for mins and min values for maxs,
    /// so first update will set correct bounds.
    #[inline]
    pub(crate) fn inverted() -> Self {
        Self::new([u16::MAX, u16::MAX, 0, 0])
    }

    /// Check if the bbox is still in its inverted state (no updates yet).
    #[inline]
    pub(crate) fn is_inverted(&self) -> bool {
        self.bbox[0] == u16::MAX
    }

    /// Update the bbox to include the given tile coordinates.
    #[inline]
    pub(crate) fn include_tile(&mut self, wtile_x: u16, wtile_y: u16) {
        self.bbox[0] = self.bbox[0].min(wtile_x); // min_x
        self.bbox[1] = self.bbox[1].min(wtile_y); // min_y
        self.bbox[2] = self.bbox[2].max(wtile_x + 1); // max_x
        self.bbox[3] = self.bbox[3].max(wtile_y + 1); // max_y
    }

    /// Scale this bounding box by the given scale factors.
    ///
    /// Multiplies each coordinate by the corresponding scale factor to convert
    /// from one coordinate system to another.
    #[inline]
    pub fn scale(&self, scale_x: u16, scale_y: u16) -> [u32; 4] {
        [
            u32::from(self.x0()) * u32::from(scale_x),
            u32::from(self.y0()) * u32::from(scale_y),
            u32::from(self.x1()) * u32::from(scale_x),
            u32::from(self.y1()) * u32::from(scale_y),
        ]
    }

    /// Expand this bounding box by the given pixel amounts in each direction.
    /// Converts pixels to wide tile coordinates and clamps to valid range.
    ///
    /// # Arguments
    /// * `left_px` - Pixels to expand leftward
    /// * `top_px` - Pixels to expand upward  
    /// * `right_px` - Pixels to expand rightward
    /// * `bottom_px` - Pixels to expand downward
    /// * `max_x` - Maximum X coordinate in wide tiles
    /// * `max_y` - Maximum Y coordinate in tile rows
    pub fn expand_by_pixels(
        &self,
        left_px: f32,
        top_px: f32,
        right_px: f32,
        bottom_px: f32,
        max_x: u16,
        max_y: u16,
    ) -> Self {
        // Convert pixel expansion to tile expansion (round up)
        let left_tiles = (left_px / WideTile::WIDTH as f32).ceil() as u16;
        let top_tiles = (top_px / Tile::HEIGHT as f32).ceil() as u16;
        let right_tiles = (right_px / WideTile::WIDTH as f32).ceil() as u16;
        let bottom_tiles = (bottom_px / Tile::HEIGHT as f32).ceil() as u16;

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
            layer_stack: vec![],
            clip_stack: vec![],
            filter_node_stack: vec![0], // Start with root node (node 0)
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
            tile.layer_ids.clear();
            tile.layer_ids.push(LayerKind::Regular(0));
        }
        self.layer_stack.clear();
        self.clip_stack.clear();
        self.filter_node_stack.clear();
        self.filter_node_stack.push(0); // Reset to root node
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

    /// Get the current layer ID.
    pub fn get_current_layer_id(&self) -> LayerId {
        self.layer_stack.last().map(|l| l.layer_id).unwrap_or(0)
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
    pub fn generate(&mut self, strip_buf: &[Strip], paint: Paint, thread_idx: u8) {
        if strip_buf.is_empty() {
            return;
        }

        // Prevent unused warning.
        let _ = thread_idx;

        // Get current clip bounding box or full viewport if no clip is active
        let bbox = self.get_bbox();

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
            let x1 = x0 + strip_width;

            // Calculate which wide tiles this strip intersects
            let wtile_x0 = (x0 / WideTile::WIDTH).max(bbox.x0());
            // It's possible that a strip extends into a new wide tile, but we don't actually
            // have as many wide tiles (e.g. because the pixmap width is only 512, but
            // strip ends at 513), so take the minimum between the rounded values and `width_tiles`.
            let wtile_x1 = x1.div_ceil(WideTile::WIDTH).min(bbox.x1());

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
                    alpha_idx: (col * u32::from(Tile::HEIGHT)) as usize,
                    #[cfg(feature = "multithreading")]
                    thread_idx,
                    paint: paint.clone(),
                    blend_mode: None,
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
                let x2 = next_strip
                    .x
                    .min(self.width.next_multiple_of(WideTile::WIDTH));
                let wfxt0 = (x1 / WideTile::WIDTH).max(bbox.x0());
                let wfxt1 = x2.div_ceil(WideTile::WIDTH).min(bbox.x1());

                // Generate fill commands for each wide tile in the fill region
                for wtile_x in wfxt0..wfxt1 {
                    let x_wtile_rel = x % WideTile::WIDTH;
                    let width = x2.min((wtile_x + 1) * WideTile::WIDTH) - x;
                    x += width;
                    self.get_mut(wtile_x, strip_y).fill(
                        x_wtile_rel,
                        width,
                        paint.clone(),
                        current_layer_id,
                    );
                    self.update_current_layer_bbox(wtile_x, strip_y);
                }
            }
        }
    }

    /// Push a new layer with the given properties.
    ///
    /// If `layer_id` is Some, rendering will be directed to that persistent layer storage.
    /// This is used for filter effects that require access to a fully-rendered layer.
    ///
    /// If `graph` is Some, builds render graph nodes for filter effects.
    pub fn push_layer(
        &mut self,
        layer_id: LayerId,
        clip_path: Option<impl Into<Box<[Strip]>>>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
        filter: Option<Filter>,
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

        // Build render graph node ONLY if we have a filter
        if let Some(filter) = &filter {
            // Create a single FilterLayout node that combines render + filter + blend
            let child_node = render_graph.add_node(RenderNodeKind::FilterLayer {
                layer_id,
                filter: filter.clone(),
                // Will be updated in pop_layer with actual bounds
                wtile_bbox: Bbox::inverted(),
            });

            // Connect to parent node if there is one
            if let Some(&parent_node) = self.filter_node_stack.last() {
                render_graph.add_edge(
                    child_node,
                    parent_node,
                    DependencyKind::DataDependency { layer_id },
                );
            }

            // Push this layout node onto the stack so next filter depends on it
            self.filter_node_stack.push(child_node);
        }

        // Determine layer kind before moving filter
        let has_filter = filter.is_some();

        let layer = Layer {
            layer_id,
            clip: clip_path.is_some(),
            blend_mode,
            opacity,
            mask,
            filter,
            wtile_bbox: Bbox::inverted(),
        };

        let needs_buf = layer.needs_buf();

        // Determine layer kind based on whether it has a filter
        let layer_kind = if has_filter {
            LayerKind::Filtered(layer_id)
        } else {
            LayerKind::Regular(layer_id)
        };

        // In case we do blending, masking or opacity, push one buffer per wide tile.
        if needs_buf {
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    self.get_mut(x, y).push_buf(layer_kind);
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
    /// If `graph` is Some, completes render graph nodes for filter effects.
    pub fn pop_layer(&mut self, render_graph: &mut RenderGraph) {
        // This method basically unwinds everything we did in `push_layer`.
        let mut layer = self.layer_stack.pop().unwrap();

        if let Some(filter) = &layer.filter {
            let expansion = filter.bounds_expansion();
            let expanded_bbox = layer.wtile_bbox.expand_by_pixels(
                expansion.left,
                expansion.top,
                expansion.right,
                expansion.bottom,
                self.width_tiles(),
                self.height_tiles(),
            );
            let clip_bbox = self.get_bbox();
            layer.wtile_bbox = expanded_bbox.intersect(&clip_bbox);
        }

        // Union this layer's bbox into the parent layer's bbox
        // This ensures parent knows about tiles used by this filtered child
        if let Some(parent_layer) = self.layer_stack.last_mut() {
            parent_layer.wtile_bbox.include_bbox(&layer.wtile_bbox);
        }

        // Update render graph node with final bounding box and propagate to parent
        if layer.filter.is_some() {
            if let Some(node_id) = self.filter_node_stack.pop() {
                // Update the render graph node with this layer's bbox
                if let Some(node) = render_graph.nodes.get_mut(node_id) {
                    if let RenderNodeKind::FilterLayer { wtile_bbox, .. } = &mut node.kind {
                        *wtile_bbox = layer.wtile_bbox;
                    }
                }
            }
        }

        // Non-graph path: use old method for filter execution
        // Apply filter BEFORE clipping (per SVG spec: filter → clip → mask → opacity → blend)
        if let Some(filter) = layer.filter.clone() {
            let layer_id = layer.layer_id;
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    self.get_mut(x, y).filter(layer_id, filter.clone());
                }
            }
        }

        if layer.clip {
            self.pop_clip();
        }

        let needs_buf = layer.needs_buf();

        if needs_buf {
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    let t = self.get_mut(x, y);

                    if let Some(mask) = layer.mask.clone() {
                        t.mask(mask);
                    }
                    t.opacity(layer.opacity);
                    t.blend(layer.blend_mode);
                    t.pop_buf();
                }
            }
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
            Bbox::empty()
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
            Bbox::new([wtile_x0, wtile_y0, wtile_x1, wtile_y1])
        };

        let parent_bbox = self.get_bbox();
        // Calculate the intersection of the parent clip bounding box and the path bounding box.
        let clip_bbox = parent_bbox.intersect(&path_bbox);

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
                    self.get_mut(wtile_x, cur_wtile_y).push_zero_clip();
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
                        self.get_mut(wtile_x, cur_wtile_y).push_zero_clip();
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
                self.get_mut(wtile_x, cur_wtile_y).push_zero_clip();
            }
            cur_wtile_x = clip_bbox.x0();
            cur_wtile_y += 1;
        }

        // Prevent unused warning.
        let _ = thread_idx;

        self.clip_stack.push(Clip {
            clip_bbox,
            strips,
            #[cfg(feature = "multithreading")]
            thread_idx,
        });
    }

    /// Get the bounding box of the current clip region or the entire viewport if no clip regions are active.
    fn get_bbox(&self) -> Bbox {
        if let Some(top) = self.clip_stack.last() {
            top.clip_bbox.clone()
        } else {
            // Convert pixel dimensions to wide tile coordinates
            Bbox::new([0, 0, self.width_tiles(), self.height_tiles()])
        }
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
            #[cfg(feature = "multithreading")]
            thread_idx,
        } = self.clip_stack.pop().unwrap();
        let n_strips = strips.len();

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
                let x_rel = u32::from(x % WideTile::WIDTH);
                let width = clipped_x1.min((wtile_x + 1) * WideTile::WIDTH) - x;

                // Create clip strip command for rendering the partial coverage
                let cmd = CmdClipAlphaFill {
                    x: x_rel,
                    width: u32::from(width),
                    alpha_idx: col as usize * Tile::HEIGHT as usize,
                    #[cfg(feature = "multithreading")]
                    thread_idx,
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
                    let x_rel = u32::from(clipped_x1 % WideTile::WIDTH);
                    self.get_mut(cur_wtile_x, cur_wtile_y)
                        .clip_fill(x_rel, u32::from(width));
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
                        self.get_mut(cur_wtile_x, cur_wtile_y)
                            .clip_fill(0, u32::from(width2));
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
    /// Maps layer ID to command ranges for this tile.
    pub layer_cmd_ranges: BTreeMap<LayerId, LayerCommandRanges>,
    /// Vector of layer IDs this tile participates in.
    pub layer_ids: Vec<LayerKind>,
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;
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
        let mut layer_cmd_ranges = BTreeMap::new();
        layer_cmd_ranges.insert(0, LayerCommandRanges::default());
        Self {
            x,
            y,
            bg: PremulColor::from_alpha_color(TRANSPARENT),
            cmds: vec![],
            n_zero_clip: 0,
            n_clip: 0,
            n_bufs: 0,
            layer_cmd_ranges,
            layer_ids: vec![LayerKind::Regular(0)],
        }
    }

    pub(crate) fn fill(&mut self, x: u16, width: u16, paint: Paint, current_layer_id: LayerId) {
        if !self.is_zero_clip() {
            match MODE {
                MODE_CPU => {
                    let bg = if let Paint::Solid(s) = &paint {
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
                        let can_override = x == 0
                            && width == WideTile::WIDTH
                            && s.is_opaque()
                            && self.n_clip == 0
                            && self.n_bufs == 0;
                        can_override.then_some(*s)
                    } else {
                        // TODO: Implement for indexed paints.
                        None
                    };

                    if let Some(bg) = bg {
                        self.cmds.clear();
                        self.bg = bg;
                        // Clear layer ranges when we clear commands
                        if let Some(ranges) = self.layer_cmd_ranges.get_mut(&current_layer_id) {
                            ranges.clear();
                        }
                    } else {
                        self.record_fill_cmd(current_layer_id, self.cmds.len());
                        self.cmds.push(Cmd::Fill(CmdFill {
                            x,
                            width,
                            paint,
                            blend_mode: None,
                        }));
                    }
                }
                MODE_HYBRID => {
                    self.record_fill_cmd(current_layer_id, self.cmds.len());
                    self.cmds.push(Cmd::Fill(CmdFill {
                        x,
                        width,
                        paint,
                        blend_mode: None,
                    }));
                }
                _ => unreachable!(),
            }
        }
    }

    pub(crate) fn strip(&mut self, cmd_strip: CmdAlphaFill, current_layer_id: LayerId) {
        if !self.is_zero_clip() {
            self.record_fill_cmd(current_layer_id, self.cmds.len());
            self.cmds.push(Cmd::AlphaFill(cmd_strip));
        }
    }

    /// Adds a new clip region to the current wide tile.
    pub fn push_clip(&mut self, layer_id: LayerId) {
        if !self.is_zero_clip() {
            self.push_buf(LayerKind::Clip(layer_id));
            self.n_clip += 1;
        }
    }

    /// Removes the most recently added clip region from the current wide tile.
    pub fn pop_clip(&mut self) {
        if !self.is_zero_clip() {
            self.pop_buf();
            self.n_clip -= 1;
        }
    }

    /// Adds a zero-winding clip region to the stack.
    pub fn push_zero_clip(&mut self) {
        self.n_zero_clip += 1;
    }

    /// Removes the most recently added zero-winding clip region.
    pub fn pop_zero_clip(&mut self) {
        self.n_zero_clip -= 1;
    }

    /// Checks if the current clip region is a zero-winding clip.
    pub fn is_zero_clip(&mut self) -> bool {
        self.n_zero_clip > 0
    }

    /// Applies a clip strip operation with the given parameters.
    pub fn clip_strip(&mut self, cmd_clip_strip: CmdClipAlphaFill) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushBuf(_))) {
            self.cmds.push(Cmd::ClipStrip(cmd_clip_strip));
        }
    }

    /// Applies a clip fill operation at the specified position and width.
    pub fn clip_fill(&mut self, x: u32, width: u32) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushBuf(_))) {
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

    /// Push a buffer.
    ///
    /// Regular layers use local blend_buf stack.
    /// Filtered layers are materialized in persistent layer storage.
    pub fn push_buf(&mut self, layer_kind: LayerKind) {
        let top_layer = layer_kind.id();
        let next_layer = self.layer_ids.last().unwrap().id();
        if matches!(layer_kind, LayerKind::Filtered(_)) {
            self.layer_cmd_ranges.insert(
                top_layer,
                LayerCommandRanges {
                    full_range: self.cmds.len()..self.cmds.len() + 1,
                    render_range: self.cmds.len() + 1..self.cmds.len() + 1,
                },
            );
        } else if matches!(layer_kind, LayerKind::Clip(_)) {
            self.layer_cmd_ranges.entry(top_layer).and_modify(|ranges| {
                ranges.full_range.end = self.cmds.len() + 1;
                ranges.render_range = self.cmds.len() + 1..self.cmds.len() + 1;
            });
        }
        self.cmds.push(Cmd::PushBuf(layer_kind));
        self.layer_ids.push(layer_kind);
        self.n_bufs += 1;
    }

    /// Pop the most recent buffer.
    pub fn pop_buf(&mut self) {
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

            // ranges.render_range.end = self.cmds.len() + 1;
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
    pub fn opacity(&mut self, opacity: f32) {
        if opacity != 1.0 {
            self.cmds.push(Cmd::Opacity(opacity));
        }
    }

    /// Apply a filter effect to the whole buffer.
    pub fn filter(&mut self, layer_id: LayerId, filter: Filter) {
        self.cmds.push(Cmd::Filter(layer_id, filter));
    }

    /// Apply a mask to the whole buffer.
    pub fn mask(&mut self, mask: Mask) {
        self.cmds.push(Cmd::Mask(mask));
    }

    /// Blend the current buffer into the previous buffer in the stack.
    pub fn blend(&mut self, blend_mode: BlendMode) {
        // Optimization: If no drawing happened since the last `PushBuf` and the blend mode
        // is not destructive, we do not need to do any blending at all.
        if !matches!(self.cmds.last(), Some(&Cmd::PushBuf(_))) || blend_mode.is_destructive() {
            self.cmds.push(Cmd::Blend(blend_mode));
        }
    }
}

/// Distinguishes between regular layers and filtered layers that are materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Regular layer using local blend_buf stack
    Regular(LayerId),
    /// Filtered layer materialized in layer_manager
    Filtered(LayerId),
    /// Clip layer
    Clip(LayerId),
}

impl LayerKind {
    /// Get the underlying layer ID
    pub fn id(&self) -> LayerId {
        match self {
            LayerKind::Regular(id) | LayerKind::Filtered(id) | LayerKind::Clip(id) => *id,
        }
    }

    /// Check if this is a filtered layer
    pub fn is_filtered(&self) -> bool {
        matches!(self, LayerKind::Filtered(_))
    }
}

/// A drawing command.
#[derive(Debug, PartialEq)]
pub enum Cmd {
    /// A fill command.
    Fill(CmdFill),
    /// A fill command with alpha mask.
    AlphaFill(CmdAlphaFill),
    /// Pushes a new buffer for drawing.
    /// Regular layers use the local `blend_buf` stack.
    /// Filtered layers are materialized in persistent layer storage.
    PushBuf(LayerKind),
    /// Pops the most recent buffer.
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
    /// Apply a filter effect to the current buffer.
    ///
    /// This command will apply a filter (e.g., blur) to the contents of the current buffer.
    /// Filters are applied before clipping, masking, blending, and opacity.
    Filter(LayerId, Filter),
    /// Apply a blend.
    ///
    /// This command will blend the contents of the current buffer into the previous buffer in
    /// the stack.
    Blend(BlendMode),
    /// Apply an opacity mask to the current buffer.
    Opacity(f32),
    /// Apply a mask to the current buffer.
    Mask(Mask),
}

/// Fill a consecutive region of a wide tile.
#[derive(Debug, Clone, PartialEq)]
pub struct CmdFill {
    /// The horizontal start position of the command in pixels.
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
    /// The blend mode to apply before drawing the contents.
    pub blend_mode: Option<BlendMode>,
}

/// Fill a consecutive region of a wide tile with an alpha mask.
#[derive(Debug, Clone, PartialEq)]
pub struct CmdAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The start index into the alpha buffer of the command.
    pub alpha_idx: usize,
    /// The index of the thread that contains the alpha values
    /// pointed to by `alpha_idx`.
    #[cfg(feature = "multithreading")]
    pub thread_idx: u8,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
    /// A blend mode to apply before drawing the contents.
    pub blend_mode: Option<BlendMode>,
}

/// Same as fill, but copies top of clip stack to next on stack.
#[derive(Debug, PartialEq, Eq)]
pub struct CmdClipFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
}

/// Same as strip, but composites top of clip stack to next on stack.
#[derive(Debug, PartialEq, Eq)]
pub struct CmdClipAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
    /// The index of the thread that contains the alpha values
    /// pointed to by `alpha_idx`.
    #[cfg(feature = "multithreading")]
    pub thread_idx: u8,
    /// The start index into the alpha buffer of the command.
    pub alpha_idx: usize,
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

#[cfg(test)]
mod tests {
    use crate::coarse::{LayerKind, MODE_CPU, Wide, WideTile};
    use crate::color::AlphaColor;
    use crate::color::palette::css::TRANSPARENT;
    use crate::paint::{Paint, PremulColor};
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
        wide.fill(
            0,
            10,
            Paint::Solid(PremulColor::from_alpha_color(TRANSPARENT)),
            0,
        );
        wide.fill(
            10,
            10,
            Paint::Solid(PremulColor::from_alpha_color(TRANSPARENT)),
            0,
        );
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 4);
    }

    #[test]
    fn dont_inline_blend_with_two_fills() {
        let paint = Paint::Solid(PremulColor::from_alpha_color(AlphaColor::from_rgba8(
            30, 30, 30, 255,
        )));
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::SrcOver);

        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.fill(0, 10, paint.clone(), 0);
        wide.fill(10, 10, paint.clone(), 0);
        wide.blend(blend_mode);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 5);
    }

    #[test]
    fn dont_inline_destructive_blend() {
        let paint = Paint::Solid(PremulColor::from_alpha_color(AlphaColor::from_rgba8(
            30, 30, 30, 255,
        )));
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::Clear);

        let mut wide = WideTile::<MODE_CPU>::new(0, 0);
        wide.push_buf(LayerKind::Regular(0));
        wide.fill(0, 10, paint.clone(), 0);
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
