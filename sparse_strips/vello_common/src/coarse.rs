// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use crate::color::palette::css::TRANSPARENT;
use crate::mask::Mask;
use crate::paint::{Paint, PremulColor};
use crate::peniko::{BlendMode, Compose, Fill, Mix};
use crate::{strip::Strip, tile::Tile};
use alloc::vec;
use alloc::{boxed::Box, vec::Vec};

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
}

impl Layer {
    /// Whether the layer actually requires allocating a new scratch buffer
    /// for drawing its contents.
    fn needs_buf(&self) -> bool {
        self.blend_mode.mix != Mix::Normal
            || self.blend_mode.compose != Compose::SrcOver
            || self.opacity != 1.0
            || self.mask.is_some()
            || !self.clip
    }
}

/// A container for wide tiles.
#[derive(Debug)]
pub struct Wide {
    /// The width of the container.
    pub width: u16,
    /// The height of the container.
    pub height: u16,
    /// The wide tiles in the container.
    pub tiles: Vec<WideTile>,
    /// The stack of layers.
    layer_stack: Vec<Layer>,
    /// The stack of active clip regions.
    clip_stack: Vec<Clip>,
}

/// A clip region.
#[derive(Debug)]
struct Clip {
    /// The intersected bounding box after clip
    pub clip_bbox: Bbox,
    /// The rendered path in sparse strip representation
    pub strips: Box<[Strip]>,
    /// The fill rule used for this clip
    pub fill_rule: Fill,
}

/// A bounding box
///
/// The first two values represent the x0 and y0 coordinates, respectively.
/// The last two values represent the x1 and y1 coordinates, respectively.
///  x0, y0 — the top-left corner of the bounding box,
///  x1, y1 — the bottom-right corner of the bounding box.   
#[derive(Debug, Clone)]
struct Bbox {
    pub bbox: [u16; 4],
}

impl Bbox {
    pub(crate) fn new(bbox: [u16; 4]) -> Self {
        Self { bbox }
    }

    /// Get the x0 coordinate of the bounding box.
    #[inline]
    pub(crate) fn x0(&self) -> u16 {
        self.bbox[0]
    }

    /// Get the y0 coordinate of the bounding box.
    #[inline]
    pub(crate) fn y0(&self) -> u16 {
        self.bbox[1]
    }

    /// Get the x1 coordinate of the bounding box.
    #[inline]
    pub(crate) fn x1(&self) -> u16 {
        self.bbox[2]
    }

    /// Get the y1 coordinate of the bounding box.
    #[inline]
    pub(crate) fn y1(&self) -> u16 {
        self.bbox[3]
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
}

impl Wide {
    /// Create a new container for wide tiles.
    pub fn new(width: u16, height: u16) -> Self {
        let width_tiles = width.div_ceil(WideTile::WIDTH);
        let height_tiles = height.div_ceil(Tile::HEIGHT);
        let mut tiles = Vec::with_capacity(usize::from(width_tiles) * usize::from(height_tiles));

        for h in 0..height_tiles {
            for w in 0..width_tiles {
                tiles.push(WideTile::new(w * WideTile::WIDTH, h * Tile::HEIGHT));
            }
        }

        Self {
            tiles,
            width,
            height,
            layer_stack: vec![],
            clip_stack: vec![],
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
        }
        self.layer_stack.clear();
        self.clip_stack.clear();
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
    pub fn get(&self, x: u16, y: u16) -> &WideTile {
        assert!(
            x < self.width_tiles() && y < self.height_tiles(),
            "attempted to access out-of-bounds wide tile"
        );

        &self.tiles[usize::from(y) * usize::from(self.width_tiles()) + usize::from(x)]
    }

    /// Get mutable access to the wide tile at a certain index.
    ///
    /// Panics if the index is out-of-range.
    pub fn get_mut(&mut self, x: u16, y: u16) -> &mut WideTile {
        assert!(
            x < self.width_tiles() && y < self.height_tiles(),
            "attempted to access out-of-bounds wide tile"
        );

        let idx = usize::from(y) * usize::from(self.width_tiles()) + usize::from(x);
        &mut self.tiles[idx]
    }

    /// Return a reference to all wide tiles.
    pub fn tiles(&self) -> &[WideTile] {
        self.tiles.as_slice()
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
    pub fn generate(&mut self, strip_buf: &[Strip], fill_rule: Fill, paint: Paint) {
        if strip_buf.is_empty() {
            return;
        }

        // Get current clip bounding box or full viewport if no clip is active
        let bbox = self.get_bbox();

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
            let mut col = strip.alpha_idx / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx / u32::from(Tile::HEIGHT);
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
                    paint: paint.clone(),
                    blend_mode: None,
                };
                x += width;
                col += u32::from(width);
                self.get_mut(wtile_x, strip_y).strip(cmd);
            }

            // Determine if the region between this strip and the next should be filled
            // based on the fill rule (NonZero or EvenOdd)
            let active_fill = match fill_rule {
                Fill::NonZero => next_strip.winding != 0,
                Fill::EvenOdd => next_strip.winding % 2 != 0,
            };

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
                    self.get_mut(wtile_x, strip_y)
                        .fill(x_wtile_rel, width, paint.clone());
                }
            }
        }
    }

    /// Push a new layer with the given properties.
    pub fn push_layer(
        &mut self,
        clip_path: Option<(impl Into<Box<[Strip]>>, Fill)>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
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

        let layer = Layer {
            clip: clip_path.is_some(),
            blend_mode,
            opacity,
            mask,
        };

        let needs_buf = layer.needs_buf();

        // In case we do blending, masking or opacity, push one buffer per wide tile.
        if needs_buf {
            for x in 0..self.width_tiles() {
                for y in 0..self.height_tiles() {
                    self.get_mut(x, y).push_buf();
                }
            }
        }

        // If we have a clip path, push another buffer in the affected wide tiles.
        // Note that it is important that we FIRST push the buffer for blending etc. and
        // only then for clipping, otherwise we will use the empty clip buffer as the backdrop
        // for blending!
        if let Some((clip, fill)) = clip_path {
            self.push_clip(clip, fill);
        }

        self.layer_stack.push(layer);
    }

    /// Pop a previously pushed layer.
    pub fn pop_layer(&mut self) {
        // This method basically unwinds everything we did in `push_layer`.

        let layer = self.layer_stack.pop().unwrap();

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
    pub fn push_clip(&mut self, strips: impl Into<Box<[Strip]>>, fill_rule: Fill) {
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
                    ((next_strip.alpha_idx - strip.alpha_idx) / u32::from(Tile::HEIGHT)) as u16;
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

            // If we've reached the bottom of the clip bounding box, stop processing
            if cur_wtile_y == clip_bbox.y1() {
                break;
            }

            // Process wide tiles to the left of this strip in the same row
            let x = strip.x;
            let wtile_x_clamped = (x / WideTile::WIDTH).min(clip_bbox.x1());
            if cur_wtile_x < wtile_x_clamped {
                // If winding is zero or doesn't match fill rule, these wide tiles are outside the path
                let is_inside = match fill_rule {
                    Fill::NonZero => strip.winding != 0,
                    Fill::EvenOdd => strip.winding % 2 != 0,
                };
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
            let width = ((next_strip.alpha_idx - strip.alpha_idx) / u32::from(Tile::HEIGHT)) as u16;
            let wtile_x1 = (x + width).div_ceil(WideTile::WIDTH).min(clip_bbox.x1());
            if cur_wtile_x < wtile_x1 {
                for wtile_x in cur_wtile_x..wtile_x1 {
                    self.get_mut(wtile_x, cur_wtile_y).push_clip();
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

        self.clip_stack.push(Clip {
            clip_bbox,
            strips,
            fill_rule,
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
            fill_rule,
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
            if cur_wtile_y == clip_bbox.y1() {
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
                let is_inside = match fill_rule {
                    Fill::NonZero => strip.winding != 0,
                    Fill::EvenOdd => strip.winding % 2 != 0,
                };
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
                ((next_strip.alpha_idx - strip.alpha_idx) / u32::from(Tile::HEIGHT)) as u16;
            let x1 = x0 + strip_width;
            let wtile_x0 = (x0 / WideTile::WIDTH).max(clip_bbox.x0());
            let wtile_x1 = x1.div_ceil(WideTile::WIDTH).min(clip_bbox.x1());

            // Calculate starting position and column for alpha mask
            let mut x = x0;
            let mut col = strip.alpha_idx / u32::from(Tile::HEIGHT);
            let clip_x = clip_bbox.x0() * WideTile::WIDTH;
            if clip_x > x {
                col += u32::from(clip_x - x);
                x = clip_x;
            }

            // Render clip strips for each affected tile and mark for popping
            for wtile_x in wtile_x0..wtile_x1 {
                // If we've moved past tile_x and have a pending pop, do it now
                if cur_wtile_x < wtile_x && core::mem::take(&mut pop_pending) {
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                }

                // Calculate the portion of the strip that affects this tile
                let x_rel = u32::from(x % WideTile::WIDTH);
                let width = x1.min((wtile_x + 1) * WideTile::WIDTH) - x;

                // Create clip strip command for rendering the partial coverage
                let cmd = CmdClipAlphaFill {
                    x: x_rel,
                    width: u32::from(width),
                    alpha_idx: col as usize * Tile::HEIGHT as usize,
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
            let is_inside = match fill_rule {
                Fill::NonZero => next_strip.winding != 0,
                Fill::EvenOdd => next_strip.winding % 2 != 0,
            };
            if is_inside && strip_y == next_strip.strip_y() {
                if cur_wtile_x >= clip_bbox.x1() {
                    continue;
                }

                let x2 = next_strip.x;
                let clipped_x2 = x2.min((cur_wtile_x + 1) * WideTile::WIDTH);
                let width = clipped_x2.saturating_sub(x1);

                // If there's a gap, fill it
                if width > 0 {
                    let x_rel = u32::from(x1 % WideTile::WIDTH);
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
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                    pop_pending = false;

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
pub struct WideTile {
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
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;

    /// Create a new wide tile.
    pub fn new(x: u16, y: u16) -> Self {
        Self {
            x,
            y,
            bg: PremulColor::from_alpha_color(TRANSPARENT),
            cmds: vec![],
            n_zero_clip: 0,
            n_clip: 0,
            n_bufs: 0,
        }
    }

    pub(crate) fn fill(&mut self, x: u16, width: u16, paint: Paint) {
        if !self.is_zero_clip() {
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
                    && width == Self::WIDTH
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
            } else {
                self.cmds.push(Cmd::Fill(CmdFill {
                    x,
                    width,
                    paint,
                    blend_mode: None,
                }));
            }
        }
    }

    pub(crate) fn strip(&mut self, cmd_strip: CmdAlphaFill) {
        if !self.is_zero_clip() {
            self.cmds.push(Cmd::AlphaFill(cmd_strip));
        }
    }

    /// Adds a new clip region to the current wide tile.
    pub fn push_clip(&mut self) {
        if !self.is_zero_clip() {
            self.push_buf();
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
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushBuf)) {
            self.cmds.push(Cmd::ClipStrip(cmd_clip_strip));
        }
    }

    /// Applies a clip fill operation at the specified position and width.
    pub fn clip_fill(&mut self, x: u32, width: u32) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushBuf)) {
            self.cmds.push(Cmd::ClipFill(CmdClipFill { x, width }));
        }
    }

    /// Push a buffer.
    pub fn push_buf(&mut self) {
        self.cmds.push(Cmd::PushBuf);
        self.n_bufs += 1;
    }

    /// Pop the most recent buffer.
    pub fn pop_buf(&mut self) {
        if matches!(self.cmds.last(), Some(&Cmd::PushBuf)) {
            // Optimization: If no drawing happened between the last `PushBuf`,
            // we can just pop it instead.
            self.cmds.pop();
        } else {
            if self.cmds.len() >= 3 {
                // If we have a non-destructive blend mode with just a single fill/strip,
                // inline the blend mode instead.
                let (_, tail) = self.cmds.split_at(self.cmds.len() - 3);

                let updated = match tail {
                    [Cmd::PushBuf, Cmd::AlphaFill(a), Cmd::Blend(b)] => {
                        if !b.is_destructive() && a.blend_mode.is_none() {
                            let mut blended = a.clone();
                            blended.blend_mode = Some(*b);
                            Some(Cmd::AlphaFill(blended))
                        } else {
                            None
                        }
                    }
                    [Cmd::PushBuf, Cmd::Fill(a), Cmd::Blend(b)] => {
                        if !b.is_destructive() && a.blend_mode.is_none() {
                            let mut blended = a.clone();
                            blended.blend_mode = Some(*b);
                            Some(Cmd::Fill(blended))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(updated) = updated {
                    self.cmds.pop();
                    self.cmds.pop();
                    self.cmds.pop();
                    self.cmds.push(updated);

                    return;
                }
            }

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

    /// Apply a mask to the whole buffer.
    pub fn mask(&mut self, mask: Mask) {
        self.cmds.push(Cmd::Mask(mask));
    }

    /// Blend the current buffer into the previous buffer in the stack.
    pub fn blend(&mut self, blend_mode: BlendMode) {
        // Optimization: If no drawing happened since the last `PushBuf` and the blend mode
        // is not destructive, we do not need to do any blending at all.
        if !matches!(self.cmds.last(), Some(&Cmd::PushBuf)) || blend_mode.is_destructive() {
            self.cmds.push(Cmd::Blend(blend_mode));
        }
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
    PushBuf,
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
    use crate::coarse::{Cmd, CmdFill, Wide, WideTile};
    use crate::color::AlphaColor;
    use crate::color::palette::css::TRANSPARENT;
    use crate::paint::{Paint, PremulColor};
    use crate::peniko::{BlendMode, Compose, Fill, Mix};
    use crate::strip::Strip;
    use alloc::{boxed::Box, vec};

    #[test]
    fn optimize_empty_layers() {
        let mut wide = WideTile::new(0, 0);
        wide.push_buf();
        wide.pop_buf();

        assert!(wide.cmds.is_empty());
    }

    #[test]
    fn basic_layer() {
        let mut wide = WideTile::new(0, 0);
        wide.push_buf();
        wide.fill(
            0,
            10,
            Paint::Solid(PremulColor::from_alpha_color(TRANSPARENT)),
        );
        wide.fill(
            10,
            10,
            Paint::Solid(PremulColor::from_alpha_color(TRANSPARENT)),
        );
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 4);
    }

    #[test]
    fn inline_blend_with_one_fill() {
        let paint = Paint::Solid(PremulColor::from_alpha_color(AlphaColor::from_rgba8(
            30, 30, 30, 255,
        )));
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::SrcOver);

        let mut wide = WideTile::new(0, 0);
        wide.push_buf();
        wide.fill(0, 10, paint.clone());
        wide.blend(blend_mode);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 1);

        let expected = Cmd::Fill(CmdFill {
            x: 0,
            width: 10,
            paint,
            blend_mode: Some(blend_mode),
        });

        assert_eq!(wide.cmds[0], expected);
    }

    #[test]
    fn dont_inline_blend_with_two_fills() {
        let paint = Paint::Solid(PremulColor::from_alpha_color(AlphaColor::from_rgba8(
            30, 30, 30, 255,
        )));
        let blend_mode = BlendMode::new(Mix::Lighten, Compose::SrcOver);

        let mut wide = WideTile::new(0, 0);
        wide.push_buf();
        wide.fill(0, 10, paint.clone());
        wide.fill(10, 10, paint.clone());
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

        let mut wide = WideTile::new(0, 0);
        wide.push_buf();
        wide.fill(0, 10, paint.clone());
        wide.blend(blend_mode);
        wide.pop_buf();

        assert_eq!(wide.cmds.len(), 4);
    }

    #[test]
    fn tile_coordinates() {
        let wide = Wide::new(1000, 258);

        let tile_1 = wide.get(1, 3);
        assert_eq!(tile_1.x, 256);
        assert_eq!(tile_1.y, 12);

        let tile_2 = wide.get(2, 15);
        assert_eq!(tile_2.x, 512);
        assert_eq!(tile_2.y, 60);
    }

    #[test]
    fn reset_clears_layer_and_clip_stacks() {
        type ClipPath = Option<(Box<[Strip]>, Fill)>;

        let mut wide = Wide::new(1000, 258);
        let no_clip_path: ClipPath = None;
        wide.push_layer(no_clip_path, BlendMode::default(), None, 0.5);

        assert_eq!(wide.layer_stack.len(), 1);
        assert_eq!(wide.clip_stack.len(), 0);

        let strip = Strip {
            x: 2,
            y: 2,
            alpha_idx: 0,
            winding: 1,
        };
        let clip_path = Some((vec![strip].into_boxed_slice(), Fill::NonZero));
        wide.push_layer(clip_path, BlendMode::default(), None, 0.09);

        assert_eq!(wide.layer_stack.len(), 2);
        assert_eq!(wide.clip_stack.len(), 1);

        wide.reset();

        assert_eq!(wide.layer_stack.len(), 0);
        assert_eq!(wide.clip_stack.len(), 0);
    }
}
