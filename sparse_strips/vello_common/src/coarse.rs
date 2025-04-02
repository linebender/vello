// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generating and processing wide tiles.

use crate::{
    color::{AlphaColor, Srgb},
    strip::Strip,
    tile::Tile,
};
use vello_api::color::PremulRgba8;
use vello_api::{paint::Paint, peniko::Fill};

/// A container for wide tiles.
#[derive(Debug)]
pub struct Wide {
    /// The width of the container.
    pub width: u16,
    /// The height of the container.
    pub height: u16,
    /// The wide tiles in the container.
    pub tiles: Vec<WideTile>,

    /// Stack of scene state, used for tracking clip regions.
    pub state_stack: Vec<SceneState>,
    /// Stack of active clip regions.
    clip_stack: Vec<Clip>,
}

/// Scene state for rendering operations.
#[derive(Debug)]
pub struct SceneState {
    /// Number of active clip regions.
    pub n_clip: usize,
}

/// A clip region.
#[derive(Debug)]
struct Clip {
    /// The intersected bounding box after clip
    pub clip_bbox: Bbox,
    /// The rendered path in sparse strip representation
    pub strips: Vec<Strip>,
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
        let mut tiles = Vec::with_capacity(usize::from(width_tiles * height_tiles));

        for w in 0..width_tiles {
            for h in 0..height_tiles {
                tiles.push(WideTile::new(w * WideTile::WIDTH, h * Tile::HEIGHT));
            }
        }

        Self {
            tiles,
            width,
            height,
            state_stack: vec![SceneState { n_clip: 0 }],
            clip_stack: vec![],
        }
    }

    /// Reset all tiles in the container.
    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.bg = AlphaColor::<Srgb>::TRANSPARENT.premultiply().to_rgba8();
            tile.cmds.clear();
        }
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
                col += (clip_x - x) as u32;
                x = clip_x;
            }

            // Generate alpha fill commands for each wide tile intersected by this strip
            for wtile_x in wtile_x0..wtile_x1 {
                let x_wtile_rel = x % WideTile::WIDTH;
                let width = x1.min((wtile_x + 1) * WideTile::WIDTH) - x;
                let cmd = CmdAlphaFill {
                    x: x_wtile_rel,
                    width,
                    alpha_idx: (col * u32::from(Tile::HEIGHT)) as usize,
                    paint: paint.clone(),
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
                x = x1;
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
    pub fn push_clip(&mut self, strips: Vec<Strip>, fill_rule: Fill) {
        let n_strips = strips.len();

        // Calculate the bounding box of the clip path in strip coordinates
        let path_bbox = if n_strips <= 1 {
            Bbox::empty()
        } else {
            // Calculate the y range from first to last strip in wide tile coordinates
            let wtile_y0 = strips[0].strip_y();
            let wtile_y1 = strips[n_strips - 1].strip_y() + 1;

            // Calculate the x range by examining all strips in wide tile coordinates
            let mut wtile_x0 = strips[0].x / WideTile::WIDTH;
            let mut wtile_x1 = wtile_x0;
            for i in 0..n_strips - 1 {
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
        for i in 0..n_strips - 1 {
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
        self.state_stack.last_mut().unwrap().n_clip += 1;
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

    /// Removes all active clip regions.
    pub fn pop_clips(&mut self) {
        while self.state_stack.last().unwrap().n_clip > 0 {
            self.pop_clip();
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
    pub fn pop_clip(&mut self) {
        self.state_stack.last_mut().unwrap().n_clip -= 1;

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
        for i in 0..n_strips - 1 {
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
            let mut col = (strip.alpha_idx / u32::from(Tile::HEIGHT)) as u16;
            let clip_x = clip_bbox.x0() * WideTile::WIDTH;
            if clip_x > x {
                col += clip_x - x;
                x = clip_x;
            }

            // Render clip strips for each affected tile and mark for popping
            for wtile_x in wtile_x0..wtile_x1 {
                // If we've moved past tile_x and have a pending pop, do it now
                if cur_wtile_x < wtile_x && core::mem::take(&mut pop_pending) {
                    self.get_mut(cur_wtile_x, cur_wtile_y).pop_clip();
                }

                // Calculate the portion of the strip that affects this tile
                let x_rel = (x % WideTile::WIDTH) as u32;
                let width = x1.min((wtile_x + 1) * WideTile::WIDTH) - x;

                // Create clip strip command for rendering the partial coverage
                let cmd = CmdClipAlphaFill {
                    x: x_rel,
                    width: width as u32,
                    alpha_idx: (col * Tile::HEIGHT) as usize,
                };
                x += width;
                col += width;

                // Apply the clip strip command and update state
                self.get_mut(wtile_x, cur_wtile_y).clip_strip(cmd);
                cur_wtile_x = wtile_x;
                pop_pending = true;
            }

            // Handle fill regions between strips based on fill rule
            let is_inside = match fill_rule {
                Fill::NonZero => next_strip.winding != 0,
                Fill::EvenOdd => next_strip.winding % 2 != 0,
            };
            if is_inside && strip_y == next_strip.strip_y() {
                let x2 = next_strip.x;
                let clipped_x2 = x2.min((cur_wtile_x + 1) * WideTile::WIDTH);
                let width = clipped_x2.saturating_sub(x1);

                // If there's a gap, fill it
                if width > 0 {
                    let x_rel = (x1 % WideTile::WIDTH) as u32;
                    self.get_mut(cur_wtile_x, cur_wtile_y)
                        .clip_fill(x_rel, width as u32);
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
                    let width2 = x2 % WideTile::WIDTH;
                    cur_wtile_x = x2 / WideTile::WIDTH;
                    if width2 > 0 {
                        self.get_mut(cur_wtile_x, cur_wtile_y)
                            .clip_fill(0, width2 as u32);
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
    pub bg: PremulRgba8,
    /// The draw commands of the tile.
    pub cmds: Vec<Cmd>,

    /// The number of zero-winding clips.
    pub n_zero_clip: usize,
    /// The number of non-zero-winding clips.
    pub n_clip: usize,
}

impl WideTile {
    /// The width of a wide tile in pixels.
    pub const WIDTH: u16 = 256;

    /// Create a new wide tile.
    pub fn new(x: u16, y: u16) -> Self {
        Self {
            x,
            y,
            bg: AlphaColor::<Srgb>::TRANSPARENT.premultiply().to_rgba8(),
            cmds: vec![],

            n_zero_clip: 0,
            n_clip: 0,
        }
    }

    pub(crate) fn fill(&mut self, x: u16, width: u16, paint: Paint) {
        if !self.is_zero_clip() {
            let Paint::Solid(s) = &paint else {
                unimplemented!()
            };
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
            let can_override = x == 0 && width == Self::WIDTH && s.a == 255 && self.n_clip == 0;

            if can_override {
                self.cmds.clear();
                self.bg = *s;
            } else {
                self.cmds.push(Cmd::Fill(CmdFill { x, width, paint }));
            }
        }
    }

    pub(crate) fn strip(&mut self, cmd_strip: CmdAlphaFill) {
        if !self.is_zero_clip() {
            self.cmds.push(Cmd::AlphaFill(cmd_strip));
        }
    }

    pub(crate) fn push(&mut self, cmd: Cmd) {
        self.cmds.push(cmd);
    }

    /// Adds a new clip region to the current wide tile.
    pub fn push_clip(&mut self) {
        if !self.is_zero_clip() {
            self.push(Cmd::PushClip);
            self.n_clip += 1;
        }
    }

    /// Removes the most recently added clip region from the current wide tile.
    pub fn pop_clip(&mut self) {
        if !self.is_zero_clip() {
            if matches!(self.cmds.last(), Some(Cmd::PushClip)) {
                // Nothing was drawn inside the clip, elide it.
                self.cmds.pop();
            } else {
                self.push(Cmd::PopClip);
            }
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
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushClip)) {
            self.cmds.push(Cmd::ClipStrip(cmd_clip_strip));
        }
    }

    /// Applies a clip fill operation at the specified position and width.
    pub fn clip_fill(&mut self, x: u32, width: u32) {
        if !self.is_zero_clip() && !matches!(self.cmds.last(), Some(Cmd::PushClip)) {
            self.cmds.push(Cmd::ClipFill(CmdClipFill { x, width }));
        }
    }
}

/// A drawing command.
#[derive(Debug)]
pub enum Cmd {
    /// A fill command.
    Fill(CmdFill),
    /// A fill command with alpha mask.
    AlphaFill(CmdAlphaFill),

    /// Pushes a new transparent buffer to the clip stack.
    ///
    /// This command is used to start a new clip region.
    PushClip,
    /// Pops the clip stack.
    ///
    /// This command is used to end the current clip region.
    PopClip,
    /// A fill command within a clipping region.
    ///
    /// This command is used to fill a region of a wide tile within a clipping region.
    ClipFill(CmdClipFill),
    /// A fill command with alpha mask within a clipping region.
    ///
    /// This command is used to fill a region of a wide tile with an alpha mask within a clipping region.
    ClipStrip(CmdClipAlphaFill),
}

/// Fill a consecutive region of a wide tile.
#[derive(Debug)]
pub struct CmdFill {
    /// The horizontal start position of the command in pixels.
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}

/// Fill a consecutive region of a wide tile with an alpha mask.
#[derive(Debug)]
pub struct CmdAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u16,
    /// The width of the command in pixels.
    pub width: u16,
    /// The start index into the alpha buffer of the command.
    pub alpha_idx: usize,
    /// The paint that should be used to fill the area.
    pub paint: Paint,
}

/// Same as fill, but copies top of clip stack to next on stack.
#[derive(Debug)]
pub struct CmdClipFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
}

/// Same as strip, but composites top of clip stack to next on stack.
#[derive(Debug)]
pub struct CmdClipAlphaFill {
    /// The horizontal start position of the command in pixels.
    pub x: u32,
    /// The width of the command in pixels.
    pub width: u32,
    /// The start index into the alpha buffer of the command.
    pub alpha_idx: usize,
}
