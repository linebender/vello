// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use vello_common::clip::ClipContext;
use vello_common::coarse::{MODE_HYBRID, Wide};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::fearless_simd::Level;
use vello_common::filter_effects::Filter;
use vello_common::glyph::{GlyphCaches, GlyphRenderer, GlyphRunBuilder, GlyphType, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{ImageId, ImageSource, Paint, PaintType};
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::recording::{PushLayerCommand, Recordable, Recorder, Recording, RenderCommand};
use vello_common::render_graph::RenderGraph;
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};

use crate::AtlasConfig;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// Round to the nearest integer for pixel snapping.
///
/// On `wasm32-unknown-unknown`, [`f64::round`] compiles to a software `libm`
/// call because its "round half away from zero" semantics don't match the
/// WebAssembly `f64.nearest` instruction (which uses banker's rounding).
/// In contrast, [`f64::floor`] maps directly to the native `f64.floor`
/// instruction.
///
/// `(x + 0.5).floor()` gives identical results to [`f64::round`] for
/// non-negative values and only differs at exactly half-integer negative
/// values (e.g. `−2.5` → `−2` instead of `−3`), which is negligible for
/// pixel snapping.
// TODO: Move somewhere in vello_common?
#[inline(always)]
fn pixel_snap(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// A blit rect for the instanced fast-path pipeline.
///
/// Represents an axis-aligned image rectangle that can be drawn by copying
/// directly from the image atlas to the screen, bypassing the strip/coarse pipeline.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlitRect {
    /// Screen-space destination position (pixel-snapped, after transform).
    /// Signed to allow rects that are partially off-screen (the GPU clips naturally).
    pub dst_x: i16,
    pub dst_y: i16,
    /// Screen-space destination size (the full rect, before image clamping).
    pub dst_w: u16,
    pub dst_h: u16,
    /// Pre-transform rect dimensions in geometry space (needed to compute the
    /// scale factor for clamping destination size to image bounds at render time).
    pub rect_w: u16,
    pub rect_h: u16,
    /// Source image reference (resolved to atlas coords at render time).
    pub image_id: ImageId,
}

/// A fence marking a strips-to-blits transition for pipeline interleaving.
///
/// References ranges in the [`Scene`]'s consolidated `all_cmd_ends` and `all_blits` buffers.
/// The cmd_ends range is implicit: flush point at index `i` corresponds to
/// `all_cmd_ends[i * n_tiles..(i + 1) * n_tiles]` where `n_tiles = width_tiles * height_tiles`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FlushPoint {
    /// Start index (inclusive) into [`Scene::all_blits`].
    pub blits_start: usize,
    /// End index (exclusive) into [`Scene::all_blits`].
    pub blits_end: usize,
}

/// Compact storage for dirty screen-space bounding boxes accumulated between flush points.
///
/// Each rect is stored as 4 `u16` values: `[x0, y0, MAX-x1, MAX-y1]` where `MAX` is
/// `u16::MAX`. The negated upper bounds allow a single SIMD `lt` comparison to test all
/// 4 overlap conditions simultaneously:
///
/// ```text
///   [d.x0, d.y0, MAX-d.x1, MAX-d.y1] < [b.x1, b.y1, MAX-b.x0, MAX-b.y0]
///   ≡ d.x0 < b.x1 AND d.y0 < b.y1 AND b.x0 < d.x1 AND b.y0 < d.y1
/// ```
///
/// The buffer is always padded to a multiple of 8 `u16` values (2 rects) using sentinel
/// values `[MAX; 4]` that can never satisfy the `lt` condition.
#[derive(Debug)]
struct DirtyRects {
    /// Flat u16 storage: 4 values per rect in `[x0, y0, MAX-x1, MAX-y1]` layout.
    data: Vec<u16>,
    /// Number of actual rects (excludes sentinel padding).
    count: usize,
}

impl DirtyRects {
    const VALS_PER_RECT: usize = 4;
    /// Sentinel value that can never satisfy `sentinel < anything` being true for all lanes.
    const SENTINEL: u16 = u16::MAX;

    fn new() -> Self {
        Self {
            data: Vec::new(),
            count: 0,
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }

    /// Push a viewport-clamped dirty rect.
    ///
    /// Coordinates must satisfy `x0 <= x1` and `y0 <= y1`.
    fn push(&mut self, x0: u16, y0: u16, x1: u16, y1: u16) {
        // Remove previous sentinel padding if present.
        if self.count & 1 != 0 {
            // Odd count means the last 4 values are a sentinel; remove them.
            self.data.truncate(self.count * Self::VALS_PER_RECT);
        }
        self.data
            .extend_from_slice(&[x0, y0, u16::MAX - x1, u16::MAX - y1]);
        self.count += 1;
        // Re-pad to a multiple of 2 rects (8 u16 values) for SIMD processing.
        if self.count & 1 != 0 {
            self.data
                .extend_from_slice(&[Self::SENTINEL; Self::VALS_PER_RECT]);
        }
    }

    /// Check whether any stored dirty rect overlaps the given blit rect.
    ///
    /// Uses SIMD-accelerated comparison via `u16x8`, processing 2 rects per iteration.
    #[inline(always)]
    fn any_overlap(
        &self,
        blit_x0: u16,
        blit_y0: u16,
        blit_x1: u16,
        blit_y1: u16,
        level: Level,
    ) -> bool {
        if self.count == 0 {
            return false;
        }
        use vello_common::fearless_simd::dispatch;
        dispatch!(level, simd => {
            Self::any_overlap_simd(simd, &self.data, blit_x0, blit_y0, blit_x1, blit_y1)
        })
    }

    #[inline(always)]
    fn any_overlap_simd<S: vello_common::fearless_simd::Simd>(
        s: S,
        data: &[u16],
        blit_x0: u16,
        blit_y0: u16,
        blit_x1: u16,
        blit_y1: u16,
    ) -> bool {
        use vello_common::fearless_simd::{Select, SimdFrom, SimdInt, u16x8};

        let blit_cmp = u16x8::simd_from(
            [
                blit_x1,
                blit_y1,
                u16::MAX - blit_x0,
                u16::MAX - blit_y0,
                blit_x1,
                blit_y1,
                u16::MAX - blit_x0,
                u16::MAX - blit_y0,
            ],
            s,
        );
        let ones = u16x8::simd_from(u16::MAX, s);
        let zeros = u16x8::simd_from(0u16, s);

        // Process 2 rects (8 u16 values) per iteration.
        for chunk in data.chunks_exact(8) {
            let dirty_vec = u16x8::simd_from(
                [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ],
                s,
            );

            // Compare: each lane true iff that overlap condition holds.
            let mask = dirty_vec.simd_lt(blit_cmp);
            let bits = mask.select(ones, zeros);

            // Reduce within each group of 4 lanes to a single per-rect result.
            // After step1: lane 0 = rect0(01 & 23 components), lane 1 = rect1(01 & 23 components)
            // (interleaved with duplicates in higher lanes).
            let step1 = bits.unzip_low(bits).and(bits.unzip_high(bits));
            let step2 = step1.unzip_low(step1).and(step1.unzip_high(step1));

            // step2.val[0] != 0 => all 4 conditions met for rect 0 (overlap).
            // step2.val[1] != 0 => all 4 conditions met for rect 1 (overlap).
            if step2.val[0] != 0 || step2.val[1] != 0 {
                return true;
            }
        }
        false
    }
}

/// Settings to apply to the render context.
#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    /// The SIMD level that should be used for rendering operations.
    pub level: Level,
    /// The configuration for the texture atlas.
    ///
    /// This controls how images are managed in GPU memory through texture atlases.
    /// The atlas system packs multiple images into larger textures to reduce the
    /// number of GPU texture bindings. This config allows customizing atlas parameters such as:
    /// - The number and size of atlases
    /// - How images are allocated across multiple atlases
    /// - Whether new atlases are automatically created when needed
    ///
    /// Adjusting these settings can affect memory usage and rendering performance
    /// depending on your application's image usage patterns.
    pub atlas_config: AtlasConfig,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::fallback()),
            atlas_config: AtlasConfig::default(),
        }
    }
}

/// A render state which contains the style properties for path rendering and
/// the current transform.
///
/// This is used to save and restore rendering state during recording operations.
#[derive(Debug)]
struct RenderState {
    /// The paint type (solid color, gradient, or image).
    pub(crate) paint: PaintType,
    /// Transform applied to the paint coordinates.
    pub(crate) paint_transform: Affine,
    /// Stroke style for path stroking operations.
    pub(crate) stroke: Stroke,
    /// Transform applied to geometry.
    pub(crate) transform: Affine,
    /// Fill rule for path filling operations.
    pub(crate) fill_rule: Fill,
    /// Blend mode for compositing.
    pub(crate) blend_mode: BlendMode,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    /// Width of the rendering surface in pixels.
    pub(crate) width: u16,
    /// Height of the rendering surface in pixels.
    pub(crate) height: u16,
    /// Wide coarse rasterizer for generating binned draw commands.
    pub(crate) wide: Wide<MODE_HYBRID>,
    clip_context: ClipContext,
    pub(crate) paint: PaintType,
    /// Transform applied to paint coordinates.
    pub(crate) paint_transform: Affine,
    pub(crate) aliasing_threshold: Option<u8>,
    /// Storage for encoded gradient and image paint data.
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    /// Whether the current paint is visible (e.g., alpha > 0).
    paint_visible: bool,
    /// Current stroke style for path stroking operations.
    pub(crate) stroke: Stroke,
    /// Current transform applied to geometry.
    pub(crate) transform: Affine,
    /// Current fill rule for path filling operations.
    pub(crate) fill_rule: Fill,
    /// Current blend mode for compositing.
    pub(crate) blend_mode: BlendMode,
    /// Generator for converting paths to strips.
    pub(crate) strip_generator: StripGenerator,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Cache for rasterized glyphs to improve text rendering performance.
    pub(crate) glyph_caches: Option<GlyphCaches>,
    /// Dependency graph for managing layer rendering order and filter effects.
    pub(crate) render_graph: RenderGraph,
    /// Flat buffer of per-tile command end indices across all flush points.
    /// Each flush point contributes exactly `n_tiles` entries in row-major order.
    pub(crate) all_cmd_ends: Vec<usize>,
    /// Flat buffer of blit rects across all flush points.
    pub(crate) all_blits: Vec<BlitRect>,
    /// Flush point metadata referencing ranges in the flat buffers above.
    pub(crate) flush_points: Vec<FlushPoint>,
    /// Whether the scene is currently accumulating blit rects (vs strip commands).
    in_blit_mode: bool, // TODO: Use enum?
    /// Screen-space bounding boxes of strip operations since the last flush point.
    /// Used by [`can_batch_blit`](Scene::can_batch_blit) to determine whether a
    /// blit rect can be folded into the previous [`FlushPoint`] without a pipeline switch.
    strips_dirty_rects: DirtyRects,
    /// SIMD level for dirty rect intersection checks.
    level: Level,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    /// Create a new render context with specific settings.
    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        let render_state = Self::default_render_state();
        let render_graph = RenderGraph::new();
        Self {
            width,
            height,
            wide: Wide::<MODE_HYBRID>::new(width, height),
            clip_context: ClipContext::new(),
            aliasing_threshold: None,
            paint: render_state.paint,
            paint_transform: render_state.paint_transform,
            encoded_paints: vec![],
            paint_visible: true,
            stroke: render_state.stroke,
            strip_generator: StripGenerator::new(width, height, settings.level),
            strip_storage: RefCell::new(StripStorage::default()),
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
            glyph_caches: Some(GlyphCaches::default()),
            render_graph,
            all_cmd_ends: vec![],
            all_blits: vec![],
            flush_points: vec![],
            in_blit_mode: false,
            strips_dirty_rects: DirtyRects::new(),
            level: settings.level,
        }
    }

    /// Create default rendering state.
    fn default_render_state() -> RenderState {
        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
        let paint_transform = Affine::IDENTITY;
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        let blend_mode = BlendMode::new(Mix::Normal, Compose::SrcOver);
        RenderState {
            transform,
            fill_rule,
            paint,
            paint_transform,
            stroke,
            blend_mode,
        }
    }

    /// Encode the current paint into a `Paint` that can be used for rendering.
    ///
    /// For solid colors, this is a simple conversion. For gradients and images,
    /// this encodes the paint data into the `encoded_paints` buffer and returns
    /// a `Paint` that references that data. The combined transform (geometry + paint)
    /// is applied during encoding.
    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(g) => g.encode_into(
                &mut self.encoded_paints,
                self.transform * self.paint_transform,
            ),
            PaintType::Image(i) => i.encode_into(
                &mut self.encoded_paints,
                self.transform * self.paint_transform,
            ),
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        let paint = self.encode_current_paint();
        self.fill_path_with(
            path,
            self.transform,
            self.fill_rule,
            paint,
            self.aliasing_threshold,
        );
    }

    /// Transition from blit mode to strip mode.
    ///
    /// Called before any operation that modifies the [`Wide`] coarse rasterizer
    /// (e.g. `fill_path_with`, `stroke_path_with`, `push_layer`, `pop_layer`).
    /// If we were accumulating blit rects, this resets the mode flag. The blits
    /// are already stored in the current [`FlushPoint`] so no data movement is needed.
    #[inline(always)]
    fn flush_blits(&mut self) {
        if self.in_blit_mode {
            self.in_blit_mode = false;
        }
    }

    /// Transition from strip mode to blit mode.
    ///
    /// Called from `try_blit_rect` when a blit rect is about to be added.
    /// Records a fence (the current per-tile command counts) and creates a new
    /// [`FlushPoint`] to collect the upcoming blit rects.
    #[inline(always)]
    fn flush_strips(&mut self) {
        if !self.in_blit_mode {
            // Append per-tile cmd counts to the flat buffer.
            let w = self.wide.width_tiles();
            let h = self.wide.height_tiles();
            for row in 0..h {
                for col in 0..w {
                    self.all_cmd_ends.push(self.wide.get(col, row).cmds.len());
                }
            }
            // Create flush point; blits range is empty until blits are added.
            let blits_start = self.all_blits.len();
            self.flush_points.push(FlushPoint {
                blits_start,
                blits_end: blits_start,
            });
            self.in_blit_mode = true;
            self.strips_dirty_rects.clear();
        }
    }

    /// Check whether a blit rect can be batched into the previous [`FlushPoint`]
    /// without creating a new pipeline switch.
    ///
    /// Returns `true` when either:
    /// - We are already in blit mode (trivially batchable), or
    /// - A previous flush point exists and the blit rect does not overlap any
    ///   strip operations recorded since that flush point.
    #[inline(always)]
    fn can_batch_blit(&self, dst_x: i16, dst_y: i16, dst_w: u16, dst_h: u16) -> bool {
        if self.in_blit_mode {
            return true;
        }
        if self.flush_points.is_empty() {
            return false;
        }
        let blit_x0 = dst_x.max(0) as u16;
        let blit_y0 = dst_y.max(0) as u16;
        let blit_x1 = blit_x0.saturating_add(dst_w).min(self.width);
        let blit_y1 = blit_y0.saturating_add(dst_h).min(self.height);
        !self
            .strips_dirty_rects
            .any_overlap(blit_x0, blit_y0, blit_x1, blit_y1, self.level)
    }

    /// Record a screen-space bounding box as dirty for the blit batching optimisation.
    ///
    /// The f64 rect (typically from `Affine::transform_rect_bbox`) is conservatively
    /// rounded outward and clamped to the viewport before being stored as compact u16.
    #[inline(always)]
    fn push_dirty_rect(&mut self, bbox: Rect) {
        let x0 = (bbox.x0.floor().max(0.0) as u32).min(u32::from(self.width)) as u16;
        let y0 = (bbox.y0.floor().max(0.0) as u32).min(u32::from(self.height)) as u16;
        let x1 = (bbox.x1.ceil().max(0.0) as u32).min(u32::from(self.width)) as u16;
        let y1 = (bbox.y1.ceil().max(0.0) as u32).min(u32::from(self.height)) as u16;
        self.strips_dirty_rects.push(x0, y0, x1, y1);
    }

    /// Record the full viewport as dirty (conservative fallback for layer ops, etc.).
    #[inline(always)]
    fn push_dirty_viewport(&mut self) {
        self.strips_dirty_rects.push(0, 0, self.width, self.height);
    }

    /// Build strips for a filled path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a path
    /// and submits them to the coarse rasterizer. The path is first converted
    /// to strips by the strip generator, then the strips are processed by the
    /// wide coarse rasterizer to generate binned draw commands.
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        self.flush_blits();
        self.push_dirty_rect(transform.transform_rect_bbox(path.bounding_box()));
        let wide = &mut self.wide;
        let strip_storage = &mut self.strip_storage.borrow_mut();
        self.strip_generator.generate_filled_path(
            path,
            fill_rule,
            transform,
            aliasing_threshold,
            strip_storage,
            self.clip_context.get(),
        );
        wide.generate(
            &strip_storage.strips,
            paint,
            self.blend_mode,
            0,
            None,
            &self.encoded_paints,
        );
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            self.fill_rule,
            self.transform,
            self.aliasing_threshold,
        );
    }

    /// Pop a clip path from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clip paths before finishing the rendering operation.
    pub fn pop_clip_path(&mut self) {
        self.clip_context.pop_clip();
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_visible {
            return;
        }

        let paint = self.encode_current_paint();
        self.stroke_path_with(path, self.transform, paint, self.aliasing_threshold);
    }

    /// Build strips for a stroked path with the given properties.
    ///
    /// This is the internal implementation that generates strips from a stroked path
    /// and submits them to the coarse rasterizer. The path is first stroked and
    /// converted to strips by the strip generator, then the strips are processed by
    /// the wide coarse rasterizer to generate binned draw commands.
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        self.flush_blits();
        let expand = self.stroke.width / 2.0;
        self.push_dirty_rect(
            transform
                .transform_rect_bbox(path.bounding_box())
                .inflate(expand, expand),
        );
        let wide = &mut self.wide;
        let strip_storage = &mut self.strip_storage.borrow_mut();

        self.strip_generator.generate_stroked_path(
            path,
            &self.stroke,
            transform,
            aliasing_threshold,
            strip_storage,
            self.clip_context.get(),
        );

        wide.generate(
            &strip_storage.strips,
            paint,
            self.blend_mode,
            0,
            None,
            &self.encoded_paints,
        );
    }

    /// Set the aliasing threshold.
    ///
    /// If set to `None` (which is the recommended option in nearly all cases),
    /// anti-aliasing will be applied.
    ///
    /// If instead set to some value, then a pixel will be fully painted if
    /// the coverage is bigger than the threshold (between 0 and 255), otherwise
    /// it will not be painted at all.
    ///
    /// Note that there is no performance benefit to disabling anti-aliasing and
    /// this functionality is simply provided for compatibility.
    pub fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.aliasing_threshold = aliasing_threshold;
    }

    /// Fill a rectangle with the current paint and fill rule.
    ///
    /// When the following conditions are all met, this uses a fast-path instanced
    /// blit pipeline that bypasses the strip/coarse pipeline entirely:
    /// - No active layers (clips or blends)
    /// - No active clip paths
    /// - Paint is an image with an `OpaqueId` source (i.e. in the atlas)
    /// - Blend mode is the default SrcOver
    /// - Combined transform is axis-aligned (no rotation/shear)
    ///
    /// Otherwise, falls back to the normal `fill_path` codepath.
    pub fn fill_rect(&mut self, rect: &Rect) {
        if self.try_blit_rect(rect) {
            return;
        }
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Attempt the fast-path blit rect pipeline. Returns `true` if the rect was
    /// handled by the blit pipeline, `false` if it should fall through to the
    /// normal strip pipeline.
    fn try_blit_rect(&mut self, rect: &Rect) -> bool {
        if !self.paint_visible {
            return true; // Invisible paint, nothing to draw either way.
        }

        // Condition 1: No active layers (clips or blends).
        if self.wide.has_layers() {
            return false;
        }

        // Condition 2: No active clip paths.
        if self.clip_context.get().is_some() {
            return false;
        }

        // Condition 3: Paint must be an image with an OpaqueId (atlas-backed).
        let image_id = match &self.paint {
            PaintType::Image(img) => match &img.image {
                ImageSource::OpaqueId(id) => *id,
                _ => return false,
            },
            _ => return false,
        };

        // Condition 4: Blend mode must be default SrcOver.
        let default_blend = BlendMode::new(Mix::Normal, Compose::SrcOver);
        if self.blend_mode != default_blend {
            return false;
        }

        // Condition 5: Geometry transform must be axis-aligned (no rotation/shear).
        let geo_coeffs = self.transform.as_coeffs();
        // coeffs: [a, b, c, d, tx, ty] where the matrix is [[a, c, tx], [b, d, ty]]
        // Axis-aligned means b == 0 and c == 0 (no shear/rotation).
        if (geo_coeffs[1] as f32).abs() > f32::EPSILON
            || (geo_coeffs[2] as f32).abs() > f32::EPSILON
        {
            return false;
        }

        // Compute the screen-space destination rect by applying the geometry transform.
        // For axis-aligned transforms: x' = a*x + tx, y' = d*y + ty
        let (a, d, tx, ty) = (geo_coeffs[0], geo_coeffs[3], geo_coeffs[4], geo_coeffs[5]);
        let x0 = a * rect.x0 + tx;
        let y0 = d * rect.y0 + ty;
        let x1 = a * rect.x1 + tx;
        let y1 = d * rect.y1 + ty;

        // Handle negative scale (flipped rect) by normalizing.
        let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

        // Pre-transform rect dimensions (geometry space).
        let rect_w = pixel_snap((rect.x1 - rect.x0).abs()).max(0.0) as u16;
        let rect_h = pixel_snap((rect.y1 - rect.y0).abs()).max(0.0) as u16;

        // Pixel-snap the destination rect. Position is signed to allow
        // partially off-screen rects (the GPU clips naturally via NDC).
        let rx0 = pixel_snap(x0);
        let ry0 = pixel_snap(y0);
        let rx1 = pixel_snap(x1);
        let ry1 = pixel_snap(y1);
        let dst_x = rx0 as i16;
        let dst_y = ry0 as i16;
        let dst_w = (rx1 - rx0).max(0.0) as u16;
        let dst_h = (ry1 - ry0).max(0.0) as u16;

        if dst_w == 0 || dst_h == 0 || rect_w == 0 || rect_h == 0 {
            return true; // Zero-size rect, nothing to draw.
        }

        // Optimisation: if the blit rect doesn't overlap any strip operations
        // since the last flush point, batch it into the previous flush point's
        // blits instead of creating a new pipeline switch.
        if self.can_batch_blit(dst_x, dst_y, dst_w, dst_h) {
            self.in_blit_mode = true;
        } else {
            self.flush_strips();
        }

        self.all_blits.push(BlitRect {
            dst_x,
            dst_y,
            dst_w,
            dst_h,
            rect_w,
            rect_h,
            image_id,
        });
        self.flush_points.last_mut().unwrap().blits_end = self.all_blits.len();

        true
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &FontData) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Push a new layer with the given properties.
    ///
    /// Only `clip_path` is supported for now.
    // TODO: Implement filter integration.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        self.flush_blits();
        self.push_dirty_viewport();
        if filter.is_some() {
            unimplemented!("Filter effects are not yet supported in vello_hybrid");
        }

        let mut strip_storage = self.strip_storage.borrow_mut();

        let clip = if let Some(c) = clip_path {
            self.strip_generator.generate_filled_path(
                c,
                self.fill_rule,
                self.transform,
                self.aliasing_threshold,
                &mut strip_storage,
                self.clip_context.get(),
            );

            Some(strip_storage.strips.as_slice())
        } else {
            None
        };

        // Mask is unsupported. Blend is partially supported.
        if mask.is_some() {
            unimplemented!()
        }

        self.wide.push_layer(
            0,
            clip,
            blend_mode.unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
            None,
            opacity.unwrap_or(1.),
            None,
            self.transform,
            &mut self.render_graph,
            0,
        );
    }

    /// Push a new clip layer.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/sparse_strips/vello_cpu/examples)
    /// example for how this method differs from `push_clip_path`.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None, None);
    }

    /// Push a new blend layer.
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }

    /// Push a new opacity layer.
    pub fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(None, None, Some(opacity), None, None);
    }

    /// Push a new mask layer.
    ///
    /// Note that masks are not yet supported in `vello_hybrid`.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a new filter layer.
    ///
    /// Note that filters are currently ignored in `vello_hybrid`.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        self.flush_blits();
        self.push_dirty_viewport();
        self.wide.pop_layer(&mut self.render_graph);
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the paint for subsequent rendering operations.
    // TODO: This API is not final. Supporting images from a pixmap is explicitly out of scope.
    //       Instead images should be passed via a backend-agnostic opaque id, and be hydrated at
    //       render time into a texture usable by the renderer backend.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
        self.paint_visible = match &self.paint {
            PaintType::Solid(color) => color.components[3] != 0.0,
            _ => true,
        };
    }

    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.paint_transform = paint_transform;
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.paint_transform = Affine::IDENTITY;
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Apply filter to the current paint (affects next drawn element)
    pub fn set_filter_effect(&mut self, _filter: Filter) {
        unimplemented!("Filter effects integration with Scene")
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        unimplemented!("Filter effects integration with Scene")
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.strip_generator.reset();
        self.clip_context.reset();
        self.strip_storage.borrow_mut().clear();
        self.encoded_paints.clear();
        self.all_cmd_ends.clear();
        self.all_blits.clear();
        self.flush_points.clear();
        self.in_blit_mode = false;
        self.strips_dirty_rects.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.paint_transform = render_state.paint_transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;

        self.glyph_caches.as_mut().unwrap().maintain();
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }
}

impl GlyphRenderer for Scene {
    fn fill_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                let paint = self.encode_current_paint();
                self.fill_path_with(
                    glyph.path,
                    prepared_glyph.transform,
                    Fill::NonZero,
                    paint,
                    self.aliasing_threshold,
                );
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn stroke_glyph(&mut self, prepared_glyph: PreparedGlyph<'_>) {
        match prepared_glyph.glyph_type {
            GlyphType::Outline(glyph) => {
                let paint = self.encode_current_paint();
                self.stroke_path_with(
                    glyph.path,
                    prepared_glyph.transform,
                    paint,
                    self.aliasing_threshold,
                );
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn take_glyph_caches(&mut self) -> GlyphCaches {
        self.glyph_caches.take().unwrap_or_default()
    }

    fn restore_glyph_caches(&mut self, cache: GlyphCaches) {
        self.glyph_caches = Some(cache);
    }
}

impl Recordable for Scene {
    fn record<F>(&mut self, recording: &mut Recording, f: F)
    where
        F: FnOnce(&mut Recorder<'_>),
    {
        let mut recorder = Recorder::new(recording, self.transform, self.take_glyph_caches());
        f(&mut recorder);
        self.glyph_caches = Some(recorder.take_glyph_caches());
    }

    fn prepare_recording(&mut self, recording: &mut Recording) {
        let buffers = recording.take_cached_strips();
        let (strip_storage, strip_start_indices) =
            self.generate_strips_from_commands(recording.commands(), buffers);
        recording.set_cached_strips(strip_storage, strip_start_indices);
    }

    fn execute_recording(&mut self, recording: &Recording) {
        let (cached_strips, cached_alphas) = recording.get_cached_strips();
        let adjusted_strips = self.prepare_cached_strips(cached_strips, cached_alphas);

        // Use pre-calculated strip start indices from when we generated the cache
        let strip_start_indices = recording.get_strip_start_indices();
        let mut range_index = 0;

        // Replay commands in order, using cached strips for geometry
        for command in recording.commands() {
            match command {
                RenderCommand::FillPath(_)
                | RenderCommand::StrokePath(_)
                | RenderCommand::FillRect(_)
                | RenderCommand::StrokeRect(_)
                | RenderCommand::FillOutlineGlyph(_)
                | RenderCommand::StrokeOutlineGlyph(_) => {
                    self.process_geometry_command(
                        strip_start_indices,
                        range_index,
                        &adjusted_strips,
                    );
                    range_index += 1;
                }
                RenderCommand::SetPaint(paint) => {
                    self.set_paint(paint.clone());
                }
                RenderCommand::SetPaintTransform(transform) => {
                    self.set_paint_transform(*transform);
                }
                RenderCommand::ResetPaintTransform => {
                    self.reset_paint_transform();
                }
                RenderCommand::SetTransform(transform) => {
                    self.set_transform(*transform);
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.set_fill_rule(*fill_rule);
                }
                RenderCommand::SetStroke(stroke) => {
                    self.set_stroke(stroke.clone());
                }
                RenderCommand::SetFilterEffect(filter) => {
                    self.set_filter_effect(filter.clone());
                }
                RenderCommand::ResetFilterEffect => {
                    self.reset_filter_effect();
                }
                RenderCommand::PushLayer(PushLayerCommand {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    filter,
                }) => {
                    self.push_layer(
                        clip_path.as_ref(),
                        *blend_mode,
                        *opacity,
                        mask.clone(),
                        filter.clone(),
                    );
                }
                RenderCommand::PopLayer => {
                    self.pop_layer();
                }
            }
        }
    }
}

/// Recording management implementation.
impl Scene {
    /// Generate strips from strip commands and capture ranges.
    ///
    /// Returns:
    /// - `collected_strips`: The generated strips.
    /// - `collected_alphas`: The generated alphas.
    /// - `strip_start_indices`: The start indices of strips for each geometry command.
    fn generate_strips_from_commands(
        &mut self,
        commands: &[RenderCommand],
        buffers: (StripStorage, Vec<usize>),
    ) -> (StripStorage, Vec<usize>) {
        let (mut strip_storage, mut strip_start_indices) = buffers;
        strip_storage.clear();
        strip_storage.set_generation_mode(GenerationMode::Append);
        strip_start_indices.clear();

        let saved_state = self.take_current_state();

        for command in commands {
            let start_index = strip_storage.strips.len();

            match command {
                RenderCommand::FillPath(path) => {
                    self.strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokePath(path) => {
                    self.strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillRect(rect) => {
                    self.strip_generator.generate_filled_path(
                        rect.to_path(DEFAULT_TOLERANCE),
                        self.fill_rule,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeRect(rect) => {
                    self.strip_generator.generate_stroked_path(
                        rect.to_path(DEFAULT_TOLERANCE),
                        &self.stroke,
                        self.transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::FillOutlineGlyph((path, glyph_transform)) => {
                    self.strip_generator.generate_filled_path(
                        path,
                        self.fill_rule,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::StrokeOutlineGlyph((path, glyph_transform)) => {
                    self.strip_generator.generate_stroked_path(
                        path,
                        &self.stroke,
                        *glyph_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        None,
                    );
                    strip_start_indices.push(start_index);
                }
                RenderCommand::SetTransform(transform) => {
                    self.transform = *transform;
                }
                RenderCommand::SetFillRule(fill_rule) => {
                    self.fill_rule = *fill_rule;
                }
                RenderCommand::SetStroke(stroke) => {
                    self.stroke = stroke.clone();
                }

                _ => {}
            }
        }

        self.restore_state(saved_state);

        (strip_storage, strip_start_indices)
    }

    fn process_geometry_command(
        &mut self,
        strip_start_indices: &[usize],
        range_index: usize,
        adjusted_strips: &[Strip],
    ) {
        self.flush_blits();
        self.push_dirty_viewport();
        assert!(
            range_index < strip_start_indices.len(),
            "Strip range index out of bounds: range_index={}, strip_start_indices.len()={}",
            range_index,
            strip_start_indices.len()
        );
        let start = strip_start_indices[range_index];
        let end = strip_start_indices
            .get(range_index + 1)
            .copied()
            .unwrap_or(adjusted_strips.len());
        let count = end - start;
        if count == 0 {
            // There are no strips to generate.
            return;
        }
        assert!(
            start < adjusted_strips.len() && count > 0,
            "Invalid strip range: start={start}, end={end}, count={count}"
        );
        let paint = self.encode_current_paint();
        self.wide.generate(
            &adjusted_strips[start..end],
            paint,
            self.blend_mode,
            0,
            None,
            &self.encoded_paints,
        );
    }

    /// Prepare cached strips for rendering by adjusting alpha indices and extending alpha buffer.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Alphas length conversion is safe in this case"
    )]
    fn prepare_cached_strips(
        &mut self,
        cached_strips: &[Strip],
        cached_alphas: &[u8],
    ) -> Vec<Strip> {
        let mut strip_storage = self.strip_storage.borrow_mut();
        // Calculate offset for alpha indices based on current buffer size.
        let alpha_offset = strip_storage.alphas.len() as u32;
        // Extend current alpha buffer with cached alphas.
        strip_storage.alphas.extend(cached_alphas);
        // Create adjusted strips with corrected alpha indices
        cached_strips
            .iter()
            .map(move |strip| {
                let mut adjusted_strip = *strip;
                adjusted_strip.set_alpha_idx(adjusted_strip.alpha_idx() + alpha_offset);
                adjusted_strip
            })
            .collect()
    }

    /// Save current rendering state.
    fn take_current_state(&mut self) -> RenderState {
        RenderState {
            paint: self.paint.clone(),
            paint_transform: self.paint_transform,
            transform: self.transform,
            fill_rule: self.fill_rule,
            blend_mode: self.blend_mode,
            stroke: core::mem::take(&mut self.stroke),
        }
    }

    /// Restore rendering state.
    fn restore_state(&mut self, state: RenderState) {
        self.paint = state.paint;
        self.paint_transform = state.paint_transform;
        self.stroke = state.stroke;
        self.transform = state.transform;
        self.fill_rule = state.fill_rule;
        self.blend_mode = state.blend_mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::kurbo::{BezPath, Rect};
    use vello_common::paint::ImageSource;
    use vello_common::peniko::ImageSampler;

    // -----------------------------------------------------------------------
    // DirtyRects unit tests
    // -----------------------------------------------------------------------

    fn test_level() -> Level {
        Level::try_detect().unwrap_or(Level::fallback())
    }

    #[test]
    fn dirty_rects_empty_no_overlap() {
        let dr = DirtyRects::new();
        assert!(!dr.any_overlap(0, 0, 100, 100, test_level()));
    }

    #[test]
    fn dirty_rects_single_overlap() {
        let mut dr = DirtyRects::new();
        dr.push(10, 10, 50, 50);
        // Overlapping rect.
        assert!(dr.any_overlap(20, 20, 60, 60, test_level()));
        // Non-overlapping rect (to the right).
        assert!(!dr.any_overlap(60, 10, 100, 50, test_level()));
        // Non-overlapping rect (below).
        assert!(!dr.any_overlap(10, 60, 50, 100, test_level()));
    }

    #[test]
    fn dirty_rects_two_rects_no_false_union() {
        let mut dr = DirtyRects::new();
        // Two small rects at opposite corners of a 1000x1000 viewport.
        dr.push(0, 0, 50, 50); // top-left
        dr.push(950, 950, 1000, 1000); // bottom-right
        // A rect in the centre should NOT overlap either (no union inflation).
        assert!(!dr.any_overlap(400, 400, 600, 600, test_level()));
        // But rects overlapping the corners should.
        assert!(dr.any_overlap(0, 0, 30, 30, test_level()));
        assert!(dr.any_overlap(960, 960, 1000, 1000, test_level()));
    }

    #[test]
    fn dirty_rects_padding_sentinel_no_false_positive() {
        let mut dr = DirtyRects::new();
        // Push an odd number of rects so sentinel padding is present.
        dr.push(10, 10, 20, 20);
        assert_eq!(dr.count, 1);
        // Data should be padded to 8 u16 values (2 rects).
        assert_eq!(dr.data.len(), 8);
        // A query far from the actual rect should not overlap, even though
        // sentinel padding is present in the second slot.
        assert!(!dr.any_overlap(500, 500, 600, 600, test_level()));
        // The actual rect should still be detected.
        assert!(dr.any_overlap(5, 5, 15, 15, test_level()));
    }

    #[test]
    fn dirty_rects_clear_resets() {
        let mut dr = DirtyRects::new();
        dr.push(0, 0, 1000, 1000);
        assert!(dr.any_overlap(500, 500, 600, 600, test_level()));
        dr.clear();
        assert!(!dr.any_overlap(500, 500, 600, 600, test_level()));
    }

    #[test]
    fn dirty_rects_adjacent_no_overlap() {
        let mut dr = DirtyRects::new();
        dr.push(0, 0, 50, 50);
        // Exactly adjacent (touching but not overlapping).
        assert!(!dr.any_overlap(50, 0, 100, 50, test_level()));
        assert!(!dr.any_overlap(0, 50, 50, 100, test_level()));
    }

    // -----------------------------------------------------------------------
    // Scene-level blit batching tests
    // -----------------------------------------------------------------------

    /// Create a minimal scene and set an OpaqueId image paint so `try_blit_rect` succeeds.
    fn scene_with_image_paint(width: u16, height: u16) -> Scene {
        let mut scene = Scene::new(width, height);
        let image_id = ImageId::new(42);
        scene.paint = PaintType::Image(vello_common::paint::Image {
            image: ImageSource::OpaqueId(image_id),
            sampler: ImageSampler::default(),
        });
        scene.paint_visible = true;
        scene
    }

    #[test]
    fn blit_batching_non_overlapping_reduces_flush_points() {
        let mut scene = scene_with_image_paint(1920, 1080);

        // First image rect at left side.
        scene.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        assert_eq!(
            scene.flush_points.len(),
            1,
            "first blit creates a flush point"
        );

        // A path drawn at the right side (non-overlapping).
        let mut path = BezPath::new();
        path.move_to((500.0, 0.0));
        path.line_to((600.0, 0.0));
        path.line_to((600.0, 100.0));
        path.line_to((500.0, 100.0));
        path.close_path();
        scene.fill_path(&path);

        // Second image rect back at the left side (non-overlapping with the path).
        scene.fill_rect(&Rect::new(0.0, 110.0, 100.0, 210.0));

        // Because the second blit doesn't overlap the path, it should be batched
        // into the first flush point (no new flush point created).
        assert_eq!(
            scene.flush_points.len(),
            1,
            "non-overlapping blit should be batched into existing flush point"
        );
        assert_eq!(scene.all_blits.len(), 2, "both blits should be recorded");
    }

    #[test]
    fn blit_batching_overlapping_creates_new_flush_point() {
        let mut scene = scene_with_image_paint(1920, 1080);

        // First image rect.
        scene.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        assert_eq!(scene.flush_points.len(), 1);

        // A path drawn overlapping the next blit's region.
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((200.0, 0.0));
        path.line_to((200.0, 200.0));
        path.line_to((0.0, 200.0));
        path.close_path();
        scene.fill_path(&path);

        // Second image rect overlaps the path.
        scene.fill_rect(&Rect::new(50.0, 50.0, 150.0, 150.0));

        // Overlapping blit must create a new flush point.
        assert_eq!(
            scene.flush_points.len(),
            2,
            "overlapping blit must create a new flush point"
        );
    }

    #[test]
    fn blit_batching_scattered_paths_center_blit_batches() {
        let mut scene = scene_with_image_paint(1920, 1080);

        // First blit at top-left.
        scene.fill_rect(&Rect::new(0.0, 0.0, 50.0, 50.0));
        assert_eq!(scene.flush_points.len(), 1);

        // Two paths at opposite corners.
        let mut tl_path = BezPath::new();
        tl_path.move_to((0.0, 100.0));
        tl_path.line_to((40.0, 100.0));
        tl_path.line_to((40.0, 140.0));
        tl_path.line_to((0.0, 140.0));
        tl_path.close_path();
        scene.fill_path(&tl_path);

        let mut br_path = BezPath::new();
        br_path.move_to((1800.0, 1000.0));
        br_path.line_to((1900.0, 1000.0));
        br_path.line_to((1900.0, 1080.0));
        br_path.line_to((1800.0, 1080.0));
        br_path.close_path();
        scene.fill_path(&br_path);

        // A blit in the centre of the screen -- no overlap with either path.
        scene.fill_rect(&Rect::new(900.0, 500.0, 1000.0, 600.0));

        assert_eq!(
            scene.flush_points.len(),
            1,
            "centre blit should batch because it doesn't overlap either corner path"
        );
    }
}
