// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

#[cfg(feature = "text")]
use crate::Resources;
#[cfg(feature = "text")]
use crate::text::GlyphRunBuilder;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::Range;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::clip::{ClipRef, ClipShape};
use vello_common::encode::{EncodeExt, EncodedPaint, invert_paint_transform};
use vello_common::fearless_simd::Level;
use vello_common::filter::FilterData;
use vello_common::filter_effects::Filter;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::{Paint, PaintType, PremulColor, Tint};
#[cfg(feature = "text")]
use vello_common::peniko::FontData;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Fill, GradientKind, ImageQuality};
use vello_common::record::{CommandRecorder, Drawable, LayerClip, LayerProps, PoppedLayer};
use vello_common::render_state::RenderState;
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use vello_common::transforms::{RootTransforms, Transforms};
use vello_common::util::{RectExt, into_fast_path_rect, strip_bbox};
use vello_common::viewport::ViewportState;

#[cfg(feature = "wgpu")]
use crate::sparse_strips::{AddPath, PathSummary, StripInput, summarize_path_tiles};
#[cfg(feature = "wgpu")]
use vello_common::clip::PathDataRef;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// Path or rectangle draw retained by the scene recorder.
#[derive(Debug)]
pub(crate) enum RecordedDraw {
    /// Path represented by generated strips.
    Path(RecordedPath),
    /// Path whose strips the GPU produces at render time (see [`Scene::set_gpu_strips`]).
    #[cfg(feature = "wgpu")]
    GpuPath(RecordedGpuPath),
    /// Rectangle retained for the direct GPU rectangle path.
    Rect(RecordedRect),
}

/// A path recorded for GPU strip generation.
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub(crate) struct RecordedGpuPath {
    /// Index of the path in the scene's [`GpuStrips`] input; its strip range is known once
    /// the GPU strips are resolved.
    pub(crate) path: u32,
    /// Bounding box of the path's strips, derived from its tiles.
    pub(crate) bbox: RectU16,
    /// Paint applied to the path.
    pub(crate) paint: Paint,
}

/// Recorded path strips and their paint.
#[derive(Debug)]
pub(crate) struct RecordedPath {
    /// Range selecting the path's strips from scene strip storage.
    pub(crate) strips: Range<usize>,
    /// Paint applied to the path.
    pub(crate) paint: Paint,
}

/// Recorded rectangle and its paint.
#[derive(Debug)]
pub(crate) struct RecordedRect {
    /// Rectangle in scene coordinates.
    pub(crate) rect: Rect,
    /// Paint applied to the rectangle.
    pub(crate) paint: Paint,
}

impl RecordedDraw {
    fn new_path(strips: Range<usize>, paint: Paint) -> Self {
        Self::Path(RecordedPath { strips, paint })
    }

    fn new_rect(rect: Rect, paint: Paint) -> Self {
        Self::Rect(RecordedRect { rect, paint })
    }
}

impl Drawable for RecordedDraw {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "recorded fast rectangles are clipped to the u16 viewport"
    )]
    fn bbox(&self, strips: &[Strip]) -> Option<RectU16> {
        match self {
            // TODO: The bbox trait should take a reference to _all_ strips, and then we select
            // the correct strips here. Otherwise, callers of this method always have to make sure
            // to slice the strips correctly themselves.
            // TODO: We also should just compute the bbox during strip generation instead of
            // recomputing it here.
            Self::Path(_) => strip_bbox(strips),
            #[cfg(feature = "wgpu")]
            Self::GpuPath(path) => (!path.bbox.is_empty()).then_some(path.bbox),
            Self::Rect(rect) => {
                let rect = rect.rect;

                (!rect.is_zero_area()).then(|| {
                    RectU16::new(
                        rect.x0.floor() as u16,
                        rect.y0.floor() as u16,
                        rect.x1.ceil() as u16,
                        rect.y1.ceil() as u16,
                    )
                    .snap_to_tile_coordinates()
                })
            }
        }
    }

    fn blend_mode(&self) -> Option<&BlendMode> {
        None
    }
}

/// Settings to apply to the render context.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub(crate) level: Level,
    pub(crate) memory_settings: MemorySettings,
    #[cfg(feature = "wgpu")]
    pub(crate) pipeline_cache: Option<wgpu::PipelineCache>,
}

impl RenderSettings {
    /// Sets the SIMD level used for rendering operations.
    ///
    /// By default, the level is detected at runtime, falling back to [`Level::baseline`].
    #[must_use]
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Returns the SIMD level used for rendering operations.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Sets the configuration for GPU memory used while rendering.
    #[must_use]
    pub fn with_memory_settings(mut self, memory_settings: MemorySettings) -> Self {
        self.memory_settings = memory_settings;
        self
    }

    /// Sets the cache used when creating the wgpu render pipelines.
    ///
    /// A cache can substantially reduce startup times, especially on low-tier devices.
    /// Requires [`wgpu::Features::PIPELINE_CACHE`] on the device.
    /// By default, no cache is used.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub fn with_pipeline_cache(mut self, pipeline_cache: wgpu::PipelineCache) -> Self {
        self.pipeline_cache = Some(pipeline_cache);
        self
    }
}

/// Settings controlling usage of GPU memory.
#[derive(Copy, Clone, Debug, Default)]
pub struct MemorySettings {
    /// Configuration for the atlas holding uploaded images.
    pub image_atlas_config: AtlasConfig,
    /// Configuration for intermediate layer and scratch textures.
    pub layers_config: LayersConfig,
}

/// Configuration for temporary textures used to render layers and effects.
///
/// Layers, blend modes, and filters need temporary GPU textures before their results can be
/// composited into the main scene. This configuration allows you to specify some limits to prevent
/// excessive allocations for complex or adversarial scenes.
///
/// In general, Vello GPU has the following priorities in that specific order:
/// - Render as many scenes as possible successfully.
/// - Use as little memory as possible.
/// - Be as performant as possible.
///
/// Vello GPU will by default prefer minimizing memory usage, even if at the cost of worse
/// performance. However, since the exact tradeoff is very application-dependent,
/// this configuration still allows you to tune the exact parameters such that better performance
/// is accepted at the cost of larger memory consumption.
#[derive(Copy, Clone, Debug)]
pub struct LayersConfig {
    /// Maximum number of intermediate textures that may be allocated.
    ///
    /// `None` permits growth up to device limits. A finite value allows you to impose a hard limit,
    /// but might result in certain scenes being rejected. The exact texture requirements for
    /// different types of scenes are considered an implementation detail, but in general:
    ///
    /// - `0` for scenes without layers, including layers introduced by COLR glyphs.
    /// - `1` for scenes with only single-child layers and no blend modes / filters.
    /// - `4` for scenes with only single-child layers with potentially blend modes
    ///   or simple filters.
    ///
    /// Any other type of scene with complex layer nesting might need more intermediate textures,
    /// depending on the exact contents.
    pub max_textures: Option<usize>,
    /// Minimum width and height of each allocated intermediate texture.
    ///
    /// By default, textures only grow to fit the largest required layer,
    /// up to [`Self::max_texture_size`], and do not currently shrink between frames. By increasing
    /// the minimum texture size, you can force Vello GPU to more aggressively batch layer draws,
    /// at the cost of larger memory consumption.
    ///
    /// Desktop GPUs may benefit from a larger minimum when rendering scenes with many layers. On
    /// mobile GPUs, experiments have shown that keeping intermediate textures smaller is
    /// more important for both memory usage and performance, even if it requires more render passes.
    /// Therefore, you should in general avoid making this value too large. The default of
    /// 512 by 512 is intended as a conservative starting point. Setting this to 1024 by 1024 is a
    /// reasonable choice, too. Higher values are only recommended if benchmarks have proven
    /// that your target workload profits from this and you don't plan on targeting mobile devices.
    pub min_texture_size: SizeU16,
    /// Maximum width and height of each allocated intermediate texture.
    ///
    /// This should be at least the size of the main scene (and potentially more to account for
    /// padding of large filters), unless you explicitly want to reject too large layers to limit
    /// peak memory consumption.
    ///
    /// Regardless of what this value is set to, device limits will always be respected.
    pub max_texture_size: SizeU16,
}

impl Default for LayersConfig {
    fn default() -> Self {
        Self {
            max_textures: None,
            min_texture_size: SizeU16::new(512),
            max_texture_size: SizeU16::new(4096),
        }
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            memory_settings: MemorySettings::default(),
            #[cfg(feature = "wgpu")]
            pipeline_cache: None,
        }
    }
}

/// A render context for GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    /// Width of the rendering surface in pixels.
    pub(crate) width: u16,
    /// Height of the rendering surface in pixels.
    pub(crate) height: u16,
    /// Viewport-dependent path and clip generation state.
    viewport_state: ViewportState,
    pub(crate) render_state: RenderState,
    /// Root transform stack.
    root_transforms: RootTransforms,
    pub(crate) aliasing_threshold: Option<u8>,
    /// Storage for encoded non-solid paint data.
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    /// Storage for generated strips and alpha values.
    pub(crate) strip_storage: RefCell<StripStorage>,
    /// Current filter effect applied to individual draw operations.
    filter: Option<Filter>,
    /// The command recorder.
    pub(crate) recorder: CommandRecorder<RecordedDraw>,
    /// SIMD level.
    #[cfg(feature = "wgpu")]
    level: Level,
    /// State of GPU strip generation, when enabled.
    #[cfg(feature = "wgpu")]
    pub(crate) gpu_strips: Option<RefCell<GpuStrips>>,
}

/// Paths whose strips are generated on the GPU, and the result once resolved.
///
/// Experimental: see [`Scene::set_gpu_strips`].
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub(crate) struct GpuStrips {
    /// The sorted tiles and lines of every recorded GPU path.
    input: StripInput,
    /// Per GPU path, the clip snapshot to intersect with, if the path was drawn under a clip.
    clips: Vec<Option<usize>>,
    /// Copies of the clip paths active while GPU paths were recorded.
    snapshots: Vec<ClipSnapshot>,
    /// Per GPU path, its range in the scene's strip storage, once resolved.
    pub(crate) ranges: Vec<Range<usize>>,
    /// Whether the GPU strips of this recording have been produced.
    resolved: bool,
    /// The GPU buffers of the resolved recording; the renderer copies their alphas into the
    /// alpha texture.
    pub(crate) buffers: Option<crate::sparse_strips::gpu::SparseStripBuffers>,
    /// Byte offset in the scene's alphas at which the GPU alphas start (a multiple of the alpha
    /// texture row size).
    pub(crate) alpha_base: usize,
    /// Per GPU path, what its strips will be (exact counts and bounding box).
    summaries: Vec<PathSummary>,
    /// Per GPU path, whether it is drawn directly from GPU-built instances (root level, solid
    /// paint, no clip) rather than through read-back strips.
    direct: Vec<bool>,
    /// Per GPU path, its `(instance base, instance count)` when direct, computed at resolve.
    pub(crate) instance_spans: Vec<Option<[u32; 2]>>,
    /// How many bytes of the scene's alphas the renderer must upload: up to the GPU alpha
    /// region when nothing follows it, since the GPU writes that region itself.
    pub(crate) alpha_upload_len: Option<usize>,
}

/// The strips of a clip path at the time a GPU path was recorded under it.
#[cfg(feature = "wgpu")]
#[derive(Debug)]
struct ClipSnapshot {
    key: (u64, usize),
    strips: Vec<Strip>,
    alphas: Vec<u8>,
    bbox: RectU16,
}

#[cfg(feature = "wgpu")]
impl GpuStrips {
    fn new(width: u16, height: u16) -> Self {
        Self {
            input: StripInput::new(width, height),
            clips: Vec::new(),
            snapshots: Vec::new(),
            ranges: Vec::new(),
            resolved: false,
            buffers: None,
            alpha_base: 0,
            summaries: Vec::new(),
            direct: Vec::new(),
            instance_spans: Vec::new(),
            alpha_upload_len: None,
        }
    }

    /// Whether any path needs its strips on the CPU (clipped, non-solid paint, or in a layer).
    fn needs_readback(&self) -> bool {
        self.direct.iter().any(|direct| !direct)
    }

    fn reset(&mut self, width: u16, height: u16) {
        self.input.clear();
        self.input.width = width;
        self.input.height = height;
        self.clips.clear();
        self.snapshots.clear();
        self.ranges.clear();
        self.resolved = false;
        // `buffers` is kept and reused by the next resolve.
        self.alpha_base = 0;
        self.alpha_upload_len = None;
        self.summaries.clear();
        self.direct.clear();
        self.instance_spans.clear();
    }

    /// Index of the snapshot of the current clip, taking it if this clip has not been seen.
    fn snapshot(&mut self, key: (u64, usize), clip: PathDataRef<'_>) -> usize {
        if let Some(last) = self.snapshots.last()
            && last.key == key
        {
            return self.snapshots.len() - 1;
        }
        self.snapshots.push(ClipSnapshot {
            key,
            strips: clip.strips.to_vec(),
            alphas: clip.alphas.to_vec(),
            bbox: clip.bbox,
        });
        self.snapshots.len() - 1
    }
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(
            width,
            height,
            Level::try_detect().unwrap_or(Level::baseline()),
        )
    }

    /// Create a new render context with a specific SIMD level.
    pub fn new_with(width: u16, height: u16, level: Level) -> Self {
        Self {
            width,
            height,
            viewport_state: ViewportState::new(width, height, level),
            render_state: RenderState::default(),
            root_transforms: RootTransforms::default(),
            aliasing_threshold: None,
            encoded_paints: vec![],
            strip_storage: RefCell::new(StripStorage::new(GenerationMode::Append)),
            filter: None,
            recorder: CommandRecorder::new(width, height),
            #[cfg(feature = "wgpu")]
            level,
            #[cfg(feature = "wgpu")]
            gpu_strips: None,
        }
    }

    /// Generate the strips of filled and stroked paths on the GPU instead of the CPU.
    ///
    /// Experimental. Flattening, tiling and sorting stay on the CPU; the GPU produces the strips
    /// and alphas, which are currently read back before scheduling so that the rest of the
    /// renderer is unchanged. Must be set before recording; [`Self::reset`] keeps the setting.
    /// Only the wgpu backend produces GPU strips.
    #[cfg(feature = "wgpu")]
    pub fn set_gpu_strips(&mut self, enabled: bool) {
        self.gpu_strips = enabled.then(|| RefCell::new(GpuStrips::new(self.width, self.height)));
    }

    /// Whether GPU strip generation is enabled.
    #[cfg(feature = "wgpu")]
    pub fn gpu_strips_enabled(&self) -> bool {
        self.gpu_strips.is_some()
    }

    /// Record a path for GPU strip generation, or on the CPU when the GPU cannot represent it.
    ///
    /// `prepare` flattens, tiles and sorts the path with the given cull bounding box.
    #[cfg(feature = "wgpu")]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the GPU input indexes paths with u32"
    )]
    fn record_gpu_path(
        &mut self,
        paint: Paint,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
        prepare: impl FnOnce(&mut StripGenerator, Option<RectU16>),
    ) {
        enum Outcome {
            None,
            Cpu(Range<usize>),
            Gpu(u32, RectU16),
        }
        let gpu_cell = self.gpu_strips.as_ref().expect("GPU strips are enabled");
        let mut gpu = gpu_cell.borrow_mut();
        let clip_key = self.viewport_state.clip_key();
        let (width, height) = (self.width, self.height);
        let strip_storage = &self.strip_storage;
        // Only unclipped, solid-paint paths are drawn straight from GPU buffers.
        let direct_paint = matches!(paint, Paint::Solid(_));
        let outcome = self
            .viewport_state
            .with_generator_and_clip(|generator, clip| {
                prepare(generator, clip.map(|clip| clip.path.bbox));
                // Paths inside filter root viewports are tiled against a padded viewport that
                // the GPU batch does not know about; render those on the CPU.
                let added = if generator.width() == width && generator.height() == height {
                    gpu.input.add_path(
                        generator.tiles(),
                        generator.lines(),
                        fill_rule,
                        aliasing_threshold,
                    )
                } else {
                    AddPath::CpuFallback
                };
                match added {
                    AddPath::Empty => Outcome::None,
                    AddPath::CpuFallback => {
                        let mut strip_storage = strip_storage.borrow_mut();
                        let start = strip_storage.strips.len();
                        generator.render_prepared(
                            fill_rule,
                            aliasing_threshold,
                            &mut strip_storage,
                            clip.map(|clip| clip.path),
                        );
                        Outcome::Cpu(start..strip_storage.strips.len())
                    }
                    AddPath::Added => {
                        let summary = summarize_path_tiles(
                            generator.tiles(),
                            generator.lines(),
                            fill_rule,
                            width,
                            height,
                        );
                        let mut bbox = summary.bbox;
                        let snapshot = clip.map(|clip| {
                            // The intersection's bbox is at most the path's within the clip's.
                            bbox = bbox.intersect(clip.path.bbox);
                            gpu.snapshot(clip_key, clip.path)
                        });
                        gpu.direct.push(direct_paint && snapshot.is_none());
                        gpu.summaries.push(summary);
                        gpu.clips.push(snapshot);
                        Outcome::Gpu(gpu.clips.len() as u32 - 1, bbox)
                    }
                }
            });
        drop(gpu);
        match outcome {
            Outcome::None => {}
            Outcome::Cpu(strips) => {
                if strips.is_empty() {
                    return;
                }
                let draw = RecordedDraw::new_path(strips.clone(), paint);
                let strip_storage = self.strip_storage.borrow();
                self.recorder.push_draw(draw, &strip_storage.strips[strips]);
            }
            Outcome::Gpu(path, bbox) => {
                let draw = RecordedDraw::GpuPath(RecordedGpuPath { path, bbox, paint });
                self.recorder.push_draw(draw, &[]);
            }
        }
    }

    /// Produce the strips of all recorded GPU paths and append them to the strip storage.
    ///
    /// Submits the GPU passes. When some path needs its strips on the CPU (clipped, non-solid
    /// paint, or in a layer), waits for them and reads the strips back; the alphas stay on the GPU
    /// and the renderer copies them into the alpha texture, at `alpha_row_bytes`-aligned rows
    /// after the CPU alphas (the scene's alpha buffer is padded to reserve that region). Paths
    /// recorded under a clip are intersected with the clip's strips on the CPU, which needs
    /// their alphas read back too. Idempotent per recording.
    #[cfg(feature = "wgpu")]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "alpha indices are u32 in strips"
    )]
    pub(crate) fn resolve_gpu_strips(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &crate::sparse_strips::gpu::SparseStripPipelines,
        alpha_row_bytes: u32,
    ) {
        use crate::sparse_strips::TILE_PIXELS;
        let Some(gpu_cell) = &self.gpu_strips else {
            return;
        };
        let mut gpu = gpu_cell.borrow_mut();
        if gpu.resolved {
            return;
        }
        gpu.resolved = true;
        let gpu = &mut *gpu;
        gpu.ranges.clear();
        if gpu.input.entries.is_empty() {
            return;
        }
        let mut storage = self.strip_storage.borrow_mut();
        let row_bytes = alpha_row_bytes.max(TILE_PIXELS) as usize;

        // Reserve the GPU alpha region: row-aligned, sized to the block bound.
        let alpha_base = storage.alphas.len().next_multiple_of(row_bytes);
        storage.alphas.resize(alpha_base, 0);
        // Direct paths get contiguous instance slots in path order; the scheduler draws each
        // span with a plain instanced draw.
        gpu.instance_spans.clear();
        let mut n_instances = 0_u32;
        for (summary, direct) in gpu.summaries.iter().zip(&gpu.direct) {
            if *direct {
                gpu.instance_spans
                    .push(Some([n_instances, summary.instances]));
                n_instances += summary.instances;
            } else {
                gpu.instance_spans.push(None);
            }
        }
        // The summaries give exact totals, so the buffers are sized to them.
        let (total_strips, total_blocks) = gpu.summaries.iter().fold((0, 0), |(s, b), summary| {
            (s + summary.strips, b + summary.blocks)
        });
        gpu.input.exact_counts = Some((total_strips, total_blocks));
        let buffers = pipelines.prepare_buffers(
            device,
            queue,
            gpu.buffers.take(),
            &gpu.input,
            (alpha_base / TILE_PIXELS as usize) as u32,
            alpha_row_bytes,
            n_instances,
        );
        storage
            .alphas
            .resize(alpha_base + buffers.alpha_rows as usize * row_bytes, 0);
        gpu.alpha_base = alpha_base;
        gpu.alpha_upload_len = Some(alpha_base);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse strips"),
        });
        pipelines.encode_timed(&mut encoder, &buffers, None, gpu.needs_readback());
        if !gpu.needs_readback() {
            // Everything is drawn from GPU buffers: submit the passes now so that the GPU
            // produces the strips while the CPU schedules the frame, and never wait for them.
            queue.submit([encoder.finish()]);
            gpu.buffers = Some(buffers);
            gpu.ranges.resize(gpu.input.paths.len(), 0..0);
            return;
        }
        let needs_alphas = gpu.clips.iter().any(Option::is_some);
        let out =
            crate::sparse_strips::gpu::read_back(device, queue, encoder, &buffers, needs_alphas);
        let buffers_alpha_bytes = buffers.alpha_rows as usize * row_bytes;
        gpu.buffers = Some(buffers);
        for (path, summary) in out.paths.iter().zip(&gpu.summaries) {
            debug_assert_eq!(
                (
                    path.strip_end - path.strip_start,
                    path.block_end - path.block_start
                ),
                (summary.strips, summary.blocks),
                "CPU path summary matches the GPU allocation"
            );
        }

        let unclipped_bbox = RectU16::new(0, 0, u16::MAX, u16::MAX);
        let alpha_base_u32 = alpha_base as u32;
        for (path, clip) in out.paths.iter().zip(&gpu.clips) {
            let range = path.strip_start as usize..path.strip_end as usize;
            let start = storage.strips.len();
            match clip {
                None => storage.strips.extend_from_slice(&out.strips[range]),
                Some(snapshot) => {
                    // The intersection reads alphas relative to the path's own buffer, so
                    // present the GPU strips with their alpha indices rebased to it.
                    let rebased: Vec<Strip> = out.strips[range]
                        .iter()
                        .map(|strip| {
                            let mut strip = *strip;
                            strip.set_alpha_idx(strip.alpha_idx() - alpha_base_u32);
                            strip
                        })
                        .collect();
                    let snapshot = &gpu.snapshots[*snapshot];
                    vello_common::clip::intersect(
                        self.level,
                        PathDataRef {
                            strips: &snapshot.strips,
                            alphas: &snapshot.alphas,
                            bbox: snapshot.bbox,
                        },
                        PathDataRef {
                            strips: &rebased,
                            alphas: &out.alphas,
                            bbox: unclipped_bbox,
                        },
                        &mut storage,
                    );
                }
            }
            gpu.ranges.push(start..storage.strips.len());
        }
        if storage.alphas.len() > alpha_base + buffers_alpha_bytes {
            // Intersections appended alphas after the GPU region; upload everything.
            gpu.alpha_upload_len = None;
        }
    }

    /// How many bytes of the scene's alphas the renderer must upload (the rest is written by
    /// the GPU), or `None` for all of them.
    #[cfg(feature = "wgpu")]
    pub(crate) fn gpu_alpha_upload_len(&self) -> Option<usize> {
        self.gpu_strips
            .as_ref()
            .and_then(|gpu| gpu.borrow().alpha_upload_len)
    }

    /// The GPU alpha copy the renderer must perform after uploading the CPU alphas: the
    /// buffer, the first alpha texture row to write, and the number of rows.
    #[cfg(feature = "wgpu")]
    #[expect(clippy::cast_possible_truncation, reason = "texture rows are u32")]
    pub(crate) fn gpu_alpha_copy(
        &self,
        alpha_row_bytes: u32,
    ) -> Option<(core::cell::Ref<'_, GpuStrips>, u32, u32)> {
        let gpu = self.gpu_strips.as_ref()?.borrow();
        if !gpu.resolved || gpu.input.entries.is_empty() {
            return None;
        }
        gpu.buffers.as_ref()?;
        let row = (gpu.alpha_base / alpha_row_bytes.max(1) as usize) as u32;
        let rows = gpu.buffers.as_ref().map_or(0, |b| b.alpha_rows);
        Some((gpu, row, rows))
    }

    fn active_rect(&self) -> Rect {
        Rect::new(
            0.0,
            0.0,
            f64::from(self.viewport_state.width()),
            f64::from(self.viewport_state.height()),
        )
    }

    fn transforms(&self) -> &Transforms {
        &self.render_state.transforms
    }

    fn transforms_mut(&mut self) -> &mut Transforms {
        &mut self.render_state.transforms
    }

    fn effective_path_transform(&self) -> Affine {
        self.root_transforms
            .effective_path_transform(self.transforms())
    }

    fn effective_paint_transform(&self) -> Affine {
        self.root_transforms
            .effective_paint_transform(self.transforms())
    }

    /// Encode the current paint into a `Paint` that can be used for rendering.
    ///
    /// For solid colors, this is a simple conversion. For gradients and images,
    /// this encodes the paint data into the `encoded_paints` buffer and returns
    /// a `Paint` that references that data. The combined transform (geometry + paint)
    /// is applied during encoding.
    ///
    /// Returns `None` if the current paint cannot cover any pixels (see [`Self::paint_has_area`]).
    fn encode_current_paint(&mut self) -> Option<Paint> {
        // Note: In vello_cpu, during fine rasterization we apply a 0.5 offset to the location
        // to account for the fact that we want to sample the pixel center instead of the top-left
        // corner. For vello_gpu, we don't need this, because the GPU itself already applies
        // this shift automatically.
        let transform = self.effective_paint_transform();
        match &self.render_state.paint {
            PaintType::Solid(s) => Some(Paint::Solid(PremulColor::from_alpha_color(*s))),
            PaintType::Gradient(g) => g.encode_into(&mut self.encoded_paints, transform, None),
            PaintType::Image(i) => {
                i.encode_into(&mut self.encoded_paints, transform, self.render_state.tint)
            }
        }
    }

    /// Whether the effective paint transform maps paint space onto a non-zero area.
    ///
    /// A singular transform collapses images, gradients and blurred rectangles onto a line or a
    /// point, so drawing with it must not touch any pixel. Checked before pushing a blend or
    /// filter layer so that a skipped draw does not leave an empty layer behind.
    fn paint_transform_has_area(&self) -> bool {
        invert_paint_transform(self.effective_paint_transform()).is_some()
    }

    /// Whether drawing with the current paint can produce any pixels.
    fn paint_has_area(&self) -> bool {
        matches!(self.render_state.paint, PaintType::Solid(_)) || self.paint_transform_has_area()
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            ctx.fill_path_with(
                path,
                ctx.effective_path_transform(),
                ctx.render_state.fill_rule,
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    /// Build strips for a filled path with the given properties and record the draw.
    fn fill_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        fill_rule: Fill,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        #[cfg(feature = "wgpu")]
        if self.gpu_strips.is_some() {
            self.record_gpu_path(
                paint,
                fill_rule,
                aliasing_threshold,
                |generator, cull_bbox| {
                    generator.prepare_filled_path(path, transform, cull_bbox);
                },
            );
            return;
        }
        self.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
            strip_generator.generate_filled_path(
                path,
                fill_rule,
                transform,
                aliasing_threshold,
                strip_storage,
                clip_path.map(|clip| clip.path),
            );
        });
    }

    /// Push a new clip path to the clip stack.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/vello_cpu/examples)
    /// example for how this method differs from `push_clip_layer`.
    pub fn push_clip_path(&mut self, path: &BezPath) {
        let transform = self.transforms().clip_path_transform();
        self.viewport_state.push_clip_path(
            path,
            self.render_state.fill_rule,
            transform,
            self.aliasing_threshold,
        );
    }

    /// Push a rectangular clip path.
    pub fn push_clip_rect(&mut self, rect: &Rect) {
        let transform = self.transforms().clip_path_transform();

        if let Some(rect) = into_fast_path_rect(*rect, &transform, self.aliasing_threshold) {
            self.viewport_state.push_clip_rect(&rect);
        } else {
            self.push_clip_path(&rect.to_path(DEFAULT_TOLERANCE));
        }
    }

    /// Pop the most recently pushed clip path or clip rectangle from the clip stack.
    ///
    /// Note that unlike `push_clip_layer`, it is permissible to have pending
    /// pushed clips before finishing the rendering operation.
    pub fn pop_clip(&mut self) {
        self.viewport_state.pop_clip();
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            ctx.stroke_path_with(
                path,
                ctx.effective_path_transform(),
                paint,
                ctx.aliasing_threshold,
            );
        });
    }

    /// Build strips for a stroked path with the given properties and record the draw.
    fn stroke_path_with(
        &mut self,
        path: &BezPath,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    ) {
        let stroke = self.render_state.stroke.clone();
        #[cfg(feature = "wgpu")]
        if self.gpu_strips.is_some() {
            self.record_gpu_path(
                paint,
                Fill::NonZero,
                aliasing_threshold,
                |generator, cull_bbox| {
                    generator.prepare_stroked_path(path, &stroke, transform, cull_bbox);
                },
            );
            return;
        }
        self.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
            strip_generator.generate_stroked_path(
                path,
                &stroke,
                transform,
                aliasing_threshold,
                strip_storage,
                clip_path.map(|clip| clip.path),
            );
        });
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
    pub fn fill_rect(&mut self, rect: &Rect) {
        if !self.paint_has_area() || rect.is_zero_area() {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };

            if let Some(bounds) = ctx.fast_rect_bounds(rect) {
                ctx.recorder
                    .push_draw(RecordedDraw::new_rect(bounds, paint), &[]);

                return;
            }

            let transform = ctx.effective_path_transform();
            if let Some(transformed_rect) =
                into_fast_path_rect(*rect, &transform, ctx.aliasing_threshold)
            {
                ctx.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
                    strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        clip_path,
                    );
                });
            } else {
                // TODO: Use a temporary storage for rect paths, like in `vello_cpu`.
                ctx.fill_path_with(
                    &rect.to_path(DEFAULT_TOLERANCE),
                    transform,
                    ctx.render_state.fill_rule,
                    paint,
                    ctx.aliasing_threshold,
                );
            }
        });
    }

    fn record_generated_path<F>(&mut self, paint: Paint, generate: F)
    where
        F: FnOnce(&mut StripGenerator, &mut StripStorage, Option<ClipRef<'_>>),
    {
        let strips = {
            let mut strip_storage = self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.viewport_state
                .with_generator_and_clip(|strip_generator, clip_path| {
                    generate(strip_generator, &mut strip_storage, clip_path);
                });
            strip_start..strip_storage.strips.len()
        };

        if strips.is_empty() {
            return;
        }

        let draw = RecordedDraw::new_path(strips.clone(), paint);
        let strip_storage = self.strip_storage.borrow();
        self.recorder.push_draw(draw, &strip_storage.strips[strips]);
    }

    fn fast_rect_bounds(&self, rect: &Rect) -> Option<Rect> {
        // We can't handle skewed rectangles.
        // TODO: Maybe support rotated rectangles (https://github.com/linebender/vello/pull/1482#discussion_r2881223621)

        let transform = self.effective_path_transform();
        let mut transformed_rect = into_fast_path_rect(*rect, &transform, self.aliasing_threshold)?
            .intersect(self.active_rect());

        if let Some(clip) = self.viewport_state.clip() {
            let ClipShape::AxisAlignedRect(clip_rect) = clip.shape else {
                return None;
            };
            transformed_rect = transformed_rect.intersect(clip_rect);
        }

        // Can't handle mirrored or zero-sized rectangles.
        if transformed_rect.is_zero_area() {
            return None;
        }

        Some(transformed_rect)
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Fill a blurred rectangle with the given corner radius and standard deviation.
    ///
    /// When `invert` is `true`, the inverse (`1 - alpha`) of the blur coverage is painted: the
    /// paint is fully opaque outside the blurred rectangle and fades to transparent inside it. This
    /// can be used to implement inset box shadows.
    ///
    /// This operation uses the current transform and paint transform. Like Vello CPU, it only
    /// uses solid paints; non-solid paints fall back to black.
    #[track_caller]
    pub fn fill_blurred_rounded_rect(
        &mut self,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
        invert: bool,
    ) {
        if !cfg!(feature = "blurred_rounded_rect") {
            panic!("blurred rounded rectangles require the `blurred_rounded_rect` feature");
        }

        if !self.paint_transform_has_area() {
            return;
        }

        self.with_optional_filter_or_blend_layer(|ctx| {
            let rect = rect.abs();
            let color = match ctx.render_state.paint {
                PaintType::Solid(s) => s,
                _ => BLACK,
            };
            let blurred_rect = BlurredRoundedRectangle {
                rect,
                color,
                radius,
                std_dev,
                invert,
            };

            let kernel_size = 2.5 * std_dev;
            let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
            let transform = ctx.effective_paint_transform();
            let Some(paint) = blurred_rect.encode_into(&mut ctx.encoded_paints, transform, None)
            else {
                return;
            };

            if let Some(bounds) = ctx.fast_rect_bounds(&inflated_rect) {
                ctx.recorder
                    .push_draw(RecordedDraw::new_rect(bounds, paint), &[]);
                return;
            }

            let path_transform = ctx.effective_path_transform();
            if let Some(transformed_rect) =
                into_fast_path_rect(inflated_rect, &path_transform, ctx.aliasing_threshold)
            {
                ctx.record_generated_path(paint, |strip_generator, strip_storage, clip_path| {
                    strip_generator.generate_filled_rect_fast(
                        &transformed_rect,
                        strip_storage,
                        clip_path,
                    );
                });
            } else {
                ctx.fill_path_with(
                    &inflated_rect.to_path(DEFAULT_TOLERANCE),
                    path_transform,
                    Fill::NonZero,
                    paint,
                    ctx.aliasing_threshold,
                );
            }
        });
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    #[cfg(feature = "text")]
    pub fn glyph_run<'a>(
        &'a mut self,
        resources: &'a mut Resources,
        font: &FontData,
    ) -> GlyphRunBuilder<'a> {
        glifo::GlyphRunBuilder::new(
            font.clone(),
            self.transforms().scene_transform(),
            *self.transforms().paint_transform(),
            crate::text::GpuGlyphRunBackend {
                scene: self,
                resources,
                atlas_cache_enabled: false,
            },
        )
    }

    /// Push a new layer with the given properties.
    ///
    /// # Panics
    ///
    /// Panics if `mask` is provided because mask layers are not yet supported.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        if mask.is_some() {
            unimplemented!("mask layers are currently not supported");
        }

        let blend_mode = blend_mode.unwrap_or_default();
        let layer_transform = self.effective_path_transform();
        let filter_data = filter.map(|filter| FilterData::new(filter, layer_transform));
        let relative_root_transform =
            filter_data
                .as_ref()
                .map_or(Affine::IDENTITY, |filter_data| {
                    let (shift_x, shift_y) = filter_data.source_shift();
                    Affine::translate((f64::from(shift_x), f64::from(shift_y)))
                });
        self.root_transforms.push_root(relative_root_transform);
        if let Some(filter_plan) = &filter_data {
            self.viewport_state.push_root_viewport(filter_plan);
        }

        let clip_path = clip_path.map(|path| {
            let mut strip_storage = self.strip_storage.borrow_mut();
            let strip_start = strip_storage.strips.len();
            self.viewport_state
                .with_generator_and_clip(|strip_generator, existing_clip| {
                    strip_generator.generate_filled_path(
                        path,
                        self.render_state.fill_rule,
                        layer_transform,
                        self.aliasing_threshold,
                        &mut strip_storage,
                        existing_clip.map(|clip| clip.path),
                    );

                    let strip_range = strip_start..strip_storage.strips.len();
                    LayerClip {
                        bbox: strip_bbox(&strip_storage.strips[strip_range.clone()])
                            .unwrap_or(RectU16::ZERO),
                        strip_range,
                        thread_idx: 0,
                    }
                })
        });

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity: opacity.unwrap_or(1.0),
                mask: None,
                clip_path,
            },
            filter_data,
        );
    }

    /// Push a new clip layer.
    ///
    /// See the explanation in the [clipping](https://github.com/linebender/vello/tree/main/vello_cpu/examples)
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
    /// # Panics
    ///
    /// This method currently always panics since masks are not supported yet.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    /// Push a new filter layer.
    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    /// Pop the last pushed layer.
    pub fn pop_layer(&mut self) {
        if self.recorder.pop_layer() == PoppedLayer::Filter {
            self.viewport_state.pop_root_viewport();
        }
        self.root_transforms.pop_root();
    }

    /// Set the blend mode for subsequent rendering operations.
    ///
    /// # Panics
    ///
    /// Panics if `blend_mode` is destructive, which is currently not supported for
    /// non-isolated blends. You need to use clip layers instead if you need that behavior.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        assert!(
            !blend_mode.is_destructive(),
            "destructive blend modes are currently not supported"
        );

        self.render_state.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.render_state.stroke = stroke;
    }

    /// Get the current stroke.
    pub fn stroke(&self) -> &Stroke {
        &self.render_state.stroke
    }

    /// Get a mutable reference to the current stroke.
    #[cfg(feature = "text")]
    pub(crate) fn stroke_mut(&mut self) -> &mut Stroke {
        &mut self.render_state.stroke
    }

    /// Set the paint for subsequent rendering operations.
    // TODO: This API is not final. Supporting images from a pixmap is explicitly out of scope.
    //       Instead images should be passed via a backend-agnostic opaque id, and be hydrated at
    //       render time into a texture usable by the renderer backend.
    #[track_caller]
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        let paint = paint.into();
        assert_paint_supported(&paint);
        self.render_state.paint = paint;
    }

    /// Set the tint for subsequent image paint operations.
    pub fn set_tint(&mut self, tint: Option<Tint>) {
        self.render_state.tint = tint;
    }

    /// Clear the tint, so subsequent image paints are drawn without tinting.
    pub fn reset_tint(&mut self) {
        self.render_state.tint = None;
    }

    /// Get the current paint.
    pub fn paint(&self) -> &PaintType {
        &self.render_state.paint
    }
    /// Set the current paint transform.
    ///
    /// The paint transform is applied to the paint after the transform of the geometry the paint
    /// is drawn in, i.e., the paint transform is applied after the global transform. This allows
    /// transforming the paint independently from the drawn geometry.
    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.transforms_mut().set_paint_transform(paint_transform);
    }

    /// Reset the current paint transform.
    pub fn reset_paint_transform(&mut self) {
        self.transforms_mut().reset_paint_transform();
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.render_state.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transforms_mut().set_transform(transform);
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transforms_mut().reset_transform();
    }

    /// Apply filter to the current paint (affects next drawn element).
    pub fn set_filter_effect(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    /// Reset the current filter effect.
    pub fn reset_filter_effect(&mut self) {
        self.filter = None;
    }

    fn with_optional_filter_or_blend_layer<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let blend_mode = self.render_state.blend_mode;
        let blend_mode = (blend_mode != BlendMode::default()).then_some(blend_mode);
        let filter = self.filter.clone();

        if blend_mode.is_some() || filter.is_some() {
            self.push_layer(None, blend_mode, None, None, filter);
            let result = f(self);
            self.pop_layer();
            result
        } else {
            f(self)
        }
    }

    /// Reset the scene and update its size.
    pub fn reset_and_resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;

        self.reset();
    }

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.viewport_state.reset(self.width, self.height);
        {
            let mut ss = self.strip_storage.borrow_mut();
            ss.clear();
        }
        self.encoded_paints.clear();

        self.root_transforms.reset();
        self.render_state.reset();
        self.aliasing_threshold = None;
        self.recorder.reset(self.width, self.height);
        self.filter = None;
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = &self.gpu_strips {
            gpu.borrow_mut().reset(self.width, self.height);
        }
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Take current rendering state and reset the existing state to its default.
    pub fn take_current_state(&mut self) -> RenderState {
        core::mem::take(&mut self.render_state)
    }

    /// Save a copy of the current rendering state.
    pub fn save_current_state(&mut self) -> RenderState {
        self.render_state.clone()
    }

    /// Restore rendering state.
    #[track_caller]
    pub fn restore_state(&mut self, state: RenderState) {
        assert_paint_supported(&state.paint);

        self.render_state = state;
    }
}

#[cfg(feature = "probe")]
impl vello_common::probe::ProbeRenderer for Scene {
    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }

    fn set_paint(&mut self, paint: PaintType) {
        Self::set_paint(self, paint);
    }

    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32, invert: bool) {
        Self::fill_blurred_rounded_rect(self, rect, radius, std_dev, invert);
    }

    fn push_layer(&mut self, blend_mode: Option<BlendMode>, opacity: Option<f32>) {
        Self::push_layer(self, None, blend_mode, opacity, None, None);
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        Self::push_filter_layer(self, filter);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    fn set_paint_transform(&mut self, paint_transform: Affine) {
        Self::set_paint_transform(self, paint_transform);
    }

    fn reset_paint_transform(&mut self) {
        Self::reset_paint_transform(self);
    }
}

#[track_caller]
fn assert_paint_supported(paint: &PaintType) {
    match paint {
        PaintType::Solid(_) => {}
        PaintType::Gradient(gradient) => assert!(
            cfg!(feature = "gradient_sweep") || !matches!(gradient.kind, GradientKind::Sweep(_)),
            "sweep gradients require the `gradient_sweep` feature"
        ),
        PaintType::Image(image) => assert!(
            cfg!(feature = "image_bicubic") || image.sampler.quality != ImageQuality::High,
            "bicubic image sampling requires the `image_bicubic` feature"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{RecordedDraw, Scene};
    #[cfg(feature = "text")]
    use crate::resources::Resources;
    #[cfg(feature = "text")]
    use alloc::sync::Arc;
    #[cfg(feature = "text")]
    use glifo::Glyph;
    use vello_common::TextureId;
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{BezPath, Rect};
    use vello_common::paint::{Image, ImageSource, Paint, PremulColor};
    use vello_common::peniko::ImageSampler;
    use vello_common::peniko::color::palette::css::BLUE;
    #[cfg(feature = "text")]
    use vello_common::peniko::{Blob, FontData};
    use vello_common::record::Drawable;

    #[test]
    fn recorded_rect_bbox_rounds_fractional_outward() {
        let draw = RecordedDraw::new_rect(
            Rect::new(0.0, 0.0, 4.5, 4.5),
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
        );

        assert_eq!(draw.bbox(&[]), Some(RectU16::new(0, 0, 8, 8)));
    }

    #[test]
    fn rectangular_clip_keeps_direct_rect_draw_path() {
        let mut scene = Scene::new(100, 100);
        scene.push_clip_rect(&Rect::new(10.25, 5.5, 80.75, 90.0));
        scene.push_clip_rect(&Rect::new(20.5, 0.0, 70.25, 60.75));

        scene.fill_rect(&Rect::new(0.0, 20.25, 50.5, 80.0));

        let RecordedDraw::Rect(rect) = &scene.recorder.draws[0] else {
            panic!("expected a directly recorded rectangle");
        };
        assert_eq!(rect.rect, Rect::new(20.5, 20.25, 50.5, 60.75));
        assert!(scene.strip_storage.borrow().strips.is_empty());
    }

    #[test]
    fn reset_restores_default_aliasing_threshold() {
        let mut scene = Scene::new(4, 4);
        scene.set_aliasing_threshold(Some(128));

        scene.reset();

        assert_eq!(scene.aliasing_threshold, None);
    }

    #[test]
    fn reset_and_resize_updates_scene_size() {
        let mut scene = Scene::new(8, 4);

        scene.reset_and_resize(4, 8);
        assert_eq!(scene.width(), 4);
        assert_eq!(scene.height(), 8);
        assert_eq!(scene.recorder.scene_size.width(), 4);
        assert_eq!(scene.recorder.scene_size.height(), 8);

        scene.set_paint(BLUE);
        scene.fill_rect(&Rect::new(0.0, 0.0, 8.0, 8.0));

        let RecordedDraw::Rect(rect) = &scene.recorder.draws[0] else {
            panic!("expected a recorded rectangle");
        };
        assert_eq!(rect.rect, Rect::new(0.0, 0.0, 4.0, 8.0));
    }

    #[test]
    fn empty_generated_path_is_not_recorded() {
        let mut scene = Scene::new(100, 100);

        scene.fill_path(&BezPath::new());

        assert!(scene.strip_storage.borrow().strips.is_empty());
        assert!(scene.recorder.draws.is_empty());
        assert!(scene.recorder.nodes.is_empty());
    }

    #[test]
    fn zero_area_external_texture_rect_is_a_noop() {
        let mut scene = Scene::new(100, 100);
        scene.set_paint(Image {
            image: ImageSource::external_texture(TextureId(7), RectU16::new(0, 0, 8, 8), false),
            sampler: ImageSampler::default(),
        });

        scene.fill_rect(&Rect::new(10.0, 10.0, 10.0, 20.0));

        assert!(scene.encoded_paints.is_empty());
        assert!(scene.strip_storage.borrow().strips.is_empty());
        assert!(scene.recorder.draws.is_empty());
        assert!(scene.recorder.nodes.is_empty());
    }

    #[cfg(feature = "text")]
    #[test]
    fn glyph_atlas_resources_are_lazy() {
        const ROBOTO_FONT: &[u8] = include_bytes!("../../assets/roboto/Roboto-Regular.ttf");

        let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);
        let glyphs = [Glyph {
            id: 1,
            x: 0.0,
            y: 0.0,
        }];

        let mut scene = Scene::new(200, 200);
        let mut resources = Resources::new(vello_common::multi_atlas::AtlasConfig::default());
        let mut triangle = BezPath::new();
        triangle.move_to((10.0, 10.0));
        triangle.line_to((90.0, 50.0));
        triangle.line_to((10.0, 90.0));
        triangle.close_path();

        scene.fill_rect(&Rect::new(10.0, 10.0, 50.0, 50.0));
        scene.fill_path(&triangle);
        scene
            .glyph_run(&mut resources, &font)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();

        assert!(resources.glyph_resources.is_none());

        scene
            .glyph_run(&mut resources, &font)
            .atlas_cache(true)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();

        assert!(resources.glyph_resources.is_some());
    }
}
