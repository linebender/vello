// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use std::ops::Range;
use std::u32;

use crate::render::{GpuStrip, RenderData};
use vello_common::coarse::{self, CmdAlphaFill, CmdFill, Wide, WideTile};
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::paint::Paint;
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};
use vello_common::{flatten, strip};

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render state which contains the style properties for path rendering and
/// the current transform.
#[derive(Debug)]
struct RenderState {
    pub(crate) paint: Paint,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: Paint,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let render_state = Self::default_render_state();
        Self {
            width,
            height,
            wide: Wide::new(width, height),
            alphas: vec![],
            line_buf: vec![],
            tiles: Tiles::new(),
            strip_buf: vec![],
            paint: render_state.paint,
            stroke: render_state.stroke,
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
        }
    }

    /// Create default rendering state.
    fn default_render_state() -> RenderState {
        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
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
            stroke,
            blend_mode,
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        self.render_path(self.fill_rule, self.paint.clone());
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, self.paint.clone());
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
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
    pub fn set_paint(&mut self, paint: Paint) {
        self.paint = paint;
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

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.alphas.clear();
        self.line_buf.clear();
        self.tiles.reset();
        self.strip_buf.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );

        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }
}

impl Scene {
    /// Prepares render data from the current context for GPU rendering
    ///
    /// This method converts the rendering context's state into a format
    /// suitable for GPU rendering, including strips and alpha values.
    pub fn prepare_render_data(&self) -> RenderData {
        let mut strips: Vec<GpuStrip> = Vec::new();
        let wide_tiles_per_row = (self.width).div_ceil(WideTile::WIDTH);
        let wide_tiles_per_col = (self.height).div_ceil(Tile::HEIGHT);
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile_idx = usize::from(wide_tile_row) * usize::from(wide_tiles_per_row)
                    + usize::from(wide_tile_col);
                let wide_tile = &self.wide.tiles[wide_tile_idx];
                process_wide_tile(wide_tile);
            }
        }
        panic!();
        RenderData {
            strips,
            alphas: self.alphas.clone(),
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum DrawKind {
    Solid,
    Other,
}

#[derive(Debug)]
enum Step {
    DrawBackground {},
    Draw {
        kind: DrawKind,
        commands: Range<u32>,
    },
    Blend {
        /// The "destination" layer of the blend.
        /// That is, the layer whose contents will be used as the "dest" in the blend.
        /// Note that this is *NOT* where the output will actually be placed (necessarily).
        /// That is, this is the layer which contains the current "state" *before* the `PushClip` operation happens.
        dest: u32,
        /// The "source" layer of the blend.
        /// That is, the layer whose contents will be used as the "source" in the blend.
        source: u32,
        /// The range of commands in the [`WideTile::cmds`] which "operate" the blend.
        commands: Range<u32>,
    },
}

#[derive(Debug)]
struct Layer {
    depth: u32,
    steps: Range<u32>,
}

fn process_wide_tile(wide_tile: &WideTile) {
    let bg = wide_tile.bg.to_u32();

    let mut current_kind = None;
    let mut current_end = 0;

    let mut steps = Vec::new();
    let mut layers: Vec<Layer> = Vec::new();
    let mut steps_temp_stack = Vec::new();
    let mut current_depth = 0;
    let mut active_blend_layer_indices: Vec<usize> = Vec::new();

    // We iterate through the commands in reverse. This allows us to track the current "depth" accurately
    // (later blends being "realised" implicitly rely on everything came before).
    for (idx, cmd) in wide_tile.cmds.iter().enumerate().rev() {
        let idx: u32 = idx.try_into().unwrap();
        match cmd {
            coarse::Cmd::Fill(CmdFill { paint, .. })
            | coarse::Cmd::AlphaFill(CmdAlphaFill { paint, .. }) => {
                let kind = match paint {
                    Paint::Solid(_) => DrawKind::Solid,
                    // TODO: Resolve to a more specific indexed kind.
                    Paint::Indexed(_) => DrawKind::Other,
                };
                if let Some(current_kind) = current_kind {
                    if current_kind != kind {
                        steps_temp_stack.push(Step::Draw {
                            kind: current_kind,
                            commands: idx..current_end,
                        });
                        current_end = idx;
                    }
                } else {
                    current_end = idx;
                }
                current_kind = Some(kind);
            }
            coarse::Cmd::PopClip => {
                if let Some(current_kind) = current_kind.take() {
                    steps_temp_stack.push(Step::Draw {
                        kind: current_kind,
                        commands: idx..current_end,
                    });
                    // We don't need to update `current_end`, as taking `current_kind` handles that.
                }
                let layer_steps_start = steps.len().try_into().unwrap();
                let this_idx = layers.len();
                steps.push(Step::Blend {
                    // Tombstone: Will be updated when handling `PushClip`.
                    dest: u32::MAX,
                    // The "source" layer is the next layer we create (after the one we're about to push).
                    source: u32::try_from(this_idx).unwrap() + 1,
                    // If there is a `ClipFill` or `ClipStrip`, the start will be decremented.
                    commands: idx..idx,
                });
                steps.extend(steps_temp_stack.drain(..).rev());

                // Push the new layer's index to the stack.

                active_blend_layer_indices.push(this_idx);
                let next_step: u32 = steps.len().try_into().unwrap();
                // Create a Layer for the current layer, i.e. the one whose drawing commands we just processed, whose "first" step is the blend.
                layers.push(Layer {
                    depth: current_depth,
                    steps: layer_steps_start..next_step,
                });
                current_depth += 1;
            }
            coarse::Cmd::ClipFill(_) | coarse::Cmd::ClipStrip(_) => {
                let this_layer = layers.last_mut().unwrap();
                let Step::Blend { commands, .. } =
                    &mut steps[usize::try_from(this_layer.steps.start).unwrap()]
                else {
                    unreachable!();
                };
                debug_assert_eq!(
                    commands.start,
                    idx + 1,
                    "Should only be `ClipFill` or `ClipStrip` before a `PopClip`"
                );
                commands.start = idx;
            }
            coarse::Cmd::PushClip => {
                if let Some(current_kind) = current_kind.take() {
                    steps_temp_stack.push(Step::Draw {
                        kind: current_kind,
                        commands: idx..current_end,
                    });
                }
                let new_depth = {
                    // Resolve the source tombstone in the blend we're "opening".
                    let pop_idx = active_blend_layer_indices.pop().unwrap();
                    let layer = &mut layers[pop_idx];
                    let Step::Blend { dest, .. } =
                        &mut steps[usize::try_from(layer.steps.start).unwrap()]
                    else {
                        unreachable!()
                    };
                    debug_assert_eq!(*dest, u32::MAX, "Each tomstone should only be set once");
                    let layer_depth = layer.depth;
                    // Mark the dest as the next layer to be created (i.e. not the one we're about to make).
                    *dest = u32::try_from(layers.len()).unwrap() + 1;
                    layer_depth
                };
                // Finalise the layer which we just finished.
                // Note that this is *not* necessarily the source layer, because there could have been other clips in between.
                let layer_steps_start = steps.len().try_into().unwrap();
                steps.extend(steps_temp_stack.drain(..).rev());
                let layer_steps_end: u32 = steps.len().try_into().unwrap();
                layers.push(Layer {
                    depth: current_depth,
                    steps: layer_steps_start..layer_steps_end,
                });
                // Set the depth to the same as the `dest` of this blend.
                current_depth = new_depth + 1;
            }
        }
    }
    let layer_steps_start = steps.len().try_into().unwrap();
    if let Some(current_kind) = current_kind.take() {
        steps_temp_stack.push(Step::Draw {
            kind: current_kind,
            commands: 0..current_end,
        });
    }
    if bg != 0 {
        steps.push(Step::DrawBackground {});
    }
    steps.extend(steps_temp_stack.drain(..).rev());
    let layer_steps_end: u32 = steps.len().try_into().unwrap();
    layers.push(Layer {
        depth: current_depth,
        steps: layer_steps_start..layer_steps_end,
    });
    debug_assert!(
        active_blend_layer_indices.is_empty(),
        "All PopClips should have a corresponding PushClips"
    );
    dbg!(layers);
    dbg!(steps);
}

impl GlyphRenderer for Scene {
    fn fill_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph {
            PreparedGlyph::Outline(glyph) => {
                flatten::fill(glyph.path, glyph.transform, &mut self.line_buf);
                self.render_path(Fill::NonZero, self.paint.clone());
            }
        }
    }

    fn stroke_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph {
            PreparedGlyph::Outline(glyph) => {
                flatten::stroke(
                    glyph.path,
                    &self.stroke,
                    glyph.transform,
                    &mut self.line_buf,
                );
                self.render_path(Fill::NonZero, self.paint.clone());
            }
        }
    }
}
