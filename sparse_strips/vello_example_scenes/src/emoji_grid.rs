// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interactive COLR emoji grid scene.

use core::fmt;
use std::sync::Arc;

use glifo::Glyph;
use skrifa::instance::{LocationRef, Size};
use skrifa::raw::FileRef;
use skrifa::raw::TableProvider;
use skrifa::{GlyphId, MetadataProvider};
use vello_common::kurbo::{Affine, Rect};
use vello_common::peniko::{Blob, FontData};

use crate::{ExampleScene, RenderingContext};

const FALLBACK_NOTO_COLR_FONT: &[u8] =
    include_bytes!("../../../examples/assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");
// If you want wasm builds to embed a custom COLR font and can guarantee
// `NOTO_COLR_PATH` is set at compile time, replace the constant above with:
//
// const FALLBACK_NOTO_COLR_FONT: &[u8] = include_bytes!(env!("NOTO_COLR_PATH"));
#[cfg(not(target_arch = "wasm32"))]
const NOTO_COLR_PATH_ENV: &str = "NOTO_COLR_PATH";
const EMOJI_COLUMNS: usize = 30;
const MIN_EMOJI_COUNT: usize = 1;
const WINDOW_STEP: usize = 100;
const INITIAL_EMOJI_COUNT: usize = 10;
const SIDE_PADDING: f32 = 16.0;
const TOP_PADDING: f32 = 24.0;
const MIN_CELL_WIDTH: f32 = 8.0;

/// Scene drawing a large adjustable grid of COLR emoji.
pub struct EmojiGridScene {
    font: FontData,
    glyphs: Vec<EmojiGlyph>,
    upem: f32,
    unit_cell_width: f32,
    unit_cell_height: f32,
    emoji_count: usize,
    window_start: usize,
    glyph_caching_enabled: bool,
}

#[derive(Clone, Copy, Debug)]
struct EmojiGlyph {
    id: u32,
    bbox_x0: f32,
    bbox_y1: f32,
}

impl fmt::Debug for EmojiGridScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmojiGridScene")
            .field("emoji_count", &self.emoji_count)
            .finish_non_exhaustive()
    }
}

impl Default for EmojiGridScene {
    fn default() -> Self {
        Self::new()
    }
}

impl EmojiGridScene {
    /// Create a new `EmojiGridScene`.
    pub fn new() -> Self {
        let font = load_emoji_font();
        let (glyphs, upem, unit_cell_width, unit_cell_height) = emoji_font_metrics(&font);
        let emoji_count = INITIAL_EMOJI_COUNT.min(glyphs.len());
        Self {
            font,
            glyphs,
            upem,
            unit_cell_width,
            unit_cell_height,
            emoji_count,
            window_start: 0,
            glyph_caching_enabled: false,
        }
    }

    fn set_count(&mut self, emoji_count: usize) {
        self.emoji_count = emoji_count.clamp(MIN_EMOJI_COUNT, self.glyphs.len());
        self.window_start = self.clamp_window_start(self.window_start);
    }

    fn increase_step(&self) -> usize {
        magnitude_step_at_or_below(self.emoji_count)
    }

    fn decrease_step(&self) -> usize {
        magnitude_step_strictly_below(self.emoji_count)
    }

    fn shift_window_forward(&mut self) {
        self.window_start = self.clamp_window_start(self.window_start.saturating_add(WINDOW_STEP));
    }

    fn shift_window_backward(&mut self) {
        self.window_start = self.window_start.saturating_sub(WINDOW_STEP);
    }

    fn clamp_window_start(&self, window_start: usize) -> usize {
        window_start.min(self.max_window_start())
    }

    fn max_window_start(&self) -> usize {
        self.glyphs.len().saturating_sub(self.emoji_count)
    }

    fn build_glyphs(&self, width: f32) -> (f32, Vec<Glyph>) {
        let available_width = (width - 2.0 * SIDE_PADDING).max(MIN_CELL_WIDTH);
        let cell_width = (available_width / EMOJI_COLUMNS as f32).max(MIN_CELL_WIDTH);
        let scale = cell_width / self.unit_cell_width.max(f32::EPSILON);
        let font_size = self.upem * scale;
        let cell_height = self.unit_cell_height * scale;
        let glyphs = (0..self.emoji_count)
            .map(|index| {
                let column = index % EMOJI_COLUMNS;
                let row = index / EMOJI_COLUMNS;
                let glyph = self.glyphs[self.window_start + index];
                Glyph {
                    id: glyph.id,
                    x: SIDE_PADDING + column as f32 * cell_width - glyph.bbox_x0 * scale,
                    y: TOP_PADDING + row as f32 * cell_height + glyph.bbox_y1 * scale,
                }
            })
            .collect();
        (font_size, glyphs)
    }
}

impl ExampleScene for EmojiGridScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        ctx.set_transform(root_transform);
        let (font_size, glyphs) = self.build_glyphs(f32::from(ctx.width()));
        ctx.glyph_run(resources, &self.font)
            .font_size(font_size)
            .hint(false)
            .atlas_cache(self.glyph_caching_enabled)
            .fill_glyphs(glyphs.into_iter());
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "+" | "=" => {
                self.set_count(self.emoji_count.saturating_add(self.increase_step()));
                true
            }
            "-" => {
                self.set_count(self.emoji_count.saturating_sub(self.decrease_step()));
                true
            }
            "l" | "L" => {
                self.shift_window_forward();
                true
            }
            "k" | "K" => {
                self.shift_window_backward();
                true
            }
            "c" | "C" => {
                self.glyph_caching_enabled = !self.glyph_caching_enabled;
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        let window_end = self.window_start + self.emoji_count;
        Some(format!(
            "Emoji grid: {} / {} glyphs | window: {}..{} | columns: {} | cache: {} | +/-: +{} / -{} | l/k: shift {} | c: toggle caching",
            self.emoji_count,
            self.glyphs.len(),
            self.window_start,
            window_end,
            EMOJI_COLUMNS,
            if self.glyph_caching_enabled {
                "on"
            } else {
                "off"
            },
            self.increase_step(),
            self.decrease_step(),
            WINDOW_STEP
        ))
    }
}

fn magnitude_step_at_or_below(value: usize) -> usize {
    let mut step = 1;
    while step <= value / 10 {
        step *= 10;
    }
    step
}

fn magnitude_step_strictly_below(value: usize) -> usize {
    magnitude_step_at_or_below(value.saturating_sub(1).max(1))
}

fn load_emoji_font() -> FontData {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(bytes) =
        std::env::var_os(NOTO_COLR_PATH_ENV).and_then(|path| std::fs::read(path).ok())
    {
        return FontData::new(Blob::new(Arc::new(bytes)), 0);
    }

    FontData::new(Blob::new(Arc::new(FALLBACK_NOTO_COLR_FONT)), 0)
}

fn emoji_font_metrics(font: &FontData) -> (Vec<EmojiGlyph>, f32, f32, f32) {
    let font_ref = {
        let file_ref = FileRef::new(font.data.as_ref()).expect("emoji font is valid");
        match file_ref {
            FileRef::Font(font_ref) => font_ref,
            FileRef::Collection(collection) => collection
                .get(font.index)
                .expect("emoji font index is valid"),
        }
    };
    let upem = f32::from(
        font_ref
            .head()
            .expect("emoji font head table")
            .units_per_em(),
    );
    let fallback_bbox = Rect::new(0.0, 0.0, f64::from(upem), f64::from(upem));
    let color_glyphs = font_ref.color_glyphs();
    let glyph_count = u32::from(font_ref.maxp().expect("emoji font maxp table").num_glyphs());

    let mut glyphs = Vec::new();
    let mut unit_cell_width = 0.0_f32;
    let mut unit_cell_height = 0.0_f32;

    for glyph_index in 0..glyph_count {
        let glyph_id = GlyphId::new(glyph_index);
        let Some(color_glyph) = color_glyphs.get(glyph_id) else {
            continue;
        };
        let bbox = color_glyph
            .bounding_box(LocationRef::default(), Size::unscaled())
            .map(convert_bbox)
            .unwrap_or(fallback_bbox);
        unit_cell_width = unit_cell_width.max(f64_to_f32(bbox.width()));
        unit_cell_height = unit_cell_height.max(f64_to_f32(bbox.height()));
        glyphs.push(EmojiGlyph {
            id: glyph_id.to_u32(),
            bbox_x0: f64_to_f32(bbox.x0),
            bbox_y1: f64_to_f32(bbox.y1),
        });
    }

    (
        glyphs,
        upem,
        unit_cell_width.max(1.0),
        unit_cell_height.max(1.0),
    )
}

fn convert_bbox(bbox: skrifa::raw::types::BoundingBox<f32>) -> Rect {
    Rect::new(
        f64::from(bbox.x_min),
        f64::from(bbox.y_min),
        f64::from(bbox.x_max),
        f64::from(bbox.y_max),
    )
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "emoji font metrics are small enough to fit in f32 for this demo scene"
)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}
