// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An example integrating Cosmic text with Vello.
//! This renders a read-only snapshot of a simplified text editor using system fonts.

use crate::{SceneParams, TestScene};

use vello::kurbo::{Affine, Point, Rect, Size};
use vello::peniko::{Blob, Color, Fill, Font};
use vello::{Glyph, Scene};

use cosmic_text::fontdb::ID;
use cosmic_text::{
    Attrs, Buffer, Cursor, Edit, Editor, FontSystem, LayoutRun, Metrics, Selection, Shaping,
};

use unicode_segmentation::UnicodeSegmentation;

use std::cmp;
use std::collections::HashMap;
use std::sync::Arc;

impl TestScene for CosmicTextScene {
    fn render(&mut self, scene: &mut Scene, _scene_params: &mut SceneParams) {
        let buffer_glyphs = &self.buffer_glyphs;
        let vello_fonts = &self.vello_fonts;

        let text_transform = Affine::translate((500.0, 300.0));

        // Draw the Glyphs
        for buffer_line in &buffer_glyphs.buffer_lines {
            for glyph_highlight in &buffer_line.glyph_highlights {
                scene.fill(
                    Fill::NonZero,
                    text_transform,
                    buffer_glyphs.glyph_highlight_color,
                    None,
                    glyph_highlight,
                );
            }

            if let Some(cursor) = &buffer_line.cursor {
                scene.fill(
                    Fill::NonZero,
                    text_transform,
                    buffer_glyphs.cursor_color,
                    None,
                    cursor,
                );
            }

            for glyph_run in &buffer_line.glyph_runs {
                let font = vello_fonts.get(&glyph_run.font).unwrap();
                let glyph_color = glyph_run.glyph_color;
                let glyphs = glyph_run.glyphs.clone();
                scene
                    .draw_glyphs(font)
                    .font_size(buffer_glyphs.font_size)
                    .brush(glyph_color)
                    .transform(text_transform)
                    .draw(Fill::NonZero, glyphs.into_iter());
            }
        }
    }
}

pub struct CosmicTextScene {
    pub font_system: FontSystem,
    pub vello_fonts: HashMap<ID, Font>,
    pub buffer_glyphs: BufferGlyphs,
}

impl Default for CosmicTextScene {
    fn default() -> Self {
        Self::new()
    }
}

struct CosmicFontBlobAdapter {
    font: Arc<cosmic_text::Font>,
}

/// Adapter to allow `cosmic_text::Font` to be used as a Blob.
impl CosmicFontBlobAdapter {
    fn new(font: Arc<cosmic_text::Font>) -> Self {
        Self { font }
    }
}

impl AsRef<[u8]> for CosmicFontBlobAdapter {
    fn as_ref(&self) -> &[u8] {
        self.font.data()
    }
}

impl CosmicTextScene {
    pub fn new() -> Self {
        let mut font_system = FontSystem::new();
        let mut vello_fonts = HashMap::new();

        // Copy fonts from cosmic_text, so vello can use them
        let font_faces: Vec<(ID, u32)> = font_system
            .db()
            .faces()
            .map(|face| (face.id, face.index))
            .collect();

        for (font_id, index) in font_faces {
            if let Some(font) = font_system.get_font(font_id) {
                // For now use an adapter, to avoid cloning the entire font data.
                // For alternatives, see https://github.com/linebender/vello/pull/739#discussion_r1858293718
                let font_blob = Blob::new(Arc::new(CosmicFontBlobAdapter::new(font)));
                let vello_font = Font::new(font_blob, index);
                vello_fonts.insert(font_id, vello_font);
            }
        }

        let text = "おはよう (ja) (ohayō) 🌅✨ (morning), こんにちは (ja) (konnichi wa) ☀️😊 (daytime), こんばんは (ja) (konban wa) 🌙🌟 (evening)";

        // Text metrics indicate the font size and line height of a buffer
        const FONT_SIZE: f32 = 24.0;
        const LINE_HEIGHT: f32 = FONT_SIZE * 1.2;
        let metrics = Metrics::new(FONT_SIZE, LINE_HEIGHT);

        // A Buffer provides shaping and layout for a UTF-8 string, create one per text widget
        let buffer = Buffer::new(&mut font_system, metrics);
        let mut editor = Editor::new(buffer);

        // Set a size for the text buffer, in pixels
        let width = 200.0;

        // The height is unbounded
        editor.with_buffer_mut(|buffer| {
            buffer.set_metrics(&mut font_system, metrics);
            buffer.set_size(&mut font_system, Some(width), None);

            // Attributes indicate what font to choose
            let attrs = Attrs::new();

            // Ensure advanced shaping is enabled for complex scripts
            buffer.set_text(&mut font_system, text, attrs, Shaping::Advanced);
        });

        // Perform shaping as desired
        editor.shape_as_needed(&mut font_system, true);

        editor.set_cursor(Cursor::new(0, 36));
        editor.set_selection(Selection::Normal(Cursor::new(0, 43)));

        Self {
            font_system,
            vello_fonts,
            buffer_glyphs: create_glyphs_for_editor(
                &editor,
                Color::from_rgba8(255, 255, 255, 255),
                Color::from_rgba8(255, 0, 0, 255),
                Color::from_rgba8(0, 0, 255, 255),
                Color::from_rgba8(255, 255, 255, 255),
            ),
        }
    }
}

pub struct BufferGlyphs {
    font_size: f32,
    glyph_highlight_color: Color,
    cursor_color: Color,
    buffer_lines: Vec<BufferLine>,
}

pub struct BufferLine {
    glyph_highlights: Vec<Rect>,
    cursor: Option<Rect>,
    glyph_runs: Vec<BufferGlyphRun>,
}

struct BufferGlyphRun {
    font: ID,
    glyphs: Vec<Glyph>,
    glyph_color: Color,
}

struct EditorInfo {
    cursor_color: Color,
    selection_color: Color,
    selected_text_color: Color,
    selection_bounds: Option<(Cursor, Cursor)>,
    cursor: Cursor,
}

impl EditorInfo {
    fn new(
        editor: &Editor,
        cursor_color: Color,
        selection_color: Color,
        selected_text_color: Color,
    ) -> Self {
        Self {
            cursor_color,
            selection_color,
            selected_text_color,
            selection_bounds: editor.selection_bounds(),
            cursor: editor.cursor(),
        }
    }
}

fn create_glyphs_for_editor(
    editor: &Editor,
    text_color: Color,
    cursor_color: Color,
    selection_color: Color,
    selected_text_color: Color,
) -> BufferGlyphs {
    editor.with_buffer(|buffer| {
        create_glyphs(
            buffer,
            text_color,
            Some(EditorInfo::new(
                editor,
                cursor_color,
                selection_color,
                selected_text_color,
            )),
        )
    })
}

fn create_glyphs(
    buffer: &Buffer,
    text_color: Color,
    editor_info: Option<EditorInfo>,
) -> BufferGlyphs {
    // Get the laid out glyphs and convert them to Glyphs for vello

    let mut last_font: Option<(ID, Color)> = None;

    let mut buffer_glyphs = BufferGlyphs {
        font_size: buffer.metrics().font_size,
        glyph_highlight_color: Color::WHITE,
        cursor_color: Color::WHITE,
        buffer_lines: vec![],
    };

    if let Some(editor_info) = &editor_info {
        buffer_glyphs.cursor_color = editor_info.cursor_color;
        buffer_glyphs.glyph_highlight_color = editor_info.selection_color;
    }

    for layout_run in buffer.layout_runs() {
        let mut current_glyphs: Vec<Glyph> = vec![];
        let line_i = layout_run.line_i;
        let line_y = layout_run.line_y as f64;
        let line_top = layout_run.line_top as f64;
        let line_height = layout_run.line_height as f64;

        let mut buffer_line = BufferLine {
            glyph_highlights: vec![],
            cursor: None,
            glyph_runs: vec![],
        };

        if let Some(editor_info) = &editor_info {
            // Highlight selection
            if let Some((start, end)) = editor_info.selection_bounds {
                if line_i >= start.line && line_i <= end.line {
                    let mut range_opt = None;
                    for glyph in layout_run.glyphs.iter() {
                        // Guess x offset based on characters
                        let cluster = &layout_run.text[glyph.start..glyph.end];
                        let total = cluster.grapheme_indices(true).count();
                        let mut c_x = glyph.x;
                        let c_w = glyph.w / total as f32;
                        for (i, c) in cluster.grapheme_indices(true) {
                            let c_start = glyph.start + i;
                            let c_end = glyph.start + i + c.len();
                            if (start.line != line_i || c_end > start.index)
                                && (end.line != line_i || c_start < end.index)
                            {
                                range_opt = match range_opt.take() {
                                    Some((min, max)) => Some((
                                        cmp::min(min, c_x as i32),
                                        cmp::max(max, (c_x + c_w) as i32),
                                    )),
                                    None => Some((c_x as i32, (c_x + c_w) as i32)),
                                };
                            } else if let Some((min, max)) = range_opt.take() {
                                buffer_line.glyph_highlights.push(Rect::from_origin_size(
                                    Point::new(min as f64, line_top),
                                    Size::new(cmp::max(0, max - min) as f64, line_height),
                                ));
                            }
                            c_x += c_w;
                        }
                    }

                    if layout_run.glyphs.is_empty() && end.line > line_i {
                        // Highlight all internal empty lines
                        range_opt = Some((0, buffer.size().0.unwrap_or(0.0) as i32));
                    }

                    if let Some((mut min, mut max)) = range_opt.take() {
                        if end.line > line_i {
                            // Draw to end of line
                            if layout_run.rtl {
                                min = 0;
                            } else {
                                max = buffer.size().0.unwrap_or(0.0) as i32;
                            }
                        }
                        buffer_line.glyph_highlights.push(Rect::from_origin_size(
                            Point::new(min as f64, line_top),
                            Size::new(cmp::max(0, max - min) as f64, line_height),
                        ));
                    }
                }
            }

            // Cursor
            if let Some((x, y)) = cursor_position(&editor_info.cursor, &layout_run) {
                buffer_line.cursor = Some(Rect::from_origin_size(
                    Point::new(x as f64, y as f64),
                    Size::new(1.0, line_height),
                ));
            }
        }

        for glyph in layout_run.glyphs {
            let mut glyph_color = match glyph.color_opt {
                Some(color) => Color::from_rgba8(color.r(), color.g(), color.b(), color.a()),
                None => text_color,
            };

            if let Some(editor_info) = &editor_info {
                if text_color.components != editor_info.selected_text_color.components {
                    if let Some((start, end)) = editor_info.selection_bounds {
                        if line_i >= start.line
                            && line_i <= end.line
                            && (start.line != line_i || glyph.end > start.index)
                            && (end.line != line_i || glyph.start < end.index)
                        {
                            glyph_color = editor_info.selected_text_color;
                        }
                    }
                }
            }

            if let Some((last_font, last_glyph_color)) = last_font {
                if last_font != glyph.font_id
                    || last_glyph_color.components != glyph_color.components
                {
                    buffer_line.glyph_runs.push(BufferGlyphRun {
                        font: last_font,
                        glyphs: current_glyphs,
                        glyph_color: last_glyph_color,
                    });
                    current_glyphs = vec![];
                }
            }

            last_font = Some((glyph.font_id, glyph_color));
            current_glyphs.push(Glyph {
                x: glyph.x,
                y: glyph.y + line_y as f32,
                id: glyph.glyph_id as u32,
            });
        }
        if !current_glyphs.is_empty() {
            let (last_font, last_color) = last_font.unwrap();
            buffer_line.glyph_runs.push(BufferGlyphRun {
                font: last_font,
                glyphs: current_glyphs,
                glyph_color: last_color,
            });
        }

        buffer_glyphs.buffer_lines.push(buffer_line);
    }

    buffer_glyphs
}

// Copied directly from https://github.com/pop-os/cosmic-text/blob/58c2ccd1fb3daf0abc792f9dd52b5766b7125ccd/src/edit/editor.rs#L66
fn cursor_position(cursor: &Cursor, run: &LayoutRun) -> Option<(i32, i32)> {
    let (cursor_glyph, cursor_glyph_offset) = cursor_glyph_opt(cursor, run)?;
    let x = match run.glyphs.get(cursor_glyph) {
        Some(glyph) => {
            // Start of detected glyph
            if glyph.level.is_rtl() {
                (glyph.x + glyph.w - cursor_glyph_offset) as i32
            } else {
                (glyph.x + cursor_glyph_offset) as i32
            }
        }
        None => match run.glyphs.last() {
            Some(glyph) => {
                // End of last glyph
                if glyph.level.is_rtl() {
                    glyph.x as i32
                } else {
                    (glyph.x + glyph.w) as i32
                }
            }
            None => {
                // Start of empty line
                0
            }
        },
    };

    Some((x, run.line_top as i32))
}

// Copied directly from https://github.com/pop-os/cosmic-text/blob/58c2ccd1fb3daf0abc792f9dd52b5766b7125ccd/src/edit/editor.rs#L30
fn cursor_glyph_opt(cursor: &Cursor, run: &LayoutRun) -> Option<(usize, f32)> {
    if cursor.line == run.line_i {
        for (glyph_i, glyph) in run.glyphs.iter().enumerate() {
            if cursor.index == glyph.start {
                return Some((glyph_i, 0.0));
            } else if cursor.index > glyph.start && cursor.index < glyph.end {
                // Guess x offset based on characters
                let mut before = 0;
                let mut total = 0;

                let cluster = &run.text[glyph.start..glyph.end];
                for (i, _) in cluster.grapheme_indices(true) {
                    if glyph.start + i < cursor.index {
                        before += 1;
                    }
                    total += 1;
                }

                let offset = glyph.w * (before as f32) / (total as f32);
                return Some((glyph_i, offset));
            }
        }
        match run.glyphs.last() {
            Some(glyph) => {
                if cursor.index == glyph.end {
                    return Some((run.glyphs.len(), 0.0));
                }
            }
            None => {
                return Some((0, 0.0));
            }
        }
    }
    None
}
