// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text workloads that exercise different glyph cache behaviors.
//!
//! Layout happens in [`Workload::prepare`], outside the measured frame. Per-frame work is limited
//! to computing run transforms, so frame timings are dominated by Glifo and the renderer.

use glifo::Glyph;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Point, Rect, Vec2};

use crate::fonts::{FaceId, Fonts, Shaper};

const BACKGROUND: AlphaColor<Srgb> = AlphaColor::from_rgb8(0x16, 0x18, 0x1d);
const TEXT: AlphaColor<Srgb> = AlphaColor::from_rgb8(0xe6, 0xe8, 0xee);
const MUTED: AlphaColor<Srgb> = AlphaColor::from_rgb8(0x9a, 0xa3, 0xb5);
const ACCENT: AlphaColor<Srgb> = AlphaColor::from_rgb8(0x7a, 0xb8, 0xff);

const PALETTE: [AlphaColor<Srgb>; 6] = [
    TEXT,
    MUTED,
    ACCENT,
    AlphaColor::from_rgb8(0xff, 0xc8, 0x57),
    AlphaColor::from_rgb8(0x8c, 0xe9, 0x9a),
    AlphaColor::from_rgb8(0xff, 0x8a, 0x9b),
];

const PROSE: &str = "Typography is the craft of endowing human language with a durable visual \
form. A glyph cache trades memory for time: the first time a glyph is drawn at a given size and \
subpixel offset, its outline is rasterized into an atlas page, and every later draw becomes a \
cheap image copy. The quick brown fox jumps over the lazy dog, while sphinx of black quartz \
judges my vow. Pack my box with five dozen liquor jugs! How vexingly quick daft zebras jump; \
0123456789 (){}[] <=> +-*/ #@&% \"quoted\" and 'single' text, plus naive cafe facade resume.";

const CODE: &str = "fn rasterize(glyph: &Glyph, atlas: &mut Atlas) -> Option<Slot> { let key = \
CacheKey::new(glyph.id, glyph.size, glyph.subpixel_x()); if let Some(slot) = atlas.get(&key) { \
return Some(slot); } let bounds = glyph.bounds().inflate(PADDING); let slot = \
atlas.allocate(bounds.width(), bounds.height())?; glyph.render_into(atlas.page(slot.page), \
slot.origin()); atlas.insert(key, slot); Some(slot) }";

/// Workload names and short descriptions, in display order.
pub(crate) const WORKLOADS: &[(&str, &str)] = &[
    (
        "paragraph",
        "Static wrapped prose at body and heading sizes. Steady state should be all cache hits.",
    ),
    (
        "scroll",
        "A long document scrolling at fractional speeds with horizontal drift, exercising subpixel buckets.",
    ),
    (
        "sizes",
        "One line per size from 6px to 96px. Large sizes stress atlas area and allocation.",
    ),
    (
        "zoom",
        "Continuously animated zoom. Every frame hits new effective sizes, forcing misses and evictions.",
    ),
    (
        "storm",
        "Tens of thousands of glyphs in drifting runs with mixed sizes and colors. Throughput test.",
    ),
    (
        "charset",
        "Every mapped glyph of Roboto and Inconsolata at three sizes. Fills the atlas with unique glyphs.",
    ),
    (
        "emoji",
        "COLR and bitmap (CBDT) emoji at several sizes, exercising color atlas entries and uploads.",
    ),
    (
        "blank",
        "Background only. Baseline for renderer overhead without any text.",
    ),
];

/// Parameters that change the prepared runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayoutParams {
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) font_scale: f32,
    pub(crate) density: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Paragraph,
    Scroll,
    Sizes,
    Zoom,
    Storm,
    Charset,
    Emoji,
    Blank,
}

impl Kind {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "paragraph" => Self::Paragraph,
            "scroll" => Self::Scroll,
            "sizes" => Self::Sizes,
            "zoom" => Self::Zoom,
            "storm" => Self::Storm,
            "charset" => Self::Charset,
            "emoji" => Self::Emoji,
            "blank" => Self::Blank,
            _ => return None,
        })
    }
}

/// Glyphs sharing a face, size, and color, positioned relative to `origin`.
pub(crate) struct Run {
    pub(crate) face: FaceId,
    pub(crate) size: f32,
    pub(crate) color: AlphaColor<Srgb>,
    pub(crate) origin: Vec2,
    pub(crate) glyphs: Vec<Glyph>,
    /// Vertical extent relative to `origin`, used for viewport culling in scrolling workloads.
    extent: (f32, f32),
    /// Per-run animation phase.
    phase: f32,
}

/// Receives the drawing commands of a frame. Implemented by each renderer backend.
pub(crate) trait TextSink {
    fn set_transform(&mut self, transform: Affine);
    fn set_color(&mut self, color: AlphaColor<Srgb>);
    fn fill_rect(&mut self, rect: &Rect);
    fn fill_glyphs(&mut self, fonts: &Fonts, run: &Run, hint: bool, atlas_cache: bool);
}

pub(crate) struct Workload {
    kind: Kind,
    prepared: Option<LayoutParams>,
    runs: Vec<Run>,
    /// Total height of the content, for scrolling workloads.
    content_height: f32,
}

impl Workload {
    pub(crate) fn new(name: &str) -> Option<Self> {
        Some(Self {
            kind: Kind::from_name(name)?,
            prepared: None,
            runs: Vec::new(),
            content_height: 0.0,
        })
    }

    pub(crate) fn invalidate(&mut self) {
        self.prepared = None;
    }

    /// Lays out the workload for `params` unless it is already prepared for them.
    pub(crate) fn prepare(&mut self, fonts: &Fonts, params: LayoutParams) {
        if self.prepared == Some(params) {
            return;
        }
        self.runs.clear();
        self.content_height = 0.0;
        let mut builder = Builder {
            fonts,
            params,
            runs: &mut self.runs,
        };
        self.content_height = match self.kind {
            Kind::Paragraph | Kind::Zoom => builder.document(params.height, 64),
            Kind::Scroll => builder.document(f32::INFINITY, 6),
            Kind::Sizes => builder.sizes(),
            Kind::Storm => builder.storm(),
            Kind::Charset => builder.charset(),
            Kind::Emoji => builder.emoji(),
            Kind::Blank => params.height,
        };
        self.prepared = Some(params);
    }

    /// Draws one frame at animation time `time_s` and returns the number of glyphs submitted.
    pub(crate) fn draw(
        &self,
        sink: &mut impl TextSink,
        fonts: &Fonts,
        view: Affine,
        time_s: f64,
        hint: bool,
        atlas_cache: bool,
    ) -> usize {
        let Some(params) = self.prepared else {
            return 0;
        };
        sink.set_transform(Affine::IDENTITY);
        sink.set_color(BACKGROUND);
        sink.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(params.width),
            f64::from(params.height),
        ));

        let scene = match self.kind {
            Kind::Scroll => {
                let speed = 37.3 * f64::from(params.font_scale);
                let range = f64::from((self.content_height - params.height).max(1.0));
                let y = (time_s * speed) % range;
                let x = 6.0 * (time_s * 0.7).sin();
                Affine::translate((x, -y))
            }
            Kind::Zoom => {
                let center = Vec2::new(f64::from(params.width), f64::from(params.height)) * 0.5;
                let scale = 1.0 + 0.6 * (time_s * 0.8).sin();
                Affine::translate(center) * Affine::scale(scale) * Affine::translate(-center)
            }
            _ => Affine::IDENTITY,
        };
        let transform = view * scene;
        let viewport = (-64.0, f64::from(params.height) + 64.0);

        let mut glyphs = 0;
        for run in &self.runs {
            let mut origin = run.origin;
            if self.kind == Kind::Storm {
                let t = time_s + f64::from(run.phase);
                origin += Vec2::new(23.0 * t.sin(), 11.0 * (t * 1.3).cos());
            }
            let run_transform = transform * Affine::translate(origin);
            if self.kind == Kind::Scroll {
                let top = (run_transform * Point::new(0.0, f64::from(run.extent.0))).y;
                let bottom = (run_transform * Point::new(0.0, f64::from(run.extent.1))).y;
                if bottom < viewport.0 || top > viewport.1 {
                    continue;
                }
            }
            sink.set_transform(run_transform);
            sink.set_color(run.color);
            sink.fill_glyphs(fonts, run, hint, atlas_cache);
            glyphs += run.glyphs.len();
        }
        glyphs
    }
}

struct Builder<'a> {
    fonts: &'a Fonts,
    params: LayoutParams,
    runs: &'a mut Vec<Run>,
}

impl Builder<'_> {
    fn push(&mut self, face: FaceId, size: f32, color: AlphaColor<Srgb>, glyphs: Vec<Glyph>) {
        self.push_at(face, size, color, Vec2::ZERO, glyphs, 0.0);
    }

    fn push_at(
        &mut self,
        face: FaceId,
        size: f32,
        color: AlphaColor<Srgb>,
        origin: Vec2,
        glyphs: Vec<Glyph>,
        phase: f32,
    ) {
        if glyphs.is_empty() {
            return;
        }
        let baseline = glyphs[0].y;
        self.runs.push(Run {
            face,
            size,
            color,
            origin,
            glyphs,
            extent: (baseline - size, baseline + size * 0.3),
            phase,
        });
    }

    fn margin(&self) -> f32 {
        24.0 * self.params.font_scale
    }

    /// Stacks wrapped sections of prose and code until `max_height` or `repeat` sections.
    ///
    /// Returns the content height.
    fn document(&mut self, max_height: f32, repeat: usize) -> f32 {
        let scale = self.params.font_scale;
        let margin = self.margin();
        let width = (self.params.width - 2.0 * margin).max(100.0);
        let mut y = margin;
        let sections = (repeat as f32 * self.params.density).ceil().max(1.0) as usize;
        for section in 0..sections {
            let blocks: [(FaceId, f32, AlphaColor<Srgb>, &str); 4] = [
                (FaceId::Roboto, 28.0, TEXT, "Glyph caching in Glifo"),
                (FaceId::Roboto, 16.0, TEXT, PROSE),
                (FaceId::Inconsolata, 14.0, ACCENT, CODE),
                (FaceId::Roboto, 12.0, MUTED, PROSE),
            ];
            for (face, size, color, text) in blocks {
                let size = size * scale;
                let line_height = size * 1.4;
                let shaper = Shaper::new(self.fonts.face(face), size);
                for (glyphs, baseline) in shaper.wrap(text, margin, y + size, width, line_height) {
                    if baseline > max_height {
                        return y;
                    }
                    self.push(face, size, color, glyphs);
                    y = baseline;
                }
                y += line_height * 0.6;
            }
            if section % 2 == 1 {
                y += 16.0 * scale;
            }
        }
        y + margin
    }

    fn sizes(&mut self) -> f32 {
        const SIZES: [f32; 23] = [
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0,
            36.0, 42.0, 48.0, 56.0, 64.0, 72.0, 84.0, 96.0,
        ];
        let margin = self.margin();
        let width = self.params.width - 2.0 * margin;
        let mut y = margin;
        for (index, size) in SIZES.iter().enumerate() {
            let size = size * self.params.font_scale;
            let face = if index % 3 == 2 {
                FaceId::Inconsolata
            } else {
                FaceId::Roboto
            };
            let shaper = Shaper::new(self.fonts.face(face), size);
            let text = format!("{size:.0}px {PROSE}");
            if let Some((glyphs, baseline)) = shaper
                .wrap(&text, margin, y + size, width, size)
                .into_iter()
                .next()
            {
                self.push(face, size, PALETTE[index % PALETTE.len()], glyphs);
                y = baseline + size * 0.35;
            }
        }
        y + margin
    }

    fn storm(&mut self) -> f32 {
        const SIZES: [f32; 6] = [10.0, 12.0, 13.0, 14.0, 16.0, 20.0];
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let words: Vec<&str> = PROSE.split_whitespace().collect();
        let target = (20_000.0 * self.params.density) as usize;
        let mut total = 0;
        while total < target {
            let face = if rng.below(4) == 0 {
                FaceId::Inconsolata
            } else {
                FaceId::Roboto
            };
            let size = SIZES[rng.below(SIZES.len())] * self.params.font_scale;
            let shaper = Shaper::new(self.fonts.face(face), size);
            let start = rng.below(words.len());
            let text = words
                .iter()
                .cycle()
                .skip(start)
                .take(3 + rng.below(4))
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            let Some((glyphs, _)) = shaper
                .wrap(&text, 0.0, 0.0, f32::INFINITY, size)
                .into_iter()
                .next()
            else {
                continue;
            };
            let origin = Vec2::new(
                rng.unit() * f64::from(self.params.width) - 40.0,
                rng.unit() * f64::from(self.params.height),
            );
            total += glyphs.len();
            let color = PALETTE[rng.below(PALETTE.len())];
            self.push_at(face, size, color, origin, glyphs, rng.unit() as f32 * 10.0);
        }
        self.params.height
    }

    fn charset(&mut self) -> f32 {
        let margin = self.margin();
        let right = self.params.width - margin;
        let mut y = margin;
        for face in [FaceId::Roboto, FaceId::Inconsolata] {
            let ids = self.fonts.face(face).mapped_glyph_ids();
            for size in [12.0, 18.0, 28.0] {
                let size = size * self.params.font_scale;
                let shaper = Shaper::new(self.fonts.face(face), size);
                let line_height = size * 1.3;
                y += line_height;
                let mut x = margin;
                let mut line = Vec::new();
                for &id in &ids {
                    let advance = shaper.advance(id).max(size * 0.3);
                    if x + advance > right {
                        self.push(face, size, TEXT, std::mem::take(&mut line));
                        x = margin;
                        y += line_height;
                    }
                    line.push(Glyph { id, x, y });
                    x += advance;
                }
                self.push(face, size, TEXT, line);
                y += line_height * 0.5;
            }
        }
        y + margin
    }

    fn emoji(&mut self) -> f32 {
        let margin = self.margin();
        let right = self.params.width - margin;
        let mut y = margin;
        for face in [FaceId::NotoColr, FaceId::NotoBitmap] {
            let ids = self.fonts.face(face).mapped_glyph_ids();
            let label = Shaper::new(self.fonts.face(FaceId::Roboto), 14.0);
            y += 20.0;
            if let Some((glyphs, _)) = label
                .wrap(self.fonts.face(face).name, margin, y, f32::INFINITY, 14.0)
                .into_iter()
                .next()
            {
                self.push(FaceId::Roboto, 14.0, MUTED, glyphs);
            }
            for size in [16.0, 24.0, 32.0, 48.0] {
                let size = size * self.params.font_scale;
                let step = size * 1.2;
                y += step;
                let mut x = margin;
                let mut line = Vec::new();
                for &id in &ids {
                    if x + step > right {
                        self.push(face, size, TEXT, std::mem::take(&mut line));
                        x = margin;
                        y += step;
                    }
                    line.push(Glyph { id, x, y });
                    x += step;
                }
                self.push(face, size, TEXT, line);
            }
            y += 16.0;
        }
        y + margin
    }
}

/// Small deterministic PRNG so workloads are identical across renderers and runs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
