// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Based on https://github.com/googlefonts/fontations/blob/cbdf8b485e955e3acee40df1344e33908805ed31/skrifa/src/bitmap.rs
#![allow(warnings)]

//! Bitmap strikes and glyphs.
use skrifa::{
    FontRef, MetadataProvider,
    instance::{LocationRef, Size},
    metrics::GlyphMetrics,
    raw::{
        FontData, TableProvider,
        tables::{bitmap, cbdt, cblc, ebdt, eblc, sbix},
        types::{GlyphId, Tag},
    },
    string::StringId,
};

/// Set of strikes, each containing embedded bitmaps of a single size.
#[derive(Clone)]
pub struct BitmapStrikes<'a>(StrikesKind<'a>);

impl<'a> BitmapStrikes<'a> {
    /// Creates a new `BitmapStrikes` for the given font.
    ///
    /// This will prefer `sbix`, `CBDT`, and `CBLC` formats in that order.
    ///
    /// To select a specific format, use [`with_format`](Self::with_format).
    pub fn new(font: &(impl TableProvider<'a> + MetadataProvider<'a>)) -> Self {
        for format in [BitmapFormat::Sbix, BitmapFormat::Cbdt, BitmapFormat::Ebdt] {
            if let Some(strikes) = Self::with_format(font, format) {
                return strikes;
            }
        }
        Self(StrikesKind::None)
    }

    /// Creates a new `BitmapStrikes` for the given font and format.
    ///
    /// Returns `None` if the requested format is not available.
    pub fn with_format(
        font: &(impl TableProvider<'a> + MetadataProvider<'a>),
        format: BitmapFormat,
    ) -> Option<Self> {
        let kind = match format {
            BitmapFormat::Sbix => StrikesKind::Sbix(
                font.sbix().ok()?,
                SbixKind::from_font(font),
                font.glyph_metrics(Size::unscaled(), LocationRef::default()),
            ),
            BitmapFormat::Cbdt => {
                StrikesKind::Cbdt(CbdtTables::new(font.cblc().ok()?, font.cbdt().ok()?))
            }
            BitmapFormat::Ebdt => {
                StrikesKind::Ebdt(EbdtTables::new(font.eblc().ok()?, font.ebdt().ok()?))
            }
        };
        Some(Self(kind))
    }

    /// Returns the format representing the underlying table for this set of
    /// strikes.
    pub fn format(&self) -> Option<BitmapFormat> {
        match &self.0 {
            StrikesKind::None => None,
            StrikesKind::Sbix(..) => Some(BitmapFormat::Sbix),
            StrikesKind::Cbdt(..) => Some(BitmapFormat::Cbdt),
            StrikesKind::Ebdt(..) => Some(BitmapFormat::Ebdt),
        }
    }

    /// Returns the number of available strikes.
    pub fn len(&self) -> usize {
        match &self.0 {
            StrikesKind::None => 0,
            StrikesKind::Sbix(sbix, ..) => sbix.strikes().len(),
            StrikesKind::Cbdt(cbdt) => cbdt.location.bitmap_sizes().len(),
            StrikesKind::Ebdt(ebdt) => ebdt.location.bitmap_sizes().len(),
        }
    }

    /// Returns `true` if there are no available strikes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the strike at the given index.
    pub fn get(&self, index: usize) -> Option<BitmapStrike<'a>> {
        let kind = match &self.0 {
            StrikesKind::None => return None,
            StrikesKind::Sbix(sbix, kind, metrics) => {
                StrikeKind::Sbix(sbix.strikes().get(index).ok()?, *kind, metrics.clone())
            }
            StrikesKind::Cbdt(tables) => StrikeKind::Cbdt(
                tables.location.bitmap_sizes().get(index).copied()?,
                tables.clone(),
            ),
            StrikesKind::Ebdt(tables) => StrikeKind::Ebdt(
                tables.location.bitmap_sizes().get(index).copied()?,
                tables.clone(),
            ),
        };
        Some(BitmapStrike(kind))
    }

    /// Returns the best matching glyph for the given size and glyph
    /// identifier.
    ///
    /// In this case, "best" means a glyph of the exact size, nearest larger
    /// size, or nearest smaller size, in that order.
    pub fn glyph_for_size(&self, size: Size, glyph_id: GlyphId) -> Option<BitmapGlyph<'a>> {
        // Return the largest size for an unscaled request
        let size = size.ppem().unwrap_or(f32::MAX);
        self.iter()
            .fold(None, |best: Option<BitmapGlyph<'a>>, entry| {
                let entry_size = entry.ppem();
                if let Some(best) = best {
                    let best_size = best.ppem_y;
                    if (entry_size >= size && entry_size < best_size)
                        || (best_size < size && entry_size > best_size)
                    {
                        entry.get(glyph_id).or(Some(best))
                    } else {
                        Some(best)
                    }
                } else {
                    entry.get(glyph_id)
                }
            })
    }

    /// Returns an iterator over all available strikes.
    pub fn iter(&self) -> impl Iterator<Item = BitmapStrike<'a>> + Clone + use<'a> {
        let this = self.clone();
        (0..this.len()).filter_map(move |ix| this.get(ix))
    }
}

#[derive(Clone)]
enum StrikesKind<'a> {
    None,
    Sbix(sbix::Sbix<'a>, SbixKind, GlyphMetrics<'a>),
    Cbdt(CbdtTables<'a>),
    Ebdt(EbdtTables<'a>),
}

/// Used to detect the Apple Color Emoji sbix font in order to apply a
/// workaround for CoreText's special cased vertical offset.
#[derive(Copy, Clone, PartialEq)]
enum SbixKind {
    Apple,
    Other,
}

impl SbixKind {
    fn from_font<'a>(font: &impl skrifa::MetadataProvider<'a>) -> Self {
        if font
            .localized_strings(skrifa::string::StringId::POSTSCRIPT_NAME)
            .next()
            .map(|s| s.chars().eq("AppleColorEmoji".chars()))
            .unwrap_or_default()
        {
            Self::Apple
        } else {
            Self::Other
        }
    }
}

/// Set of embedded bitmap glyphs of a specific size.
#[derive(Clone)]
pub struct BitmapStrike<'a>(StrikeKind<'a>);

impl<'a> BitmapStrike<'a> {
    /// Returns the pixels-per-em (size) of this strike.
    pub fn ppem(&self) -> f32 {
        match &self.0 {
            StrikeKind::Sbix(sbix, ..) => sbix.ppem() as f32,
            StrikeKind::Cbdt(size, _) => size.ppem_y() as f32,
            StrikeKind::Ebdt(size, _) => size.ppem_y() as f32,
        }
    }

    /// Returns a bitmap glyph for the given identifier, if available.
    pub fn get(&self, glyph_id: GlyphId) -> Option<BitmapGlyph<'a>> {
        match &self.0 {
            StrikeKind::Sbix(sbix, kind, metrics) => {
                let glyph = sbix.glyph_data(glyph_id).ok()??;
                if glyph.graphic_type() != Tag::new(b"png ") {
                    return None;
                }
                let glyf_bb = metrics.bounds(glyph_id).unwrap_or_default();
                let lsb = metrics.left_side_bearing(glyph_id).unwrap_or_default();
                let ppem = sbix.ppem() as f32;
                let png_data = glyph.data();
                // PNG format:
                // 8 byte header, IHDR chunk (4 byte length, 4 byte chunk type), width, height
                let reader = FontData::new(png_data);
                let width = reader.read_at::<u32>(16).ok()?;
                let height = reader.read_at::<u32>(20).ok()?;
                // CoreText appears to special case Apple Color Emoji, adding
                // a 100 font unit vertical offset. We do the same but only
                // when both vertical offsets are 0 to avoid incorrect
                // rendering if Apple ever does encode the offset directly in
                // the font.
                let bearing_y = if glyf_bb.y_min == 0.0
                    && glyph.origin_offset_y() == 0
                    && *kind == SbixKind::Apple
                {
                    100.0
                } else {
                    glyf_bb.y_min
                };
                Some(BitmapGlyph {
                    data: BitmapData::Png(glyph.data()),
                    bearing_x: lsb,
                    bearing_y,
                    inner_bearing_x: glyph.origin_offset_x() as f32,
                    inner_bearing_y: glyph.origin_offset_y() as f32,
                    ppem_x: ppem,
                    ppem_y: ppem,
                    width,
                    height,
                    advance: metrics.advance_width(glyph_id).unwrap_or_default(),
                    placement_origin: Origin::BottomLeft,
                })
            }
            StrikeKind::Cbdt(size, tables) => {
                let location = size
                    .location(tables.location.offset_data(), glyph_id)
                    .ok()?;
                let data = tables.data.data(&location).ok()?;
                BitmapGlyph::from_bdt(&size, &data)
            }
            StrikeKind::Ebdt(size, tables) => {
                let location = size
                    .location(tables.location.offset_data(), glyph_id)
                    .ok()?;
                let data = tables.data.data(&location).ok()?;
                BitmapGlyph::from_bdt(&size, &data)
            }
        }
    }
}

#[derive(Clone)]
enum StrikeKind<'a> {
    Sbix(sbix::Strike<'a>, SbixKind, GlyphMetrics<'a>),
    Cbdt(bitmap::BitmapSize, CbdtTables<'a>),
    Ebdt(bitmap::BitmapSize, EbdtTables<'a>),
}

#[derive(Clone)]
struct BdtTables<L, D> {
    location: L,
    data: D,
}

impl<L, D> BdtTables<L, D> {
    fn new(location: L, data: D) -> Self {
        Self { location, data }
    }
}

type CbdtTables<'a> = BdtTables<cblc::Cblc<'a>, cbdt::Cbdt<'a>>;
type EbdtTables<'a> = BdtTables<eblc::Eblc<'a>, ebdt::Ebdt<'a>>;

/// An embedded bitmap glyph.
#[derive(Clone)]
pub struct BitmapGlyph<'a> {
    pub data: BitmapData<'a>,
    pub bearing_x: f32,
    pub bearing_y: f32,
    pub inner_bearing_x: f32,
    pub inner_bearing_y: f32,
    pub ppem_x: f32,
    pub ppem_y: f32,
    pub advance: f32,
    pub width: u32,
    pub height: u32,
    pub placement_origin: Origin,
}

impl<'a> BitmapGlyph<'a> {
    fn from_bdt(
        bitmap_size: &bitmap::BitmapSize,
        bitmap_data: &bitmap::BitmapData<'a>,
    ) -> Option<Self> {
        let metrics = BdtMetrics::new(&bitmap_data);
        let (ppem_x, ppem_y) = (bitmap_size.ppem_x() as f32, bitmap_size.ppem_y() as f32);
        let bpp = bitmap_size.bit_depth();
        let data = match bpp {
            32 => {
                match &bitmap_data.content {
                    bitmap::BitmapContent::Data(bitmap::BitmapDataFormat::Png, bytes) => {
                        BitmapData::Png(bytes)
                    }
                    // 32-bit formats are always byte aligned
                    bitmap::BitmapContent::Data(bitmap::BitmapDataFormat::ByteAligned, bytes) => {
                        BitmapData::Bgra(bytes)
                    }
                    _ => return None,
                }
            }
            1 | 2 | 4 | 8 => {
                let (data, is_packed) = match &bitmap_data.content {
                    bitmap::BitmapContent::Data(bitmap::BitmapDataFormat::ByteAligned, bytes) => {
                        (bytes, false)
                    }
                    bitmap::BitmapContent::Data(bitmap::BitmapDataFormat::BitAligned, bytes) => {
                        (bytes, true)
                    }
                    _ => return None,
                };
                BitmapData::Mask(MaskData {
                    bpp,
                    is_packed,
                    data,
                })
            }
            // All other bit depth values are invalid
            _ => return None,
        };
        Some(Self {
            data,
            bearing_x: 0.0,
            bearing_y: 0.0,
            inner_bearing_x: metrics.inner_bearing_x,
            inner_bearing_y: metrics.inner_bearing_y,
            ppem_x,
            ppem_y,
            width: metrics.width,
            height: metrics.height,
            advance: metrics.advance,
            placement_origin: Origin::TopLeft,
        })
    }
}

struct BdtMetrics {
    inner_bearing_x: f32,
    inner_bearing_y: f32,
    advance: f32,
    width: u32,
    height: u32,
}

impl BdtMetrics {
    fn new(data: &bitmap::BitmapData) -> Self {
        match data.metrics {
            bitmap::BitmapMetrics::Small(metrics) => Self {
                inner_bearing_x: metrics.bearing_x() as f32,
                inner_bearing_y: metrics.bearing_y() as f32,
                advance: metrics.advance() as f32,
                width: metrics.width() as u32,
                height: metrics.height() as u32,
            },
            bitmap::BitmapMetrics::Big(metrics) => Self {
                inner_bearing_x: metrics.hori_bearing_x() as f32,
                inner_bearing_y: metrics.hori_bearing_y() as f32,
                advance: metrics.hori_advance() as f32,
                width: metrics.width() as u32,
                height: metrics.height() as u32,
            },
        }
    }
}

/// Determines the origin point for drawing a bitmap glyph.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Origin {
    TopLeft,
    BottomLeft,
}

/// Data content of a bitmap.
#[derive(Clone)]
pub enum BitmapData<'a> {
    /// Uncompressed 32-bit color bitmap data, pre-multiplied in BGRA order
    /// and encoded in the sRGB color space.
    Bgra(&'a [u8]),
    /// Compressed PNG bitmap data.
    Png(&'a [u8]),
    /// Data representing a single channel alpha mask.
    Mask(MaskData<'a>),
}

/// A single channel alpha mask.
#[derive(Clone)]
pub struct MaskData<'a> {
    /// Number of bits-per-pixel. Always 1, 2, 4 or 8.
    pub bpp: u8,
    /// True if each row of the data is bit-aligned. Otherwise, each row
    /// is padded to the next byte.
    pub is_packed: bool,
    /// Raw bitmap data.
    pub data: &'a [u8],
}

/// The format (or table) containing the data backing a set of bitmap strikes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BitmapFormat {
    Sbix,
    Cbdt,
    Ebdt,
}
