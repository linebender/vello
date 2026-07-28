// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU paint packing for scheduled strip draws.

use crate::util::pack_u16_pair;
use vello_common::TextureId;
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::paint::{ImageSource, Paint};

const COLOR_SOURCE_PAYLOAD: u32 = 0;
pub(crate) const COLOR_SOURCE_LAYER: u32 = 1;

const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;
const PAINT_TYPE_LINEAR_GRADIENT: u32 = 2;
const PAINT_TYPE_RADIAL_GRADIENT: u32 = 3;
const PAINT_TYPE_SWEEP_GRADIENT: u32 = 4;
const PAINT_TYPE_BLURRED_ROUNDED_RECT: u32 = 5;

// See the layout information in `render.wgsl`.
pub(crate) const COLOR_SOURCE_SHIFT: u32 = 29;
const PAINT_TYPE_SHIFT: u32 = 26;
const PAINT_TEXTURE_INDEX_MASK: u32 = (1 << PAINT_TYPE_SHIFT) - 1;

/// Shader-ready paint metadata for a strip.
#[derive(Clone, Copy)]
pub(crate) struct PackedPaint {
    /// Value source for the strip's payload field.
    payload: PaintPayload,
    /// Packed paint kind, source, and data offset.
    pub(crate) paint: u32,
    /// External texture required by this paint, if any.
    pub(crate) external_texture_id: Option<TextureId>,
    /// Whether the paint is fully opaque.
    pub(crate) opaque: bool,
}

impl PackedPaint {
    pub(crate) fn payload_at(self, x: u16, y: u16) -> u32 {
        match self.payload {
            PaintPayload::Solid(rgba) => rgba,
            PaintPayload::Position => pack_u16_pair(x, y),
        }
    }
}

/// Source used to populate a strip's paint payload.
#[derive(Clone, Copy)]
enum PaintPayload {
    /// Premultiplied RGBA value for a solid paint.
    Solid(u32),
    /// Scene-space position used to evaluate a non-solid paint.
    Position,
}

/// Resolves recorded paints to their encoded GPU offsets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintResolver<'a> {
    /// Encoded non-solid paints indexed by [`Paint`].
    encoded: &'a [EncodedPaint],
    /// GPU data offset corresponding to each encoded paint.
    gpu_offsets: &'a [u32],
}

impl<'a> PaintResolver<'a> {
    pub(crate) fn new(encoded: &'a [EncodedPaint], gpu_offsets: &'a [u32]) -> Self {
        Self {
            encoded,
            gpu_offsets,
        }
    }

    #[inline]
    pub(crate) fn pack(self, paint: &Paint) -> PackedPaint {
        match paint {
            Paint::Solid(color) => PackedPaint {
                payload: PaintPayload::Solid(color.as_premul_rgba8().to_u32()),
                paint: (COLOR_SOURCE_PAYLOAD << COLOR_SOURCE_SHIFT)
                    | (PAINT_TYPE_SOLID << PAINT_TYPE_SHIFT),
                external_texture_id: None,
                opaque: color.is_opaque(),
            },
            Paint::Indexed(indexed_paint) => {
                let paint_id = indexed_paint.index();
                let gpu_offset = self.gpu_offsets[paint_id];
                let encoded_paint = &self.encoded[paint_id];

                let (paint_type, external_texture_id) = match encoded_paint {
                    EncodedPaint::Image(encoded_image) => match &encoded_image.source {
                        ImageSource::OpaqueId { .. } => (PAINT_TYPE_IMAGE, None),
                        _ => unimplemented!("Unsupported image source"),
                    },
                    EncodedPaint::ExternalTexture(texture) => {
                        (PAINT_TYPE_IMAGE, Some(texture.texture_id))
                    }
                    EncodedPaint::Gradient(gradient) => {
                        let paint_type = match &gradient.kind {
                            EncodedKind::Linear(_) => PAINT_TYPE_LINEAR_GRADIENT,
                            EncodedKind::Radial(_) => PAINT_TYPE_RADIAL_GRADIENT,
                            EncodedKind::Sweep(_) => PAINT_TYPE_SWEEP_GRADIENT,
                        };
                        (paint_type, None)
                    }
                    EncodedPaint::BlurredRoundedRect(_) => (PAINT_TYPE_BLURRED_ROUNDED_RECT, None),
                };

                PackedPaint {
                    payload: PaintPayload::Position,
                    paint: (COLOR_SOURCE_PAYLOAD << COLOR_SOURCE_SHIFT)
                        | (paint_type << PAINT_TYPE_SHIFT)
                        | (gpu_offset & PAINT_TEXTURE_INDEX_MASK),
                    external_texture_id,
                    opaque: !encoded_paint.may_have_transparency(),
                }
            }
        }
    }
}
