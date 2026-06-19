// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU paint packing for the hybrid renderer.

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

#[derive(Clone, Copy)]
pub(crate) struct PackedPaint {
    pub(crate) payload: u32,
    pub(crate) paint: u32,
    pub(crate) external_texture_id: Option<TextureId>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Paints<'a> {
    encoded: &'a [EncodedPaint],
    gpu_offsets: &'a [u32],
}

impl<'a> Paints<'a> {
    pub(crate) fn new(encoded: &'a [EncodedPaint], gpu_offsets: &'a [u32]) -> Self {
        Self {
            encoded,
            gpu_offsets,
        }
    }

    pub(crate) fn is_opaque(self, paint: &Paint) -> bool {
        paint.is_opaque(self.encoded)
    }

    #[inline]
    pub(crate) fn pack(self, paint: &Paint, (x, y): (u16, u16)) -> PackedPaint {
        match paint {
            Paint::Solid(color) => {
                let rgba = color.as_premul_rgba8().to_u32();
                let paint_packed = (COLOR_SOURCE_PAYLOAD << 30) | (PAINT_TYPE_SOLID << 27);

                PackedPaint {
                    payload: rgba,
                    paint: paint_packed,
                    external_texture_id: None,
                }
            }
            Paint::Indexed(indexed_paint) => {
                let paint_id = indexed_paint.index();
                let gpu_offset = self.gpu_offsets[paint_id];
                let encoded_paint = &self.encoded[paint_id];

                match encoded_paint {
                    EncodedPaint::Image(encoded_image) => match &encoded_image.source {
                        ImageSource::OpaqueId { .. } => {
                            let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                                | (PAINT_TYPE_IMAGE << 26)
                                | (gpu_offset & 0x03FF_FFFF);
                            let scene_strip_xy = ((y as u32) << 16) | (x as u32);

                            PackedPaint {
                                payload: scene_strip_xy,
                                paint: paint_packed,
                                external_texture_id: None,
                            }
                        }
                        _ => unimplemented!("Unsupported image source"),
                    },
                    EncodedPaint::ExternalTexture(texture) => {
                        let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                            | (PAINT_TYPE_IMAGE << 26)
                            | (gpu_offset & 0x03FF_FFFF);
                        let scene_strip_xy = ((y as u32) << 16) | (x as u32);

                        PackedPaint {
                            payload: scene_strip_xy,
                            paint: paint_packed,
                            external_texture_id: Some(texture.texture_id),
                        }
                    }
                    EncodedPaint::Gradient(gradient) => {
                        let gradient_paint_type = match &gradient.kind {
                            EncodedKind::Linear(_) => PAINT_TYPE_LINEAR_GRADIENT,
                            EncodedKind::Radial(_) => PAINT_TYPE_RADIAL_GRADIENT,
                            EncodedKind::Sweep(_) => PAINT_TYPE_SWEEP_GRADIENT,
                        };
                        let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                            | (gradient_paint_type << 26)
                            | (gpu_offset & 0x03FF_FFFF);
                        let scene_strip_xy = ((y as u32) << 16) | (x as u32);

                        PackedPaint {
                            payload: scene_strip_xy,
                            paint: paint_packed,
                            external_texture_id: None,
                        }
                    }
                    EncodedPaint::BlurredRoundedRect(_) => {
                        let paint_packed = (COLOR_SOURCE_PAYLOAD << 29)
                            | (PAINT_TYPE_BLURRED_ROUNDED_RECT << 26)
                            | (gpu_offset & 0x03FF_FFFF);
                        let scene_strip_xy = ((y as u32) << 16) | (x as u32);

                        PackedPaint {
                            payload: scene_strip_xy,
                            paint: paint_packed,
                            external_texture_id: None,
                        }
                    }
                }
            }
        }
    }
}
