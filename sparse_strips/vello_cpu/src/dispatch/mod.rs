// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "multithreading")]
pub(crate) mod multi_threaded;
pub(crate) mod single_threaded;

use crate::RasterizerSettings;
use crate::coarse::{CommandBucketer, LayerClip};
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::Range;
use vello_common::encode::EncodedPaint;
use vello_common::filter_effects::Filter;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;
use vello_common::strip::Strip;

#[derive(Debug)]
pub(crate) enum RecordedCmd {
    Fill {
        thread_idx: u8,
        strip_range: Range<usize>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    PushLayer {
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
        content_bbox: RectU16,
    },
    CompositeFilterLayer {
        layer_id: usize,
        bbox: RectU16,
        src_x: u16,
        src_y: u16,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
    },
    PopLayer,
}

pub(crate) fn replay_recorded_commands(
    cmds: &[RecordedCmd],
    strips: &[Strip],
    bucketer: &mut CommandBucketer,
    encoded_paints: &[EncodedPaint],
    origin: (u16, u16),
) {
    let translated_strips = if origin == (0, 0) {
        Vec::new()
    } else {
        strips
            .iter()
            .map(|strip| translate_strip(*strip, origin))
            .collect::<Vec<_>>()
    };
    let strips = if origin == (0, 0) {
        strips
    } else {
        translated_strips.as_slice()
    };

    for cmd in cmds {
        match cmd {
            RecordedCmd::Fill {
                thread_idx,
                strip_range,
                paint,
                blend_mode,
                mask,
            } => {
                bucketer.generate_fill(
                    &strips[strip_range.clone()],
                    paint.clone(),
                    *blend_mode,
                    mask.clone(),
                    *thread_idx,
                    origin,
                    encoded_paints,
                );
            }
            RecordedCmd::PushLayer {
                blend_mode,
                opacity,
                mask,
                clip,
                ..
            } => bucketer.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone()),
            RecordedCmd::CompositeFilterLayer {
                layer_id,
                bbox,
                src_x,
                src_y,
                blend_mode,
                opacity,
                mask,
                clip,
            } => {
                let bbox = translate_bbox(*bbox, origin);
                bucketer.push_layer(*blend_mode, *opacity, mask.clone(), clip.clone());
                bucketer.generate_filter_layer(*layer_id, bbox, (*src_x, *src_y));
                bucketer.pop_layer(strips);
            }
            RecordedCmd::PopLayer => bucketer.pop_layer(strips),
        }
    }
}

fn translate_strip(strip: Strip, origin: (u16, u16)) -> Strip {
    let x = if strip.is_sentinel() {
        strip.x
    } else {
        strip.x.saturating_sub(origin.0)
    };
    Strip::new(
        x,
        strip.y.saturating_sub(origin.1),
        strip.alpha_idx(),
        strip.fill_gap(),
    )
}

fn translate_bbox(bbox: RectU16, origin: (u16, u16)) -> RectU16 {
    if bbox.is_empty() {
        return bbox;
    }
    RectU16::new(
        bbox.x0.saturating_sub(origin.0),
        bbox.y0.saturating_sub(origin.1),
        bbox.x1.saturating_sub(origin.0),
        bbox.y1.saturating_sub(origin.1),
    )
}

pub(crate) trait Dispatcher: Debug + Send {
    fn has_layers(&self) -> bool;
    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    /// Fill a pixel-aligned rectangle with the current paint.
    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    );
    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    );
    fn pop_clip_path(&mut self);
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    );
    fn pop_layer(&mut self);
    fn reset(&mut self);
    fn flush(&mut self);
    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    );
}
