// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "f32_pipeline")]
use crate::fine::F32Kernel;
#[cfg(feature = "u8_pipeline")]
use crate::fine::U8Kernel;
use crate::fine::{
    COLOR_COMPONENTS, CompositeType, FineKernel, Numeric, NumericVec, SimdLinearKind,
    SimdRadialKind, SimdSweepKind, TILE_HEIGHT_COMPONENTS, calculate_t_vals,
};
use crate::peniko::{BlendMode, ImageQuality};
use crate::util::EncodedImageExt;
use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "u8_pipeline")]
use bytemuck::cast_slice;
use core::{iter, marker::PhantomData};
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::fearless_simd::*;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, ImageSource, Paint, PremulColor, Tint};
use vello_common::strip::Strip;
use vello_common::tile::Tile;

const DEPTH_BUCKET_WIDTH: u16 = 32;
const PIXEL_CENTER_OFFSET: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Cmd {
    Fill(FillCmd),
    AlphaFill(AlphaFillCmd),
    PushLayer(u32),
    PopLayer(u32),
}

impl Cmd {
    #[inline(always)]
    fn fill_x(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.x,
            Self::AlphaFill(cmd) => cmd.x,
            Self::PushLayer(_) | Self::PopLayer(_) => unreachable!(),
        }
    }

    #[inline(always)]
    fn fill_width(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.width,
            Self::AlphaFill(cmd) => cmd.width,
            Self::PushLayer(_) | Self::PopLayer(_) => unreachable!(),
        }
    }

    #[inline(always)]
    fn fill_attrs_idx(&self) -> u32 {
        match self {
            Self::Fill(cmd) => cmd.attrs_idx,
            Self::AlphaFill(cmd) => cmd.attrs_idx,
            Self::PushLayer(_) | Self::PopLayer(_) => unreachable!(),
        }
    }

    #[inline(always)]
    fn is_fill(&self) -> bool {
        matches!(self, Self::Fill(_) | Self::AlphaFill(_))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FillAttrs {
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
    pub(crate) path_id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerClip;

#[derive(Debug, Clone)]
pub(crate) struct LayerProps {
    pub(crate) mask: Option<Mask>,
    pub(crate) blend_mode: BlendMode,
    pub(crate) clip_path: Option<LayerClip>,
    pub(crate) opacity: f32,
}

#[derive(Debug)]
struct ActiveLayer {
    props_idx: u32,
    occupied_rows: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<Cmd>,
    pub(crate) opaque: Vec<FillCmd>,
    bounds: Option<(u16, u16)>,
    opaque_bounds: Option<(u16, u16)>,
    max_opaque_path_id: u32,
    layer_depth: usize,
}

impl RowCommands {
    fn new() -> Self {
        Self {
            cmds: Vec::new(),
            opaque: Vec::new(),
            bounds: None,
            opaque_bounds: None,
            max_opaque_path_id: 0,
            layer_depth: 0,
        }
    }

    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
        self.bounds = None;
        self.opaque_bounds = None;
        self.max_opaque_path_id = 0;
        self.layer_depth = 0;
    }

    fn push_fill_cmd(&mut self, cmd: Cmd, width: u16) {
        debug_assert!(cmd.is_fill());
        self.include_bounds(cmd.fill_x(), cmd.fill_width(), width);
        self.cmds.push(cmd);
    }

    fn push_layer(&mut self, props_idx: u32) {
        self.cmds.push(Cmd::PushLayer(props_idx));
        self.layer_depth += 1;
    }

    fn pop_layer(&mut self, props_idx: u32) {
        self.cmds.push(Cmd::PopLayer(props_idx));
        self.layer_depth -= 1;
    }

    fn push_opaque(&mut self, cmd: FillCmd, width: u16, path_id: u32) {
        self.include_bounds(cmd.x, cmd.width, width);
        self.include_opaque_bounds(cmd.x, cmd.width, width);
        self.max_opaque_path_id = self.max_opaque_path_id.max(path_id);
        self.opaque.push(cmd);
    }

    fn bounds(&self) -> Option<(u16, u16)> {
        self.bounds
    }

    fn depth_affects(&self, x: u16, cmd_width: u16, path_id: u32) -> bool {
        if path_id >= self.max_opaque_path_id {
            return false;
        }

        let Some((opaque_start, opaque_end)) = self.opaque_bounds else {
            return false;
        };

        let end = x.saturating_add(cmd_width);
        if x >= opaque_end || end <= opaque_start {
            return false;
        }

        true
    }

    fn include_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.bounds = Some(match self.bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }

    fn include_opaque_bounds(&mut self, x: u16, cmd_width: u16, width: u16) {
        let start = x.min(width);
        let end = start.saturating_add(cmd_width).min(width);
        if start >= end {
            return;
        }

        self.opaque_bounds = Some(match self.opaque_bounds {
            Some((old_start, old_end)) => (old_start.min(start), old_end.max(end)),
            None => (start, end),
        });
    }
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    width: u16,
    clip_bboxes: Vec<RectU16>,
    rows: Vec<RowCommands>,
    attrs: Vec<FillAttrs>,
    layer_props: Vec<LayerProps>,
    active_layers: Vec<ActiveLayer>,
    next_path_id: u32,
}

impl CommandBucketer {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let full_clip_bbox = Self::full_clip_bbox(width, height);
        let num_rows = usize::from(full_clip_bbox.height() / Tile::HEIGHT);
        Self {
            width,
            clip_bboxes: vec![full_clip_bbox],
            rows: (0..num_rows).map(|_| RowCommands::new()).collect(),
            attrs: Vec::new(),
            layer_props: Vec::new(),
            active_layers: Vec::new(),
            next_path_id: 1,
        }
    }

    fn full_clip_bbox(width: u16, height: u16) -> RectU16 {
        RectU16::new(
            0,
            0,
            Self::ceil_to_tile_width(width),
            Self::ceil_to_tile_height(height),
        )
    }

    fn ceil_to_tile_width(width: u16) -> u16 {
        width
            .checked_next_multiple_of(Tile::WIDTH)
            .unwrap_or(u16::MAX)
    }

    fn ceil_to_tile_height(height: u16) -> u16 {
        height
            .checked_next_multiple_of(Tile::HEIGHT)
            .unwrap_or(u16::MAX)
    }

    pub(crate) fn rows(&self) -> &[RowCommands] {
        &self.rows
    }

    pub(crate) fn width(&self) -> u16 {
        self.clip_bboxes[0].width()
    }

    pub(crate) fn attrs(&self) -> &[FillAttrs] {
        &self.attrs
    }

    pub(crate) fn layer_props(&self) -> &[LayerProps] {
        &self.layer_props
    }

    pub(crate) fn has_unpopped_layers(&self) -> bool {
        !self.active_layers.is_empty()
    }

    pub(crate) fn reset(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
        self.attrs.clear();
        self.layer_props.clear();
        self.active_layers.clear();
        self.next_path_id = 1;
        self.clip_bboxes.truncate(1);
    }

    pub(crate) fn push_layer(
        &mut self,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip_path: Option<LayerClip>,
    ) {
        let props_idx = self.layer_props.len() as u32;
        self.layer_props.push(LayerProps {
            mask,
            blend_mode,
            clip_path,
            opacity,
        });
        self.active_layers.push(ActiveLayer {
            props_idx,
            occupied_rows: Vec::new(),
        });
    }

    pub(crate) fn pop_layer(&mut self) {
        let mut layer = self.active_layers.pop().unwrap();
        for row_idx in layer.occupied_rows.drain(..) {
            let row = &mut self.rows[row_idx];
            debug_assert_eq!(row.layer_depth, self.active_layers.len() + 1);
            row.pop_layer(layer.props_idx);
        }
    }

    #[inline(always)]
    fn ensure_row_layers(&mut self, row_idx: usize) {
        let layer_depth = self.rows[row_idx].layer_depth;
        if layer_depth == self.active_layers.len() {
            return;
        }

        for layer_idx in layer_depth..self.active_layers.len() {
            let props_idx = self.active_layers[layer_idx].props_idx;
            self.rows[row_idx].push_layer(props_idx);
            self.active_layers[layer_idx].occupied_rows.push(row_idx);
        }
    }

    pub(crate) fn generate(
        &mut self,
        strip_buf: &[Strip],
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        let path_id = self.next_path_id;
        self.next_path_id = self
            .next_path_id
            .checked_add(1)
            .expect("row-bucket path ID overflow");
        let attrs_idx = self.attrs.len() as u32;
        self.attrs.push(FillAttrs {
            paint: paint.clone(),
            blend_mode,
            mask: mask.clone(),
            path_id,
        });
        let can_depth_cull =
            self.active_layers.is_empty() && blend_mode == BlendMode::default() && mask.is_none();
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let clip_x0 = clip_bbox.x0;
        let clip_x1 = clip_bbox.x1;
        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];
            let strip_y = strip.strip_y();
            let row_y = strip_y * Tile::HEIGHT;
            if row_y < clip_bbox.y0 {
                continue;
            }
            if row_y >= clip_bbox.y1 {
                break;
            }

            let row_idx = strip_y as usize;
            if row_idx >= self.rows.len() {
                break;
            }

            let next_strip = &strip_buf[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let x0 = strip.x;
            let x1 = x0.saturating_add(strip_width);
            let clipped_x0 = x0.max(clip_x0);
            let clipped_x1 = x1.min(clip_x1);

            if clipped_x0 < clipped_x1 {
                let alpha_idx =
                    strip.alpha_idx() + u32::from(clipped_x0 - x0) * u32::from(Tile::HEIGHT);
                self.ensure_row_layers(row_idx);
                self.rows[row_idx].push_fill_cmd(
                    Cmd::AlphaFill(AlphaFillCmd {
                        x: clipped_x0,
                        width: clipped_x1 - clipped_x0,
                        alpha_idx,
                        attrs_idx,
                    }),
                    self.width,
                );
            }

            if next_strip.fill_gap() && strip_y == next_strip.strip_y() {
                let fill_x0 = x1.max(clip_x0);
                let fill_x1 = next_strip.x.min(clip_x1);
                if fill_x0 < fill_x1 {
                    self.push_fill(
                        row_idx,
                        fill_x0,
                        fill_x1 - fill_x0,
                        &paint,
                        path_id,
                        attrs_idx,
                        can_depth_cull,
                        encoded_paints,
                    );
                }
            }
        }
    }

    fn push_fill(
        &mut self,
        row_idx: usize,
        x: u16,
        width: u16,
        paint: &Paint,
        path_id: u32,
        attrs_idx: u32,
        can_depth_cull: bool,
        encoded_paints: &[EncodedPaint],
    ) {
        self.ensure_row_layers(row_idx);
        let row = &mut self.rows[row_idx];
        if !can_depth_cull || !paint_is_opaque(paint, encoded_paints) {
            row.push_fill_cmd(
                Cmd::Fill(FillCmd {
                    x,
                    width,
                    attrs_idx,
                }),
                self.width,
            );
            return;
        }

        let end = x + width;
        let aligned_x = x.next_multiple_of(DEPTH_BUCKET_WIDTH).min(end);
        let aligned_end = (end / DEPTH_BUCKET_WIDTH) * DEPTH_BUCKET_WIDTH;

        if aligned_x >= aligned_end {
            row.push_fill_cmd(
                Cmd::Fill(FillCmd {
                    x,
                    width,
                    attrs_idx,
                }),
                self.width,
            );
            return;
        }

        if x < aligned_x {
            row.push_fill_cmd(
                Cmd::Fill(FillCmd {
                    x,
                    width: aligned_x - x,
                    attrs_idx,
                }),
                self.width,
            );
        }

        if aligned_x < aligned_end {
            row.push_opaque(
                FillCmd {
                    x: aligned_x,
                    width: aligned_end - aligned_x,
                    attrs_idx,
                },
                self.width,
                path_id,
            );
        }

        if aligned_end < end {
            row.push_fill_cmd(
                Cmd::Fill(FillCmd {
                    x: aligned_end,
                    width: end - aligned_end,
                    attrs_idx,
                }),
                self.width,
            );
        }
    }
}

fn paint_is_opaque(paint: &Paint, encoded_paints: &[EncodedPaint]) -> bool {
    match paint {
        Paint::Solid(color) => color.is_opaque(),
        Paint::Indexed(index) => match &encoded_paints[index.index()] {
            EncodedPaint::Gradient(gradient) => !gradient.may_have_transparency,
            EncodedPaint::Image(image) => !image.may_have_transparency,
            EncodedPaint::ExternalTexture(texture) => !texture.may_have_transparency,
            EncodedPaint::BlurredRoundedRect(_) => false,
        },
    }
}

pub(crate) trait RowRenderKernel<S: Simd>: FineKernel<S> {
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>);

    fn pack_prefix(
        simd: S,
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        out_width: usize,
        out: &mut [u8],
    );

    fn pack_tail(
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        height: usize,
        out_width: usize,
        out: &mut [u8],
    );
}

#[cfg(feature = "u8_pipeline")]
impl<S: Simd> RowRenderKernel<S> for U8Kernel {
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>) {
        let color = <Self as FineKernel<S>>::extract_color(color);
        if color[3] == <Self as FineKernel<S>>::Numeric::ONE && alphas.is_none() {
            <Self as FineKernel<S>>::copy_solid(simd, dest, color);
        } else {
            <Self as FineKernel<S>>::alpha_composite_solid(simd, dest, color, alphas);
        }
    }

    fn pack_prefix(
        simd: S,
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                pack_u8_prefix(simd, scratch, x, width, out_width, out);
            },
        );
    }

    fn pack_tail(
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        height: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        for y in 0..height {
            let row_start = (y * out_width + x) * COLOR_COMPONENTS;
            let row = &mut out[row_start..][..width * COLOR_COMPONENTS];
            for (dx, pixel) in row.chunks_exact_mut(COLOR_COMPONENTS).enumerate() {
                let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * (x + dx) + y);
                pixel.copy_from_slice(&scratch[idx..idx + COLOR_COMPONENTS]);
            }
        }
    }
}

#[cfg(feature = "f32_pipeline")]
impl<S: Simd> RowRenderKernel<S> for F32Kernel {
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>) {
        let color = <Self as FineKernel<S>>::extract_color(color);
        if color[3] == <Self as FineKernel<S>>::Numeric::ONE && alphas.is_none() {
            <Self as FineKernel<S>>::copy_solid(simd, dest, color);
        } else {
            <Self as FineKernel<S>>::alpha_composite_solid(simd, dest, color, alphas);
        }
    }

    fn pack_prefix(
        _simd: S,
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        <Self as RowRenderKernel<S>>::pack_tail(
            scratch,
            x,
            width,
            Tile::HEIGHT as usize,
            out_width,
            out,
        );
    }

    fn pack_tail(
        scratch: &[Self::Numeric],
        x: usize,
        width: usize,
        height: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        for y in 0..height {
            let row_start = (y * out_width + x) * COLOR_COMPONENTS;
            let row = &mut out[row_start..][..width * COLOR_COMPONENTS];
            for (dx, pixel) in row.chunks_exact_mut(COLOR_COMPONENTS).enumerate() {
                let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * (x + dx) + y);
                let src = &scratch[idx..idx + COLOR_COMPONENTS];
                pixel[0] = (src[0] * 255.0 + 0.5) as u8;
                pixel[1] = (src[1] * 255.0 + 0.5) as u8;
                pixel[2] = (src[2] * 255.0 + 0.5) as u8;
                pixel[3] = (src[3] * 255.0 + 0.5) as u8;
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct RowFine<S: Simd, T: RowRenderKernel<S>> {
    simd: S,
    out_width: u16,
    buffer_width: u16,
    buffers: Vec<Vec<T::Numeric>>,
    buffer_pool: Vec<Vec<T::Numeric>>,
    paint_buf: Vec<T::Numeric>,
    f32_buf: Vec<f32>,
    depth: Vec<u32>,
    _marker: PhantomData<T>,
}

impl<S: Simd, T: RowRenderKernel<S>> RowFine<S, T> {
    fn new(simd: S, out_width: u16, buffer_width: u16) -> Self {
        let scratch_len = usize::from(buffer_width) * TILE_HEIGHT_COMPONENTS;
        Self {
            simd,
            out_width,
            buffer_width,
            // TODO
            buffers: vec![vec![T::Numeric::ZERO; scratch_len]],
            buffer_pool: Vec::new(),
            paint_buf: Vec::new(),
            f32_buf: Vec::new(),
            depth: vec![0; usize::from(buffer_width.div_ceil(DEPTH_BUCKET_WIDTH))],
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn scratch(&self) -> &[T::Numeric] {
        debug_assert_eq!(self.buffers.len(), 1);
        &self.buffers[0]
    }

    #[inline(always)]
    fn scratch_mut(&mut self) -> &mut [T::Numeric] {
        self.buffers.last_mut().unwrap()
    }

    fn clear_range(&mut self, x: u16, width: u16) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let depth_start = usize::from(x / DEPTH_BUCKET_WIDTH);
        let depth_end = ((usize::from(x) + usize::from(width))
            .div_ceil(usize::from(DEPTH_BUCKET_WIDTH)))
        .min(self.depth.len());

        self.simd.vectorize(
            #[inline(always)]
            || {
                self.buffers[0][scratch_start..scratch_start + scratch_len].fill(T::Numeric::ZERO);
                self.depth[depth_start..depth_end].fill(0);
            },
        )
    }

    fn push_layer(&mut self, x: u16, width: u16) {
        let mut buf = self
            .buffer_pool
            .pop()
            .unwrap_or_else(|| vec![T::Numeric::ZERO; self.buffers[0].len()]);
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        buf[scratch_start..scratch_start + scratch_len].fill(T::Numeric::ZERO);
        self.buffers.push(buf);
    }

    fn pop_layer(&mut self, row_y: u16, x: u16, width: u16, props: &LayerProps) {
        let scratch_start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let scratch_len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let (source, rest) = self.buffers.split_last_mut().unwrap();
        let target = rest.last_mut().unwrap();
        let source = &mut source[scratch_start..scratch_start + scratch_len];
        let target = &mut target[scratch_start..scratch_start + scratch_len];

        let _ = &props.clip_path;

        if let Some(mask) = props.mask.as_ref() {
            Self::apply_mask(self.simd, source, x, row_y, width, mask);
        }

        if props.opacity != 1.0 {
            T::apply_mask(
                self.simd,
                source,
                iter::repeat(T::NumericVec::from_f32(
                    self.simd,
                    f32x16::splat(self.simd, props.opacity),
                )),
            );
        }

        if props.blend_mode == BlendMode::default() {
            T::alpha_composite_buffer(self.simd, target, source, None);
        } else {
            T::blend(
                self.simd,
                target,
                x,
                row_y,
                source
                    .chunks_exact(T::Composite::LENGTH)
                    .map(|s| T::Composite::from_slice(self.simd, s)),
                props.blend_mode,
                None,
                None,
            );
        }

        let popped = self.buffers.pop().unwrap();
        self.buffer_pool.push(popped);
    }

    fn apply_mask(simd: S, target: &mut [T::Numeric], x: u16, y: u16, width: u16, mask: &Mask) {
        let y = u32::from(y) + u32x4::from_slice(simd, &[0, 1, 2, 3]);
        let iter = (x..x.saturating_add(width)).map(|x| {
            let x_in_range = x < mask.width();

            macro_rules! sample {
                ($idx:expr) => {
                    if x_in_range && (y[$idx] as u16) < mask.height() {
                        mask.sample(x, y[$idx] as u16)
                    } else {
                        0
                    }
                };
            }

            let s1 = sample!(0);
            let s2 = sample!(1);
            let s3 = sample!(2);
            let s4 = sample!(3);

            let samples = u8x16::from_slice(
                simd,
                &[
                    s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3, s4, s4, s4, s4,
                ],
            );
            T::NumericVec::from_u8(simd, samples)
        });

        T::apply_mask(simd, target, iter);
    }

    #[inline(always)]
    fn fill_solid(&mut self, x: u16, width: u16, color: PremulColor, alphas: Option<&[u8]>) {
        if width == 0 {
            return;
        }

        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let simd = self.simd;
        let scratch = self.scratch_mut();
        T::fill_solid(simd, &mut scratch[start..start + len], color, alphas);
    }

    #[inline(always)]
    fn fill_solid_with_attrs(
        &mut self,
        x: u16,
        y: u16,
        width: u16,
        color: PremulColor,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        alphas: Option<&[u8]>,
    ) {
        if blend_mode == BlendMode::default() && mask.is_none() {
            self.fill_solid(x, width, color, alphas);
            return;
        }

        if width == 0 {
            return;
        }

        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        let color = T::extract_color(color);
        let simd = self.simd;
        let color = T::Composite::from_color(simd, color);
        let scratch = self.scratch_mut();
        T::blend(
            simd,
            &mut scratch[start..start + len],
            x,
            y,
            iter::repeat(color),
            blend_mode,
            alphas,
            mask,
        );
    }

    fn fill_indexed(
        &mut self,
        x: u16,
        y: u16,
        width: u16,
        paint_index: usize,
        blend_mode: BlendMode,
        mask: Option<&Mask>,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        alphas: Option<&[u8]>,
    ) {
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        if self.paint_buf.len() < len {
            self.paint_buf.resize(len, T::Numeric::ZERO);
        }

        let t_len = usize::from(width) * Tile::HEIGHT as usize;
        if self.f32_buf.len() < t_len {
            self.f32_buf.resize(t_len, 0.0);
        }

        let scratch = self.buffers.last_mut().unwrap();
        fill_indexed_paint::<S, T>(
            self.simd,
            scratch,
            &mut self.paint_buf,
            &mut self.f32_buf,
            x,
            y,
            width,
            paint_index,
            blend_mode,
            mask,
            encoded_paints,
            image_resolver,
            alphas,
        );
    }

    #[inline(always)]
    fn render_opaque(
        &mut self,
        cmd: FillCmd,
        row_y: u16,
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let start = usize::from(cmd.x / DEPTH_BUCKET_WIDTH);
        let end = usize::from((cmd.x + cmd.width) / DEPTH_BUCKET_WIDTH);

        if start + 1 == end {
            if self.depth[start] == 0 {
                match &attrs.paint {
                    Paint::Solid(color) => self.fill_solid(cmd.x, cmd.width, *color, None),
                    Paint::Indexed(index) => self.fill_indexed(
                        cmd.x,
                        row_y,
                        cmd.width,
                        index.index(),
                        attrs.blend_mode,
                        attrs.mask.as_ref(),
                        encoded_paints,
                        image_resolver,
                        None,
                    ),
                }
                self.depth[start] = attrs.path_id;
            }
            return;
        }

        let mut idx = start;
        while idx < end {
            while idx < end && self.depth[idx] != 0 {
                idx += 1;
            }

            let run_start = idx;
            while idx < end && self.depth[idx] == 0 {
                idx += 1;
            }

            if run_start == idx {
                continue;
            }

            let x = (run_start as u16) * DEPTH_BUCKET_WIDTH;
            let width = (idx - run_start) as u16 * DEPTH_BUCKET_WIDTH;
            match &attrs.paint {
                Paint::Solid(color) => self.fill_solid(x, width, *color, None),
                Paint::Indexed(index) => self.fill_indexed(
                    x,
                    row_y,
                    width,
                    index.index(),
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    encoded_paints,
                    image_resolver,
                    None,
                ),
            }
            for depth in &mut self.depth[run_start..idx] {
                *depth = attrs.path_id;
            }
        }
    }

    #[inline(always)]
    fn render_cmd(
        &mut self,
        cmd: Cmd,
        row_y: u16,
        alphas: &[u8],
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
        use_depth: bool,
    ) {
        let cmd_x = cmd.fill_x();
        let cmd_end = (cmd_x + cmd.fill_width()).min(self.out_width);
        if cmd_x >= cmd_end {
            return;
        }

        if !use_depth {
            self.render_cmd_span(
                cmd,
                cmd_x,
                cmd_end,
                row_y,
                alphas,
                attrs,
                encoded_paints,
                image_resolver,
            );
            return;
        }

        let start = usize::from(cmd_x / DEPTH_BUCKET_WIDTH);
        let end = usize::from(cmd_end.div_ceil(DEPTH_BUCKET_WIDTH));

        if start + 1 == end {
            if self.depth[start] <= attrs.path_id {
                self.render_cmd_span(
                    cmd,
                    cmd_x,
                    cmd_end,
                    row_y,
                    alphas,
                    attrs,
                    encoded_paints,
                    image_resolver,
                );
            }
            return;
        }

        let mut idx = start;
        while idx < end {
            while idx < end && self.depth[idx] > attrs.path_id {
                idx += 1;
            }

            let run_start = idx;
            while idx < end && self.depth[idx] <= attrs.path_id {
                idx += 1;
            }

            if run_start == idx {
                continue;
            }

            let x = cmd_x.max(run_start as u16 * DEPTH_BUCKET_WIDTH);
            let end = cmd_end.min(idx as u16 * DEPTH_BUCKET_WIDTH);
            self.render_cmd_span(
                cmd,
                x,
                end,
                row_y,
                alphas,
                attrs,
                encoded_paints,
                image_resolver,
            );
        }
    }

    #[inline(always)]
    fn render_cmd_span(
        &mut self,
        cmd: Cmd,
        x: u16,
        end: u16,
        row_y: u16,
        alphas: &[u8],
        attrs: &FillAttrs,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        match cmd {
            Cmd::Fill(_) => match &attrs.paint {
                Paint::Solid(color) => self.fill_solid_with_attrs(
                    x,
                    row_y,
                    end - x,
                    *color,
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    None,
                ),
                Paint::Indexed(index) => self.fill_indexed(
                    x,
                    row_y,
                    end - x,
                    index.index(),
                    attrs.blend_mode,
                    attrs.mask.as_ref(),
                    encoded_paints,
                    image_resolver,
                    None,
                ),
            },
            Cmd::AlphaFill(fill) => {
                let alpha_offset =
                    fill.alpha_idx as usize + usize::from(x - fill.x) * Tile::HEIGHT as usize;
                match &attrs.paint {
                    Paint::Solid(color) => self.fill_solid_with_attrs(
                        x,
                        row_y,
                        end - x,
                        *color,
                        attrs.blend_mode,
                        attrs.mask.as_ref(),
                        Some(&alphas[alpha_offset..]),
                    ),
                    Paint::Indexed(index) => {
                        self.fill_indexed(
                            x,
                            row_y,
                            end - x,
                            index.index(),
                            attrs.blend_mode,
                            attrs.mask.as_ref(),
                            encoded_paints,
                            image_resolver,
                            Some(&alphas[alpha_offset..]),
                        );
                    }
                }
            }
            Cmd::PushLayer(_) | Cmd::PopLayer(_) => unreachable!(),
        }
    }

    fn pack(&self, row_idx: usize, row_height: usize, x: u16, width: u16, buffer: &mut [u8]) {
        let out_width = usize::from(self.out_width);
        let offset = row_idx * Tile::HEIGHT as usize * out_width * COLOR_COMPONENTS;
        let len = row_height * out_width * COLOR_COMPONENTS;
        let out = &mut buffer[offset..offset + len];
        let x = usize::from(x);
        let width = usize::from(width);
        let end = x + width;

        let prefix_start = if row_height == Tile::HEIGHT as usize {
            x.next_multiple_of(Tile::WIDTH as usize).min(end)
        } else {
            end
        };

        if x < prefix_start {
            T::pack_tail(
                self.scratch(),
                x,
                prefix_start - x,
                row_height,
                out_width,
                out,
            );
        }

        let prefix_width = if row_height == Tile::HEIGHT as usize {
            (end - prefix_start) / Tile::WIDTH as usize * Tile::WIDTH as usize
        } else {
            0
        };

        if prefix_width > 0 {
            T::pack_prefix(
                self.simd,
                self.scratch(),
                prefix_start,
                prefix_width,
                out_width,
                out,
            );
        }

        let tail_start = prefix_start + prefix_width;
        if tail_start < end {
            T::pack_tail(
                self.scratch(),
                tail_start,
                end - tail_start,
                row_height,
                out_width,
                out,
            );
        }
    }
}

fn fill_indexed_paint<S: Simd, T: RowRenderKernel<S>>(
    simd: S,
    scratch: &mut [T::Numeric],
    paint_buf: &mut [T::Numeric],
    f32_buf: &mut [f32],
    x: u16,
    y: u16,
    width: u16,
    paint_index: usize,
    blend_mode: BlendMode,
    mask: Option<&Mask>,
    encoded_paints: &[EncodedPaint],
    image_resolver: &dyn ImageResolver,
    alphas: Option<&[u8]>,
) {
    if width == 0 {
        return;
    }

    let width = usize::from(width);
    let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
    let len = width * TILE_HEIGHT_COMPONENTS;
    let dest = &mut scratch[start..start + len];
    let color_buf = &mut paint_buf[..len];
    let encoded_paint = &encoded_paints[paint_index];

    let sampler_x = f64::from(x) + PIXEL_CENTER_OFFSET;
    let sampler_y = f64::from(y) + PIXEL_CENTER_OFFSET;
    let default_blend = blend_mode == BlendMode::default();

    macro_rules! fill_complex_paint {
        ($may_have_transparency:expr, $filler:expr) => {
            fill_complex_paint!($may_have_transparency, $filler, None::<&Tint>)
        };
        ($may_have_transparency:expr, $filler:expr, $tint:expr) => {
            if $may_have_transparency || alphas.is_some() || !default_blend || mask.is_some() {
                T::apply_painter(simd, color_buf, $filler);
                if let Some(tint) = $tint {
                    T::apply_tint(simd, color_buf, tint);
                }

                if default_blend && mask.is_none() {
                    T::alpha_composite_buffer(simd, dest, color_buf, alphas);
                } else {
                    T::blend(
                        simd,
                        dest,
                        x,
                        y,
                        color_buf
                            .chunks_exact(T::Composite::LENGTH)
                            .map(|s| T::Composite::from_slice(simd, s)),
                        blend_mode,
                        alphas,
                        mask,
                    );
                }
            } else {
                T::apply_painter(simd, dest, $filler);
                if let Some(tint) = $tint {
                    T::apply_tint(simd, dest, tint);
                }
            }
        };
    }

    match encoded_paint {
        EncodedPaint::BlurredRoundedRect(rect) => {
            fill_complex_paint!(
                true,
                T::blurred_rounded_rectangle_painter(simd, rect, sampler_x, sampler_y)
            );
        }
        EncodedPaint::Gradient(gradient) => {
            let t_vals = &mut f32_buf[..width * Tile::HEIGHT as usize];

            match &gradient.kind {
                EncodedKind::Linear(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdLinearKind::new(simd, *kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );
                    fill_complex_paint!(
                        gradient.may_have_transparency,
                        T::gradient_painter(simd, gradient, t_vals)
                    );
                }
                EncodedKind::Sweep(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdSweepKind::new(simd, kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );
                    fill_complex_paint!(
                        gradient.may_have_transparency,
                        T::gradient_painter(simd, gradient, t_vals)
                    );
                }
                EncodedKind::Radial(kind) => {
                    calculate_t_vals(
                        simd,
                        SimdRadialKind::new(simd, kind),
                        t_vals,
                        gradient,
                        sampler_x,
                        sampler_y,
                    );

                    if kind.has_undefined() {
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter_with_undefined(simd, gradient, t_vals)
                        );
                    } else {
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter(simd, gradient, t_vals)
                        );
                    }
                }
            }
        }
        EncodedPaint::Image(image) => {
            let pixmap = match &image.source {
                ImageSource::Pixmap(pixmap) => pixmap.clone(),
                ImageSource::OpaqueId { id, .. } => image_resolver
                    .resolve(*id)
                    .unwrap_or_else(|| panic!("Image {:?} not found in registry", id)),
            };
            let tint = image.tint.as_ref();

            match (image.has_skew(), image.nearest_neighbor()) {
                (false, false) => {
                    if image.sampler.quality == ImageQuality::Medium {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::plain_medium_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    } else {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::high_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    }
                }
                (true, false) => {
                    if image.sampler.quality == ImageQuality::Medium {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::medium_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    } else {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::high_quality_image_painter(
                                simd, image, &pixmap, sampler_x, sampler_y
                            ),
                            tint
                        );
                    }
                }
                (false, true) => {
                    fill_complex_paint!(
                        image.may_have_transparency,
                        T::plain_nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                        tint
                    );
                }
                (true, true) => {
                    fill_complex_paint!(
                        image.may_have_transparency,
                        T::nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                        tint
                    );
                }
            }
        }
        EncodedPaint::ExternalTexture(_) => {
            unimplemented!("External textures are not supported by `vello_cpu`")
        }
    }
}

#[cfg(feature = "u8_pipeline")]
#[inline(always)]
fn pack_u8_prefix<S: Simd>(
    simd: S,
    scratch: &[u8],
    x: usize,
    width: usize,
    out_width: usize,
    out: &mut [u8],
) {
    const CHUNK_LENGTH: usize = Tile::WIDTH as usize * TILE_HEIGHT_COMPONENTS;

    let row_stride = out_width * COLOR_COMPONENTS;
    let (row0, out) = out.split_at_mut(row_stride);
    let (row1, out) = out.split_at_mut(row_stride);
    let (row2, out) = out.split_at_mut(row_stride);
    let (row3, _) = out.split_at_mut(row_stride);

    let scratch_start = x * TILE_HEIGHT_COMPONENTS;
    for (idx, col) in scratch[scratch_start..scratch_start + width * TILE_HEIGHT_COMPONENTS]
        .chunks_exact(CHUNK_LENGTH)
        .enumerate()
    {
        let dest_idx = (x + idx * Tile::WIDTH as usize) * COLOR_COMPONENTS;
        let casted: &[u32; 16] = cast_slice::<u8, u32>(col).try_into().unwrap();

        let loaded = simd.load_interleaved_128_u32x16(casted).to_bytes();
        let (loaded_lo, loaded_hi) = simd.split_u8x64(loaded);
        let (loaded_1, loaded_2) = simd.split_u8x32(loaded_lo);
        let (loaded_3, loaded_4) = simd.split_u8x32(loaded_hi);
        loaded_1.store_slice(&mut row0[dest_idx..][..16]);
        loaded_2.store_slice(&mut row1[dest_idx..][..16]);
        loaded_3.store_slice(&mut row2[dest_idx..][..16]);
        loaded_4.store_slice(&mut row3[dest_idx..][..16]);
    }
}

pub(crate) fn rasterize<S: Simd, T: RowRenderKernel<S>>(
    simd: S,
    bucketer: &CommandBucketer,
    alphas: &[u8],
    buffer: &mut [u8],
    width: u16,
    height: u16,
    encoded_paints: &[EncodedPaint],
    image_resolver: &dyn ImageResolver,
) {
    let mut fine = RowFine::<S, T>::new(simd, width, bucketer.width());
    buffer.fill(0);

    for (row_idx, row) in bucketer.rows().iter().enumerate() {
        let row_y = row_idx as u16 * Tile::HEIGHT;
        if row_y >= height {
            break;
        }

        let row_height = usize::from((height - row_y).min(Tile::HEIGHT));
        let Some((row_start, row_end)) = row.bounds() else {
            continue;
        };

        fine.clear_range(row_start, row_end - row_start);

        for &cmd in row.opaque.iter().rev() {
            let attrs = &bucketer.attrs()[cmd.attrs_idx as usize];
            fine.render_opaque(cmd, row_y, attrs, encoded_paints, image_resolver);
        }
        for &cmd in &row.cmds {
            // TODO: CHeck whether having fill/alpha fill commands in
            // a separate vector leads to better performance.
            match cmd {
                Cmd::Fill(_) | Cmd::AlphaFill(_) => {
                    let attrs = &bucketer.attrs()[cmd.fill_attrs_idx() as usize];
                    let use_depth =
                        row.depth_affects(cmd.fill_x(), cmd.fill_width(), attrs.path_id);
                    fine.render_cmd(
                        cmd,
                        row_y,
                        alphas,
                        attrs,
                        encoded_paints,
                        image_resolver,
                        use_depth,
                    );
                }
                Cmd::PushLayer(props_idx) => {
                    let _ = &bucketer.layer_props()[props_idx as usize];
                    fine.push_layer(row_start, row_end - row_start);
                }
                Cmd::PopLayer(props_idx) => {
                    let props = &bucketer.layer_props()[props_idx as usize];
                    fine.pop_layer(row_y, row_start, row_end - row_start, props);
                }
            }
        }

        fine.pack(row_idx, row_height, row_start, row_end - row_start, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::{Cmd, CommandBucketer};
    use alloc::vec::Vec;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::encode::EncodeExt;
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Affine;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::{BlendMode, ColorStop, Gradient};
    use vello_common::strip::Strip;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    #[test]
    fn opaque_fill_splits_to_aligned_middle() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(3, 0, 0, false), Strip::new(100, 0, 0, true)];

        bucketer.generate(
            &strips,
            Paint::Solid(color(RED)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].x, 32);
        assert_eq!(row.opaque[0].width, 64);
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 3 && cmd.width == 29));
        assert!(matches!(row.cmds[1], Cmd::Fill(cmd) if cmd.x == 96 && cmd.width == 4));
    }

    #[test]
    fn opaque_indexed_fill_splits_to_aligned_middle() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(3, 0, 0, false), Strip::new(100, 0, 0, true)];
        let mut encoded_paints = Vec::new();
        let paint = Gradient::new_linear((0., 0.), (128., 0.))
            .with_stops([ColorStop::from((0.0, RED)), ColorStop::from((1.0, BLUE))])
            .encode_into(&mut encoded_paints, Affine::IDENTITY, None);

        bucketer.generate(&strips, paint, BlendMode::default(), None, &encoded_paints);

        let row = &bucketer.rows()[0];
        assert_eq!(row.opaque.len(), 1);
        assert_eq!(row.opaque[0].x, 32);
        assert_eq!(row.opaque[0].width, 64);
        assert_eq!(row.cmds.len(), 2);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 3 && cmd.width == 29));
        assert!(matches!(row.cmds[1], Cmd::Fill(cmd) if cmd.x == 96 && cmd.width == 4));
    }

    #[test]
    fn transparent_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(96, 0, 0, true)];

        bucketer.generate(
            &strips,
            Paint::Solid(color(AlphaColor::from_rgba8(255, 0, 0, 128))),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 0 && cmd.width == 96));
    }

    #[test]
    fn alpha_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(8, 0, 32, false)];

        bucketer.generate(
            &strips,
            Paint::Solid(color(BLUE)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(
            matches!(row.cmds[0], Cmd::AlphaFill(cmd) if cmd.x == 0 && cmd.width == 8 && cmd.alpha_idx == 0)
        );
    }

    #[test]
    fn short_unaligned_opaque_fill_stays_in_regular_commands() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(8, 0, 0, false), Strip::new(16, 0, 0, true)];

        bucketer.generate(
            &strips,
            Paint::Solid(color(RED)),
            BlendMode::default(),
            None,
            &[],
        );

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 8 && cmd.width == 8));
    }
}
