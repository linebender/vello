// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "f32_pipeline")]
use crate::fine::F32Kernel;
#[cfg(feature = "u8_pipeline")]
use crate::fine::U8Kernel;
use crate::fine::{
    COLOR_COMPONENTS, CompositeType, FineKernel, Numeric, SimdLinearKind, SimdRadialKind,
    SimdSweepKind, TILE_HEIGHT_COMPONENTS, calculate_t_vals,
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
}

impl Cmd {
    #[inline(always)]
    fn x(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.x,
            Self::AlphaFill(cmd) => cmd.x,
        }
    }

    #[inline(always)]
    fn width(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.width,
            Self::AlphaFill(cmd) => cmd.width,
        }
    }

    #[inline(always)]
    fn attrs_idx(&self) -> u32 {
        match self {
            Self::Fill(cmd) => cmd.attrs_idx,
            Self::AlphaFill(cmd) => cmd.attrs_idx,
        }
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

#[derive(Debug, Default)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<Cmd>,
    pub(crate) opaque: Vec<FillCmd>,
    bounds: Option<(u16, u16)>,
}

impl RowCommands {
    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
        self.bounds = None;
    }

    fn push_cmd(&mut self, cmd: Cmd, width: u16) {
        self.include_bounds(cmd.x(), cmd.width(), width);
        self.cmds.push(cmd);
    }

    fn push_opaque(&mut self, cmd: FillCmd, width: u16) {
        self.include_bounds(cmd.x, cmd.width, width);
        self.opaque.push(cmd);
    }

    fn bounds(&self) -> Option<(u16, u16)> {
        self.bounds
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
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    width: u16,
    rows: Vec<RowCommands>,
    attrs: Vec<FillAttrs>,
    next_path_id: u32,
}

impl CommandBucketer {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let num_rows = height.div_ceil(Tile::HEIGHT) as usize;
        Self {
            width,
            rows: (0..num_rows).map(|_| RowCommands::default()).collect(),
            attrs: Vec::new(),
            next_path_id: 1,
        }
    }

    pub(crate) fn rows(&self) -> &[RowCommands] {
        &self.rows
    }

    pub(crate) fn attrs(&self) -> &[FillAttrs] {
        &self.attrs
    }

    pub(crate) fn reset(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
        self.attrs.clear();
        self.next_path_id = 1;
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
        let can_depth_cull = blend_mode == BlendMode::default() && mask.is_none();

        for i in 0..strip_buf.len() - 1 {
            let strip = &strip_buf[i];
            if strip.x >= self.width {
                continue;
            }

            let row_idx = strip.strip_y() as usize;
            if row_idx >= self.rows.len() {
                break;
            }

            let next_strip = &strip_buf[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let x0 = strip.x;
            let x1 = x0.saturating_add(strip_width);
            let clipped_x1 = x1.min(self.width);

            if x0 < clipped_x1 {
                self.rows[row_idx].push_cmd(
                    Cmd::AlphaFill(AlphaFillCmd {
                        x: x0,
                        width: clipped_x1 - x0,
                        alpha_idx: strip.alpha_idx(),
                        attrs_idx,
                    }),
                    self.width,
                );
            }

            if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
                let fill_x0 = clipped_x1;
                let fill_x1 = next_strip.x.min(self.width);
                if fill_x0 < fill_x1 {
                    self.push_fill(
                        row_idx,
                        fill_x0,
                        fill_x1 - fill_x0,
                        &paint,
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
        attrs_idx: u32,
        can_depth_cull: bool,
        encoded_paints: &[EncodedPaint],
    ) {
        let row = &mut self.rows[row_idx];
        if !can_depth_cull || !paint_is_opaque(paint, encoded_paints) {
            row.push_cmd(
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
            row.push_cmd(
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
            row.push_cmd(
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
            );
        }

        if aligned_end < end {
            row.push_cmd(
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
    width: u16,
    scratch: Vec<T::Numeric>,
    paint_buf: Vec<T::Numeric>,
    f32_buf: Vec<f32>,
    depth: Vec<u32>,
    _marker: PhantomData<T>,
}

impl<S: Simd, T: RowRenderKernel<S>> RowFine<S, T> {
    fn new(simd: S, width: u16) -> Self {
        let scratch_width = usize::from(width.next_multiple_of(Tile::WIDTH));
        let scratch_len = scratch_width * TILE_HEIGHT_COMPONENTS;
        Self {
            simd,
            width,
            scratch: vec![T::Numeric::ZERO; scratch_len],
            paint_buf: Vec::new(),
            f32_buf: Vec::new(),
            depth: vec![0; usize::from(width.div_ceil(DEPTH_BUCKET_WIDTH))],
            _marker: PhantomData,
        }
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
                self.scratch[scratch_start..scratch_start + scratch_len].fill(T::Numeric::ZERO);
                self.depth[depth_start..depth_end].fill(0);
            },
        )
    }

    #[inline(always)]
    fn fill_solid(&mut self, x: u16, width: u16, color: PremulColor, alphas: Option<&[u8]>) {
        if width == 0 {
            return;
        }

        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        T::fill_solid(
            self.simd,
            &mut self.scratch[start..start + len],
            color,
            alphas,
        );
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
        T::blend(
            self.simd,
            &mut self.scratch[start..start + len],
            x,
            y,
            iter::repeat(T::Composite::from_color(self.simd, color)),
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

        fill_indexed_paint::<S, T>(
            self.simd,
            &mut self.scratch,
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

    fn render_opaque(
        &mut self,
        cmd: FillCmd,
        row_y: u16,
        attrs: &[FillAttrs],
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let attrs = &attrs[cmd.attrs_idx as usize];
        let start = usize::from(cmd.x / DEPTH_BUCKET_WIDTH);
        let end = usize::from((cmd.x + cmd.width) / DEPTH_BUCKET_WIDTH);

        self.for_depth_runs(
            start,
            end,
            |depth| *depth == 0,
            |this, run| {
                let x = (run.start as u16) * DEPTH_BUCKET_WIDTH;
                let width = (run.end - run.start) as u16 * DEPTH_BUCKET_WIDTH;
                match &attrs.paint {
                    Paint::Solid(color) => this.fill_solid(x, width, *color, None),
                    Paint::Indexed(index) => this.fill_indexed(
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
                for depth in &mut this.depth[run.start..run.end] {
                    *depth = attrs.path_id;
                }
            },
        );
    }

    fn render_cmd(
        &mut self,
        cmd: Cmd,
        row_y: u16,
        alphas: &[u8],
        attrs: &[FillAttrs],
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let cmd_x = cmd.x();
        let cmd_end = (cmd_x + cmd.width()).min(self.width);
        if cmd_x >= cmd_end {
            return;
        }

        let attrs = &attrs[cmd.attrs_idx() as usize];
        let start = usize::from(cmd_x / DEPTH_BUCKET_WIDTH);
        let end = usize::from(cmd_end.div_ceil(DEPTH_BUCKET_WIDTH));
        self.for_visible_runs(attrs.path_id, start, end, |this, run| {
            let x = cmd_x.max(run.start as u16 * DEPTH_BUCKET_WIDTH);
            let end = cmd_end.min(run.end as u16 * DEPTH_BUCKET_WIDTH);
            match cmd {
                Cmd::Fill(_) => match &attrs.paint {
                    Paint::Solid(color) => this.fill_solid_with_attrs(
                        x,
                        row_y,
                        end - x,
                        *color,
                        attrs.blend_mode,
                        attrs.mask.as_ref(),
                        None,
                    ),
                    Paint::Indexed(index) => this.fill_indexed(
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
                        Paint::Solid(color) => this.fill_solid_with_attrs(
                            x,
                            row_y,
                            end - x,
                            *color,
                            attrs.blend_mode,
                            attrs.mask.as_ref(),
                            Some(&alphas[alpha_offset..]),
                        ),
                        Paint::Indexed(index) => {
                            this.fill_indexed(
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
            }
        });
    }

    fn for_depth_runs<P, F>(&mut self, start: usize, end: usize, mut predicate: P, mut f: F)
    where
        P: FnMut(&u32) -> bool,
        F: FnMut(&mut Self, core::ops::Range<usize>),
    {
        let mut run_start = None;
        let mut idx = start;
        while idx < end {
            if idx + 4 <= end {
                let chunk = u32x4::from_slice(self.simd, &self.depth[idx..idx + 4]);
                let selected = self.simd.select_u32x4(
                    chunk.simd_eq(u32x4::splat(self.simd, 0)),
                    u32x4::splat(self.simd, 1),
                    u32x4::splat(self.simd, 0),
                );
                let all_zero = selected[0] + selected[1] + selected[2] + selected[3] == 4;
                if all_zero && predicate(&0) {
                    if run_start.is_none() {
                        run_start = Some(idx);
                    }
                    idx += 4;
                    continue;
                }
            }

            if predicate(&self.depth[idx]) {
                if run_start.is_none() {
                    run_start = Some(idx);
                }
            } else if let Some(start) = run_start.take() {
                f(self, start..idx);
            }
            idx += 1;
        }

        if let Some(start) = run_start {
            f(self, start..end);
        }
    }

    fn for_visible_runs<F>(&mut self, path_id: u32, start: usize, end: usize, mut f: F)
    where
        F: FnMut(&mut Self, core::ops::Range<usize>),
    {
        let mut run_start = None;
        let mut idx = start;
        while idx < end {
            if idx + 4 <= end {
                let chunk = u32x4::from_slice(self.simd, &self.depth[idx..idx + 4]);
                let hidden = self.simd.select_u32x4(
                    self.simd
                        .simd_lt_u32x4(u32x4::splat(self.simd, path_id), chunk),
                    u32x4::splat(self.simd, 1),
                    u32x4::splat(self.simd, 0),
                );
                let hidden_count = hidden[0] + hidden[1] + hidden[2] + hidden[3];
                if hidden_count == 0 {
                    if run_start.is_none() {
                        run_start = Some(idx);
                    }
                    idx += 4;
                    continue;
                }
                if hidden_count == 4 {
                    if let Some(start) = run_start.take() {
                        f(self, start..idx);
                    }
                    idx += 4;
                    continue;
                }
            }

            if self.depth[idx] <= path_id {
                if run_start.is_none() {
                    run_start = Some(idx);
                }
            } else if let Some(start) = run_start.take() {
                f(self, start..idx);
            }
            idx += 1;
        }

        if let Some(start) = run_start {
            f(self, start..end);
        }
    }

    fn pack(&self, row_idx: usize, row_height: usize, x: u16, width: u16, buffer: &mut [u8]) {
        let out_width = usize::from(self.width);
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
                &self.scratch,
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
                &self.scratch,
                prefix_start,
                prefix_width,
                out_width,
                out,
            );
        }

        let tail_start = prefix_start + prefix_width;
        if tail_start < end {
            T::pack_tail(
                &self.scratch,
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
    let mut fine = RowFine::<S, T>::new(simd, width);
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
            fine.render_opaque(cmd, row_y, bucketer.attrs(), encoded_paints, image_resolver);
        }
        for &cmd in &row.cmds {
            fine.render_cmd(
                cmd,
                row_y,
                alphas,
                bucketer.attrs(),
                encoded_paints,
                image_resolver,
            );
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
