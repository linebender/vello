// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "f32_pipeline")]
use crate::fine::F32Kernel;
#[cfg(feature = "u8_pipeline")]
use crate::fine::U8Kernel;
use crate::fine::{COLOR_COMPONENTS, FineKernel, Numeric, TILE_HEIGHT_COMPONENTS};
use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "u8_pipeline")]
use bytemuck::cast_slice;
use core::marker::PhantomData;
use vello_common::color::palette::css::TRANSPARENT;
use vello_common::fearless_simd::*;
use vello_common::paint::{Paint, PremulColor};
use vello_common::strip::Strip;
use vello_common::tile::Tile;

const DEPTH_BUCKET_WIDTH: u16 = 32;

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
    fn path_id(&self) -> u32 {
        match self {
            Self::Fill(cmd) => cmd.path_id,
            Self::AlphaFill(cmd) => cmd.path_id,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) color: PremulColor,
    pub(crate) path_id: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) color: PremulColor,
    pub(crate) path_id: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpaqueCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) color: PremulColor,
    pub(crate) path_id: u32,
}

#[derive(Debug, Default)]
pub(crate) struct RowCommands {
    pub(crate) cmds: Vec<Cmd>,
    pub(crate) opaque: Vec<OpaqueCmd>,
}

impl RowCommands {
    fn clear(&mut self) {
        self.cmds.clear();
        self.opaque.clear();
    }
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    width: u16,
    rows: Vec<RowCommands>,
    next_path_id: u32,
}

impl CommandBucketer {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let num_rows = height.div_ceil(Tile::HEIGHT) as usize;
        Self {
            width,
            rows: (0..num_rows).map(|_| RowCommands::default()).collect(),
            next_path_id: 1,
        }
    }

    pub(crate) fn rows(&self) -> &[RowCommands] {
        &self.rows
    }

    pub(crate) fn reset(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
        self.next_path_id = 1;
    }

    pub(crate) fn generate(&mut self, strip_buf: &[Strip], paint: Paint) {
        let Paint::Solid(color) = paint else {
            unimplemented!("row-bucket prototype only supports solid paints");
        };

        if strip_buf.is_empty() {
            return;
        }

        let path_id = self.next_path_id;
        self.next_path_id = self
            .next_path_id
            .checked_add(1)
            .expect("row-bucket path ID overflow");

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
                self.rows[row_idx].cmds.push(Cmd::AlphaFill(AlphaFillCmd {
                    x: x0,
                    width: clipped_x1 - x0,
                    alpha_idx: strip.alpha_idx(),
                    color,
                    path_id,
                }));
            }

            if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
                let fill_x0 = clipped_x1;
                let fill_x1 = next_strip.x.min(self.width);
                if fill_x0 < fill_x1 {
                    self.push_fill(row_idx, fill_x0, fill_x1 - fill_x0, color, path_id);
                }
            }
        }
    }

    fn push_fill(&mut self, row_idx: usize, x: u16, width: u16, color: PremulColor, path_id: u32) {
        let row = &mut self.rows[row_idx];
        if !color.is_opaque() {
            row.cmds.push(Cmd::Fill(FillCmd {
                x,
                width,
                color,
                path_id,
            }));
            return;
        }

        let end = x + width;
        let aligned_x = x.next_multiple_of(DEPTH_BUCKET_WIDTH).min(end);
        let aligned_end = (end / DEPTH_BUCKET_WIDTH) * DEPTH_BUCKET_WIDTH;

        if aligned_x >= aligned_end {
            row.cmds.push(Cmd::Fill(FillCmd {
                x,
                width,
                color,
                path_id,
            }));
            return;
        }

        if x < aligned_x {
            row.cmds.push(Cmd::Fill(FillCmd {
                x,
                width: aligned_x - x,
                color,
                path_id,
            }));
        }

        if aligned_x < aligned_end {
            row.opaque.push(OpaqueCmd {
                x: aligned_x,
                width: aligned_end - aligned_x,
                color,
                path_id,
            });
        }

        if aligned_end < end {
            row.cmds.push(Cmd::Fill(FillCmd {
                x: aligned_end,
                width: end - aligned_end,
                color,
                path_id,
            }));
        }
    }
}

pub(crate) trait RowRenderKernel<S: Simd>: FineKernel<S> {
    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>);

    fn pack_prefix(
        simd: S,
        scratch: &[Self::Numeric],
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
        <Self as FineKernel<S>>::alpha_composite_solid(simd, dest, color, alphas);
    }

    fn pack_prefix(
        simd: S,
        scratch: &[Self::Numeric],
        width: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                pack_u8_prefix(simd, scratch, width, out_width, out);
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
        <Self as FineKernel<S>>::alpha_composite_solid(simd, dest, color, alphas);
    }

    fn pack_prefix(
        _simd: S,
        scratch: &[Self::Numeric],
        width: usize,
        out_width: usize,
        out: &mut [u8],
    ) {
        <Self as RowRenderKernel<S>>::pack_tail(
            scratch,
            0,
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
            depth: vec![0; usize::from(width.div_ceil(DEPTH_BUCKET_WIDTH))],
            _marker: PhantomData,
        }
    }

    fn clear(&mut self) {
        self.simd.vectorize(#[inline(always)] || {
            self.scratch.fill(T::Numeric::ZERO);
            self.depth.fill(0);
        })
    }

    fn fill(&mut self, x: u16, width: u16, color: PremulColor, alphas: Option<&[u8]>) {
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

    fn render_opaque(&mut self, cmd: OpaqueCmd) {
        let start = usize::from(cmd.x / DEPTH_BUCKET_WIDTH);
        let end = usize::from((cmd.x + cmd.width) / DEPTH_BUCKET_WIDTH);

        self.for_depth_runs(
            start,
            end,
            |depth| *depth == 0,
            |this, run| {
                let x = (run.start as u16) * DEPTH_BUCKET_WIDTH;
                let width = (run.end - run.start) as u16 * DEPTH_BUCKET_WIDTH;
                this.fill(x, width, cmd.color, None);
                for depth in &mut this.depth[run.start..run.end] {
                    *depth = cmd.path_id;
                }
            },
        );
    }

    fn render_cmd(&mut self, cmd: Cmd, alphas: &[u8]) {
        let cmd_x = cmd.x();
        let cmd_end = (cmd_x + cmd.width()).min(self.width);
        if cmd_x >= cmd_end {
            return;
        }

        let start = usize::from(cmd_x / DEPTH_BUCKET_WIDTH);
        let end = usize::from(cmd_end.div_ceil(DEPTH_BUCKET_WIDTH));
        self.for_visible_runs(cmd.path_id(), start, end, |this, run| {
            let x = cmd_x.max(run.start as u16 * DEPTH_BUCKET_WIDTH);
            let end = cmd_end.min(run.end as u16 * DEPTH_BUCKET_WIDTH);
            match cmd {
                Cmd::Fill(fill) => this.fill(x, end - x, fill.color, None),
                Cmd::AlphaFill(fill) => {
                    let alpha_offset =
                        fill.alpha_idx as usize + usize::from(x - fill.x) * Tile::HEIGHT as usize;
                    this.fill(x, end - x, fill.color, Some(&alphas[alpha_offset..]));
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

    fn pack(&self, row_idx: usize, row_height: usize, buffer: &mut [u8]) {
        let width = usize::from(self.width);
        let offset = row_idx * Tile::HEIGHT as usize * width * COLOR_COMPONENTS;
        let len = row_height * width * COLOR_COMPONENTS;
        let out = &mut buffer[offset..offset + len];

        let prefix_width = if row_height == Tile::HEIGHT as usize {
            width / Tile::WIDTH as usize * Tile::WIDTH as usize
        } else {
            0
        };

        if prefix_width > 0 {
            T::pack_prefix(self.simd, &self.scratch, prefix_width, width, out);
        }

        if prefix_width < width {
            T::pack_tail(
                &self.scratch,
                prefix_width,
                width - prefix_width,
                row_height,
                width,
                out,
            );
        }
    }
}

#[cfg(feature = "u8_pipeline")]
#[inline(always)]
fn pack_u8_prefix<S: Simd>(
    simd: S,
    scratch: &[u8],
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

    for (idx, col) in scratch[..width * TILE_HEIGHT_COMPONENTS]
        .chunks_exact(CHUNK_LENGTH)
        .enumerate()
    {
        let dest_idx = idx * Tile::WIDTH as usize * COLOR_COMPONENTS;
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
) {
    let mut fine = RowFine::<S, T>::new(simd, width);
    let transparent = PremulColor::from_alpha_color(TRANSPARENT);

    for (row_idx, row) in bucketer.rows().iter().enumerate() {
        let row_y = row_idx as u16 * Tile::HEIGHT;
        if row_y >= height {
            break;
        }

        fine.clear();
        fine.fill(0, width, transparent, None);

        for &cmd in row.opaque.iter().rev() {
            fine.render_opaque(cmd);
        }
        for &cmd in &row.cmds {
            fine.render_cmd(cmd, alphas);
        }

        let row_height = usize::from((height - row_y).min(Tile::HEIGHT));
        fine.pack(row_idx, row_height, buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::{Cmd, CommandBucketer};
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::strip::Strip;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    #[test]
    fn opaque_fill_splits_to_aligned_middle() {
        let mut bucketer = CommandBucketer::new(128, 4);
        let strips = [Strip::new(3, 0, 0, false), Strip::new(100, 0, 0, true)];

        bucketer.generate(&strips, Paint::Solid(color(RED)));

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

        bucketer.generate(&strips, Paint::Solid(color(BLUE)));

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

        bucketer.generate(&strips, Paint::Solid(color(RED)));

        let row = &bucketer.rows()[0];
        assert!(row.opaque.is_empty());
        assert_eq!(row.cmds.len(), 1);
        assert!(matches!(row.cmds[0], Cmd::Fill(cmd) if cmd.x == 8 && cmd.width == 8));
    }
}
