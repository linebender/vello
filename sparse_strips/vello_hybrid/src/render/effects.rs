// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared GPU-side planning for layer effects.

use crate::filter::{FilterInstanceData, pass_kind};
use crate::schedule_new::{BlendOp, FilterOp, FilterScratchRegion};
use crate::util::{IntRect, IntSize};
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::filter::gaussian_blur::DecimationSizer;
use vello_common::peniko::{Compose, Mix};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuBlendInstance {
    pub(crate) dest_origin: [u32; 2],
    pub(crate) source_origin: [u32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) texture_indices: [u32; 2],
    pub(crate) blend_mode: [u32; 2],
    pub(crate) opacity: u32,
    pub(crate) target_size: [u32; 2],
    pub(crate) bbox_origin: [u32; 2],
    pub(crate) source_scene_origin: [u32; 2],
    pub(crate) source_size: [u32; 2],
    pub(crate) _padding: u32,
}

pub(crate) fn gpu_blend_instance(blend: BlendOp, target_size: (u32, u32)) -> GpuBlendInstance {
    let dest_x = blend.parent.x + u32::from(blend.bbox.x0 - blend.parent.scene_bbox.x0);
    let dest_y = blend.parent.y + u32::from(blend.bbox.y0 - blend.parent.scene_bbox.y0);

    GpuBlendInstance {
        dest_origin: [dest_x, dest_y],
        source_origin: [blend.source.x, blend.source.y],
        size: [
            u32::from(blend.bbox.width()),
            u32::from(blend.bbox.height()),
        ],
        texture_indices: [
            u32::try_from(blend.parent.texture_index)
                .expect("layer texture index fits into shader payload"),
            u32::try_from(blend.source.texture_index)
                .expect("layer texture index fits into shader payload"),
        ],
        blend_mode: [
            pack_mix(blend.blend_mode.mix),
            pack_compose(blend.blend_mode.compose),
        ],
        opacity: u32::from(opacity_to_u8(blend.opacity)),
        target_size: [target_size.0, target_size.1],
        bbox_origin: [u32::from(blend.bbox.x0), u32::from(blend.bbox.y0)],
        source_scene_origin: [
            u32::from(blend.source.scene_bbox.x0),
            u32::from(blend.source.scene_bbox.y0),
        ],
        source_size: [
            u32::from(blend.source.scene_bbox.width()),
            u32::from(blend.source.scene_bbox.height()),
        ],
        _padding: 0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FilterTexture {
    Layer(usize),
    Scratch(usize),
}

#[derive(Debug)]
pub(crate) struct ScheduledFilterBatch {
    pub(crate) input: FilterTexture,
    pub(crate) output: FilterTexture,
    pub(crate) original: FilterTexture,
    pub(crate) instances: Vec<FilterInstanceData>,
}

pub(crate) fn build_scheduled_filter_batches(
    filters: &[FilterOp],
    target_size: (u32, u32),
    batches: &mut Vec<ScheduledFilterBatch>,
) {
    let mut passes = Vec::new();
    for filter in filters {
        build_scheduled_filter_passes(*filter, target_size, &mut passes);
    }

    let Some(max_step) = passes.iter().map(|pass| pass.step).max() else {
        return;
    };

    for step in 0..=max_step {
        let step_batch_start = batches.len();
        for pass in passes.iter().copied().filter(|pass| pass.step == step) {
            let original = pass.original.unwrap_or(pass.input);
            if let Some(batch) = batches[step_batch_start..].iter_mut().find(|batch| {
                batch.output == pass.output
                    && batch.input == pass.input
                    && batch.original == original
            }) {
                batch.instances.push(pass.instance);
            } else {
                batches.push(ScheduledFilterBatch {
                    input: pass.input,
                    output: pass.output,
                    original,
                    instances: alloc::vec![pass.instance],
                });
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScheduledFilterPass {
    step: usize,
    input: FilterTexture,
    output: FilterTexture,
    original: Option<FilterTexture>,
    instance: FilterInstanceData,
}

fn build_scheduled_filter_passes(
    op: FilterOp,
    target_size: (u32, u32),
    out: &mut Vec<ScheduledFilterPass>,
) {
    let mut builder = FilterPassBuilder::new(op, target_size);
    match op.gpu_filter.filter_type() {
        crate::filter::filter_type::OFFSET => {
            builder.emit_to_scratch(pass_kind::OFFSET);
            builder.emit_copy_back();
        }
        crate::filter::filter_type::FLOOD => {
            builder.emit_to_scratch(pass_kind::FLOOD);
            builder.emit_copy_back();
        }
        crate::filter::filter_type::GAUSSIAN_BLUR => {
            builder.emit_blur_sequence(op.gpu_filter.n_decimations());
            builder.emit_copy_back();
        }
        crate::filter::filter_type::DROP_SHADOW => {
            builder.emit_to_scratch(pass_kind::OFFSET);
            builder.emit_blur_sequence(op.gpu_filter.n_decimations());
            builder.emit_drop_shadow_composite();
            builder.emit_copy_back();
        }
        _ => unreachable!("unsupported filter type was encoded"),
    }
    builder.finish(out);
}

fn pack_mix(mix: Mix) -> u32 {
    match mix {
        Mix::Normal => 0,
        Mix::Multiply => 1,
        Mix::Screen => 2,
        Mix::Overlay => 3,
        Mix::Darken => 4,
        Mix::Lighten => 5,
        Mix::ColorDodge => 6,
        Mix::ColorBurn => 7,
        Mix::HardLight => 8,
        Mix::SoftLight => 9,
        Mix::Difference => 10,
        Mix::Exclusion => 11,
        Mix::Hue => 12,
        Mix::Saturation => 13,
        Mix::Color => 14,
        Mix::Luminosity => 15,
    }
}

fn pack_compose(compose: Compose) -> u32 {
    match compose {
        Compose::Clear => 0,
        Compose::Copy => 1,
        Compose::Dest => 2,
        Compose::SrcOver => 3,
        Compose::DestOver => 4,
        Compose::SrcIn => 5,
        Compose::DestIn => 6,
        Compose::SrcOut => 7,
        Compose::DestOut => 8,
        Compose::SrcAtop => 9,
        Compose::DestAtop => 10,
        Compose::Xor => 11,
        Compose::Plus => 12,
        Compose::PlusLighter => 13,
    }
}

#[derive(Debug)]
struct FilterPassBuilder {
    op: FilterOp,
    target_size: (u32, u32),
    passes: Vec<ScheduledFilterPass>,
    sizer: DecimationSizer,
    toggle: usize,
    first: bool,
    step: usize,
}

impl FilterPassBuilder {
    fn new(op: FilterOp, target_size: (u32, u32)) -> Self {
        let mut sizer = DecimationSizer::default();
        sizer.reset(
            u16::try_from(op.layer.width).expect("filter layer width fits into DecimationSizer"),
            u16::try_from(op.layer.height).expect("filter layer height fits into DecimationSizer"),
        );
        Self {
            op,
            target_size,
            passes: Vec::new(),
            sizer,
            toggle: 0,
            first: true,
            step: 0,
        }
    }

    fn finish(self, out: &mut Vec<ScheduledFilterPass>) {
        out.extend(self.passes);
    }

    fn initial_texture(&self) -> FilterTexture {
        FilterTexture::Layer(self.op.layer.texture_index)
    }

    fn scratch_region(&self, index: usize) -> FilterScratchRegion {
        self.op.scratches[index].expect("filter pass requires allocated scratch region")
    }

    fn texture_offset(&self, texture: FilterTexture) -> [u32; 2] {
        match texture {
            FilterTexture::Layer(_) => [self.op.layer.x, self.op.layer.y],
            FilterTexture::Scratch(index) => {
                let scratch = self.scratch_region(index);
                [scratch.x, scratch.y]
            }
        }
    }

    fn input(&mut self) -> FilterTexture {
        if self.first {
            self.first = false;
            self.initial_texture()
        } else {
            FilterTexture::Scratch((self.toggle + 1) % 2)
        }
    }

    fn apply_pass_dimensions(&mut self, kind: u32) -> ([u32; 2], [u32; 2]) {
        match kind {
            pass_kind::DOWNSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.downscale();
                (
                    [u32::from(sw), u32::from(sh)],
                    [u32::from(dw), u32::from(dh)],
                )
            }
            pass_kind::UPSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.upscale();
                (
                    [u32::from(sw), u32::from(sh)],
                    [u32::from(dw), u32::from(dh)],
                )
            }
            _ => {
                let (w, h) = self.sizer.current();
                let size = [u32::from(w), u32::from(h)];
                (size, size)
            }
        }
    }

    fn emit(&mut self, kind: u32, output: FilterTexture, original: Option<FilterTexture>) {
        let (src_size, dest_size) = self.apply_pass_dimensions(kind);
        let input = self.input();
        let src_offset = self.texture_offset(input);
        let dest_offset = self.texture_offset(output);
        let original_offset = self.texture_offset(original.unwrap_or(self.initial_texture()));

        self.passes.push(ScheduledFilterPass {
            step: self.step,
            input,
            output,
            original,
            instance: FilterInstanceData {
                src: IntRect::new(src_offset, src_size),
                dest: IntRect::new(dest_offset, dest_size),
                dest_atlas_size: IntSize([self.target_size.0, self.target_size.1]),
                filter_data_offset: self.op.filter_data_offset,
                original: IntRect::new(
                    original_offset,
                    [self.op.layer.width, self.op.layer.height],
                ),
                pass_kind: kind,
            },
        });
        self.step += 1;
    }

    fn emit_to_scratch(&mut self, kind: u32) {
        let scratch = self.toggle;
        self.emit(kind, FilterTexture::Scratch(scratch), None);
        self.toggle = (self.toggle + 1) % 2;
    }

    fn emit_copy_back(&mut self) {
        let input = if self.first {
            self.initial_texture()
        } else {
            FilterTexture::Scratch((self.toggle + 1) % 2)
        };
        self.first = false;
        let src_offset = self.texture_offset(input);
        let dest_offset = self.texture_offset(self.initial_texture());
        self.passes.push(ScheduledFilterPass {
            step: self.step,
            input,
            output: self.initial_texture(),
            original: None,
            instance: FilterInstanceData {
                src: IntRect::new(src_offset, [self.op.layer.width, self.op.layer.height]),
                dest: IntRect::new(dest_offset, [self.op.layer.width, self.op.layer.height]),
                dest_atlas_size: IntSize([self.target_size.0, self.target_size.1]),
                filter_data_offset: self.op.filter_data_offset,
                original: IntRect::new([0, 0], [self.op.layer.width, self.op.layer.height]),
                pass_kind: pass_kind::COPY,
            },
        });
        self.step += 1;
    }

    fn emit_blur_sequence(&mut self, n_decimations: usize) {
        for _ in 0..n_decimations {
            self.emit_to_scratch(pass_kind::DOWNSCALE);
        }
        self.emit_to_scratch(pass_kind::BLUR_H);

        let mut final_pass = pass_kind::BLUR_V;
        if n_decimations > 0 {
            self.emit_to_scratch(pass_kind::BLUR_V);
            for _ in 0..n_decimations - 1 {
                self.emit_to_scratch(pass_kind::UPSCALE);
            }
            final_pass = pass_kind::UPSCALE;
        }
        self.emit_to_scratch(final_pass);
    }

    fn emit_drop_shadow_composite(&mut self) {
        let scratch = self.toggle;
        self.emit(
            pass_kind::COMPOSITE_DROP_SHADOW,
            FilterTexture::Scratch(scratch),
            Some(self.initial_texture()),
        );
        self.toggle = (self.toggle + 1) % 2;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "opacity is clamped to the normalized u8 range before packing"
)]
fn opacity_to_u8(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}
