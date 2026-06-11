// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared planning for scheduled layer filters.

use crate::{
    filter::{GpuFilterData, filter_type, pass_kind},
    scene::FilterData,
};
use smallvec::SmallVec;
use vello_common::filter::{PreparedFilter, gaussian_blur::DecimationSizer};

type FilterPasses = SmallVec<[ScheduledFilterPass; 8]>;

/// A render target role in a scheduled filter pass.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FilterPassSource {
    /// The rendered, unfiltered layer contents.
    Source,
    /// First scratch target.
    Scratch0,
    /// Second scratch target.
    Scratch1,
    /// The final filtered target sampled by parent layers.
    Final,
}

/// One GPU filter pass after expanding a high-level filter into executable work.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScheduledFilterPass {
    pub(crate) input: FilterPassSource,
    pub(crate) output: FilterPassSource,
    pub(crate) pass_kind: u32,
    pub(crate) src_size: (u16, u16),
    pub(crate) dst_size: (u16, u16),
}

/// Backend-independent filter execution plan for a recorded layer.
#[derive(Debug)]
pub(crate) struct LayerFilterPlan {
    gpu_filter: GpuFilterData,
    passes: FilterPasses,
    uses_scratch: bool,
}

impl LayerFilterPlan {
    /// Expand the layer filter into the GPU passes required to execute it.
    pub(crate) fn new(filter_data: &FilterData, source_size: (u16, u16)) -> Self {
        let prepared = PreparedFilter::new(&filter_data.filter, &filter_data.transform);
        let gpu_filter = GpuFilterData::from(&prepared);
        let mut plan = Self {
            gpu_filter,
            passes: SmallVec::new(),
            uses_scratch: gpu_filter.is_multi_pass(),
        };

        if !gpu_filter.is_multi_pass() {
            let pass_kind = match gpu_filter.filter_type() {
                filter_type::OFFSET => pass_kind::OFFSET,
                filter_type::FLOOD => pass_kind::FLOOD,
                _ => unimplemented!("unsupported single-pass filter"),
            };
            plan.push(
                FilterPassSource::Source,
                FilterPassSource::Final,
                pass_kind,
                source_size,
                source_size,
            );
            return plan;
        }

        plan.push_multi_pass(source_size);
        plan
    }

    /// Serialized data consumed by filter shaders.
    #[inline]
    pub(crate) fn gpu_filter(&self) -> GpuFilterData {
        self.gpu_filter
    }

    /// Whether executing this filter requires two scratch targets.
    #[inline]
    pub(crate) fn uses_scratch(&self) -> bool {
        self.uses_scratch
    }

    /// Scheduled GPU passes.
    #[inline]
    pub(crate) fn passes(&self) -> &[ScheduledFilterPass] {
        &self.passes
    }

    fn push_multi_pass(&mut self, source_size: (u16, u16)) {
        let mut current = FilterPassSource::Source;
        let mut toggle = 0_usize;
        let mut sizer = DecimationSizer::default();
        sizer.reset(source_size.0, source_size.1);

        if self.gpu_filter.filter_type() == filter_type::DROP_SHADOW {
            let current_size = sizer.current();
            let output = next_filter_scratch(&mut toggle);
            self.push(
                current,
                output,
                pass_kind::OFFSET,
                current_size,
                current_size,
            );
            current = output;
        }

        let n_decimations = self.gpu_filter.n_decimations();
        for _ in 0..n_decimations {
            let src_size = sizer.current();
            let dst_size = sizer.downscale();
            let output = next_filter_scratch(&mut toggle);
            self.push(current, output, pass_kind::DOWNSCALE, src_size, dst_size);
            current = output;
        }

        let current_size = sizer.current();
        let output = next_filter_scratch(&mut toggle);
        self.push(
            current,
            output,
            pass_kind::BLUR_H,
            current_size,
            current_size,
        );
        current = output;

        let is_drop_shadow = self.gpu_filter.filter_type() == filter_type::DROP_SHADOW;
        let mut final_pass = pass_kind::BLUR_V;
        if n_decimations > 0 {
            let current_size = sizer.current();
            let output = next_filter_scratch(&mut toggle);
            self.push(
                current,
                output,
                pass_kind::BLUR_V,
                current_size,
                current_size,
            );
            current = output;

            for _ in 0..n_decimations - 1 {
                let src_size = sizer.current();
                let dst_size = sizer.upscale();
                let output = next_filter_scratch(&mut toggle);
                self.push(current, output, pass_kind::UPSCALE, src_size, dst_size);
                current = output;
            }

            final_pass = pass_kind::UPSCALE;
        }

        let src_size = sizer.current();
        let dst_size = if n_decimations > 0 {
            sizer.upscale()
        } else {
            src_size
        };
        let output = if is_drop_shadow {
            next_filter_scratch(&mut toggle)
        } else {
            FilterPassSource::Final
        };
        self.push(current, output, final_pass, src_size, dst_size);
        current = output;

        if is_drop_shadow {
            self.push(
                current,
                FilterPassSource::Final,
                pass_kind::COMPOSITE_DROP_SHADOW,
                dst_size,
                dst_size,
            );
        }
    }

    fn push(
        &mut self,
        input: FilterPassSource,
        output: FilterPassSource,
        pass_kind: u32,
        src_size: (u16, u16),
        dst_size: (u16, u16),
    ) {
        self.passes.push(ScheduledFilterPass {
            input,
            output,
            pass_kind,
            src_size,
            dst_size,
        });
    }
}

/// Select the backend target for a scheduled filter pass endpoint.
pub(crate) fn filter_pass_target<'a, T>(
    source: FilterPassSource,
    initial: &'a T,
    scratch_0: &'a T,
    scratch_1: &'a T,
    final_target: &'a T,
) -> &'a T {
    match source {
        FilterPassSource::Source => initial,
        FilterPassSource::Scratch0 => scratch_0,
        FilterPassSource::Scratch1 => scratch_1,
        FilterPassSource::Final => final_target,
    }
}

fn next_filter_scratch(toggle: &mut usize) -> FilterPassSource {
    let output = match *toggle {
        0 => FilterPassSource::Scratch0,
        _ => FilterPassSource::Scratch1,
    };
    *toggle = (*toggle + 1) % 2;
    output
}
