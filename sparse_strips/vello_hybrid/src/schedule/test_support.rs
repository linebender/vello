// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{RendererBackend, Schedule, ScheduleStorage};
use crate::draw::Draw;
use crate::paint::PaintResolver;
use crate::scene::LayersConfig;
use crate::target::RootTarget;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::filter_effects::Filter;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::kurbo::{Rect, Shape};
use vello_common::peniko::BlendMode;
use vello_common::peniko::color::palette::css::BLUE;

pub(super) struct SceneCase {
    pub(super) scene: Scene,
}

impl SceneCase {
    pub(super) fn new(width: u16, height: u16) -> Self {
        Self {
            scene: Scene::new(width, height),
        }
    }

    pub(super) fn draw(&mut self, rect: Rect, alpha: f32) {
        self.scene.set_paint(BLUE.with_alpha(alpha));
        self.scene.fill_rect(&rect);
    }

    pub(super) fn draw_at(&mut self, x: f64, alpha: f32) {
        self.draw(Rect::new(x, 0.0, x + 4.0, 4.0), alpha);
    }

    pub(super) fn layer(&mut self, f: impl FnOnce(&mut Self)) {
        self.layer_with(None, None, None, f);
    }

    pub(super) fn layer_with(
        &mut self,
        clip: Option<Rect>,
        blend_mode: Option<BlendMode>,
        filter: Option<Filter>,
        f: impl FnOnce(&mut Self),
    ) {
        let clip = clip.map(|rect| rect.to_path(0.1));
        self.scene
            .push_layer(clip.as_ref(), blend_mode, None, None, filter);
        f(self);
        self.scene.pop_layer();
    }

    pub(super) fn schedule(
        &self,
        root_target: RootTarget,
        texture_size: SizeU16,
        max_textures: usize,
    ) -> Result<ScheduledCase, RenderError> {
        let mut storage = ScheduleStorage::default();
        let schedule = self.schedule_into(&mut storage, root_target, texture_size, max_textures)?;

        Ok(ScheduledCase {
            schedule,
            storage,
            root_target,
        })
    }

    pub(super) fn schedule_into(
        &self,
        storage: &mut ScheduleStorage,
        root_target: RootTarget,
        texture_size: SizeU16,
        max_textures: usize,
    ) -> Result<Schedule, RenderError> {
        let encoded = self.scene.encoded_paints.borrow();
        let offsets = vec![0; encoded.len()];
        Schedule::try_new(
            storage,
            &self.scene,
            root_target,
            PaintResolver::new(&encoded, &offsets),
            texture_size,
            LayersConfig {
                max_textures: Some(max_textures),
                ..Default::default()
            },
        )
    }

    pub(super) fn schedule_root(&self) -> ScheduledCase {
        self.schedule(RootTarget::UserSurface, SizeU16::new(64), 8)
            .unwrap()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DrawView {
    pub(super) x: Vec<u16>,
    pub(super) has_child_layer: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct RoundView {
    pub(super) binding: [Option<u16>; 2],
    pub(super) even: DrawView,
    pub(super) odd: DrawView,
    pub(super) root: DrawView,
    pub(super) clears: [Vec<RectU16>; 2],
    pub(super) filter_passes: [usize; 2],
    pub(super) blend_passes: [usize; 2],
}

pub(super) struct ScheduledCase {
    pub(super) schedule: Schedule,
    pub(super) storage: ScheduleStorage,
    root_target: RootTarget,
}

impl ScheduledCase {
    pub(super) fn execute<R: RendererBackend>(mut self, renderer: &mut R) {
        super::execute(renderer, &mut self.storage, self.schedule, self.root_target);
    }

    pub(super) fn page_counts(&self) -> [usize; 2] {
        self.schedule.layer_page_counts()
    }

    pub(super) fn scratch_texture(&self) -> bool {
        self.schedule.scratch_texture()
    }

    pub(super) fn views(&self) -> Vec<RoundView> {
        self.schedule
            .rounds
            .rounds
            .iter()
            .map(|round| RoundView {
                binding: round.texture_binding_page_indices(),
                even: self.draw_view(&round.layer_texture_passes[0].draw),
                odd: self.draw_view(&round.layer_texture_passes[1].draw),
                root: self.draw_view(&round.root_draw),
                clears: round.layer_texture_clears.clone(),
                filter_passes: core::array::from_fn(|index| {
                    round.layer_texture_passes[index].filter_ranges.len()
                }),
                blend_passes: core::array::from_fn(|index| {
                    round.layer_texture_passes[index].blend_ranges.len()
                }),
            })
            .collect()
    }

    pub(super) fn opaque_x(&self) -> Vec<u16> {
        self.storage
            .buffers
            .draw_buffers
            .opaque_strips
            .iter()
            .map(|strip| strip.x)
            .collect()
    }

    pub(super) fn total_clears(&self) -> usize {
        self.schedule
            .rounds
            .rounds
            .iter()
            .flat_map(|round| &round.layer_texture_clears)
            .map(Vec::len)
            .sum()
    }

    fn draw_view(&self, draw: &Draw) -> DrawView {
        DrawView {
            x: self
                .draw_strips(draw)
                .into_iter()
                .map(|strip| strip.x)
                .collect(),
            has_child_layer: draw.has_child_layer,
        }
    }

    fn draw_strips<'a>(&'a self, draw: &'a Draw) -> Vec<&'a GpuStrip> {
        use crate::util::VecExt;

        self.storage
            .buffers
            .draw_buffers
            .strips
            .ranged(&draw.strip_ranges)
            .iter()
            .collect()
    }
}
