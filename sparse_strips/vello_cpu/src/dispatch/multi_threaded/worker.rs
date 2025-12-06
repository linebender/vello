// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::Level;
use crate::dispatch::multi_threaded::{
    CoarseTask, CoarseTaskSender, CoarseTaskType, RenderTask, RenderTaskType,
};
use std::vec::Vec;
use vello_common::clip::PathDataRef;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};

#[derive(Debug)]
pub(crate) struct Worker {
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    thread_id: u8,
}

impl Worker {
    pub(crate) fn new(width: u16, height: u16, thread_id: u8, level: Level) -> Self {
        let strip_generator = StripGenerator::new(width, height, level);
        let strip_storage = StripStorage::default();

        Self {
            strip_generator,
            strip_storage,
            thread_id,
        }
    }

    pub(crate) fn init(&mut self, alphas: Vec<u8>) {
        self.strip_storage.alphas = alphas;
    }

    pub(crate) fn thread_id(&self) -> u8 {
        self.thread_id
    }

    pub(crate) fn reset(&mut self) {
        self.strip_generator.reset();
    }

    pub(crate) fn run_render_task(
        &mut self,
        mut render_task: RenderTask,
        result_sender: &mut CoarseTaskSender,
    ) {
        let num_tasks = render_task.allocation_group.render_tasks.len();
        self.strip_storage.strips.clear();
        self.strip_storage
            .set_generation_mode(GenerationMode::Append);
        let task_idx = render_task.idx;
        let path_clip = render_task.clip_path.as_ref().map(|c| PathDataRef {
            strips: c.strips.as_ref(),
            alphas: c.alphas.as_ref(),
        });

        for task in render_task
            .allocation_group
            .render_tasks
            .drain(0..num_tasks)
        {
            match task {
                RenderTaskType::FillPath {
                    path_range,
                    transform,
                    paint,
                    fill_rule,
                    blend_mode,
                    aliasing_threshold,
                    mask,
                } => {
                    let start = self.strip_storage.strips.len() as u32;
                    let path = &render_task.allocation_group.path
                        [path_range.start as usize..path_range.end as usize];

                    self.strip_generator.generate_filled_path(
                        path.iter().copied(),
                        fill_rule,
                        transform,
                        aliasing_threshold,
                        &mut self.strip_storage,
                        path_clip,
                    );
                    let end = self.strip_storage.strips.len() as u32;

                    let coarse_command = CoarseTaskType::RenderPath {
                        thread_id: self.thread_id,
                        strips: start..end,
                        blend_mode,
                        paint,
                        mask,
                    };

                    render_task
                        .allocation_group
                        .coarse_tasks
                        .push(coarse_command);
                }
                RenderTaskType::StrokePath {
                    path_range,
                    transform,
                    paint,
                    blend_mode,
                    stroke,
                    aliasing_threshold,
                    mask,
                } => {
                    let start = self.strip_storage.strips.len() as u32;
                    let path = &render_task.allocation_group.path
                        [path_range.start as usize..path_range.end as usize];

                    self.strip_generator.generate_stroked_path(
                        path.iter().copied(),
                        &stroke,
                        transform,
                        aliasing_threshold,
                        &mut self.strip_storage,
                        path_clip,
                    );
                    let end = self.strip_storage.strips.len() as u32;

                    let coarse_command = CoarseTaskType::RenderPath {
                        thread_id: self.thread_id,
                        strips: start..end,
                        blend_mode,
                        paint,
                        mask,
                    };

                    render_task
                        .allocation_group
                        .coarse_tasks
                        .push(coarse_command);
                }
                RenderTaskType::PushLayer {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    fill_rule,
                    aliasing_threshold,
                } => {
                    let clip = if let Some((path_range, transform)) = clip_path {
                        let start = self.strip_storage.strips.len() as u32;
                        let path = &render_task.allocation_group.path
                            [path_range.start as usize..path_range.end as usize];

                        self.strip_generator.generate_filled_path(
                            path.iter().copied(),
                            fill_rule,
                            transform,
                            aliasing_threshold,
                            &mut self.strip_storage,
                            path_clip,
                        );

                        let end = self.strip_storage.strips.len() as u32;

                        Some(start..end)
                    } else {
                        None
                    };

                    let coarse_command = CoarseTaskType::PushLayer {
                        thread_id: self.thread_id,
                        clip_path: clip,
                        blend_mode,
                        mask,
                        opacity,
                    };

                    render_task
                        .allocation_group
                        .coarse_tasks
                        .push(coarse_command);
                }
                RenderTaskType::PopLayer => {
                    render_task
                        .allocation_group
                        .coarse_tasks
                        .push(CoarseTaskType::PopLayer);
                }
                RenderTaskType::WideCommand {
                    strip_buf,
                    paint,
                    thread_idx,
                    blend_mode,
                } => {
                    let coarse_command = CoarseTaskType::RenderWideCommand {
                        thread_id: thread_idx,
                        strips: strip_buf,
                        paint,
                        blend_mode,
                        mask: None,
                    };

                    render_task
                        .allocation_group
                        .coarse_tasks
                        .push(coarse_command);
                }
            }
        }

        let taken_strips = std::mem::replace(
            &mut self.strip_storage.strips,
            render_task.allocation_group.strips,
        );
        render_task.allocation_group.strips = taken_strips;

        let task = CoarseTask {
            allocation_group: render_task.allocation_group,
        };

        result_sender.send(task_idx as usize, task).unwrap();
    }

    pub(crate) fn finalize(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.strip_storage.alphas)
    }
}
