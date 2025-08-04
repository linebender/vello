// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::Level;
use crate::dispatch::multi_threaded::{CoarseTask, CoarseTaskSender, Path, RenderTask};
use crate::peniko::Fill;
use crate::strip_generator::StripGenerator;
use std::vec::Vec;
use vello_common::strip::Strip;

#[derive(Debug)]
pub(crate) struct Worker {
    strip_generator: StripGenerator,
    thread_id: u8,
}

impl Worker {
    pub(crate) fn new(width: u16, height: u16, thread_id: u8, level: Level) -> Self {
        let strip_generator = StripGenerator::new(width, height, level);

        Self {
            strip_generator,
            thread_id,
        }
    }

    pub(crate) fn init(&mut self, alphas: Vec<u8>) {
        self.strip_generator.set_alpha_buf(alphas);
    }

    pub(crate) fn thread_id(&self) -> u8 {
        self.thread_id
    }

    pub(crate) fn reset(&mut self) {
        self.strip_generator.reset();
    }

    pub(crate) fn run_render_tasks(
        &mut self,
        task_idx: u32,
        tasks: Vec<RenderTask>,
        result_sender: &mut CoarseTaskSender,
    ) {
        let mut task_buf = Vec::with_capacity(tasks.len());

        for task in tasks {
            match task {
                RenderTask::FillPath {
                    path,
                    transform,
                    paint,
                    fill_rule,
                    anti_alias,
                } => {
                    let func = |strips: &[Strip]| {
                        let coarse_command = CoarseTask::Render {
                            thread_id: self.thread_id,
                            strips: strips.into(),
                            fill_rule,
                            paint,
                        };

                        task_buf.push(coarse_command);
                    };

                    match path {
                        Path::Bez(b) => {
                            self.strip_generator
                                .generate_filled_path(&b, fill_rule, transform, anti_alias, func);
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_filled_path(
                                s.elements(),
                                fill_rule,
                                transform,
                                anti_alias,
                                func,
                            );
                        }
                    }
                }
                RenderTask::StrokePath {
                    path,
                    transform,
                    paint,
                    stroke,
                    anti_alias,
                } => {
                    let func = |strips: &[Strip]| {
                        let coarse_command = CoarseTask::Render {
                            thread_id: self.thread_id,
                            strips: strips.into(),
                            fill_rule: Fill::NonZero,
                            paint,
                        };

                        task_buf.push(coarse_command);
                    };

                    match path {
                        Path::Bez(b) => {
                            self.strip_generator
                                .generate_stroked_path(&b, &stroke, transform, anti_alias, func);
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_stroked_path(
                                s.elements(),
                                &stroke,
                                transform,
                                anti_alias,
                                func,
                            );
                        }
                    }
                }
                RenderTask::PushLayer {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    fill_rule,
                    anti_alias,
                } => {
                    let clip = if let Some((c, transform)) = clip_path {
                        let mut strip_buf = &[][..];
                        self.strip_generator.generate_filled_path(
                            c,
                            fill_rule,
                            transform,
                            anti_alias,
                            |strips| strip_buf = strips,
                        );

                        Some((strip_buf.into(), fill_rule))
                    } else {
                        None
                    };

                    let coarse_command = CoarseTask::PushLayer {
                        thread_id: self.thread_id,
                        clip_path: clip,
                        blend_mode,
                        mask,
                        opacity,
                    };

                    task_buf.push(coarse_command);
                }
                RenderTask::PopLayer => {
                    task_buf.push(CoarseTask::PopLayer);
                }
            }
        }

        result_sender.send(task_idx as usize, task_buf).unwrap();
    }

    pub(crate) fn finalize(&mut self) -> Vec<u8> {
        self.strip_generator.take_alpha_buf()
    }
}
