// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::Level;
use crate::dispatch::multi_threaded::{CoarseTask, CoarseTaskSender, Path, RenderTask};
use std::vec::Vec;
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

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
                    aliasing_threshold,
                } => {
                    match path {
                        Path::Bez(b) => {
                            self.strip_generator.generate_filled_path(
                                &b,
                                fill_rule,
                                transform,
                                aliasing_threshold,
                                &mut self.strip_storage,
                            );
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_filled_path(
                                s.elements(),
                                fill_rule,
                                transform,
                                aliasing_threshold,
                                &mut self.strip_storage,
                            );
                        }
                    }

                    let strips: &[Strip] = &self.strip_storage.strips;

                    let coarse_command = CoarseTask::Render {
                        thread_id: self.thread_id,
                        strips: strips.into(),
                        paint,
                    };

                    task_buf.push(coarse_command);
                }
                RenderTask::StrokePath {
                    path,
                    transform,
                    paint,
                    stroke,
                    aliasing_threshold,
                } => {
                    match path {
                        Path::Bez(b) => {
                            self.strip_generator.generate_stroked_path(
                                &b,
                                &stroke,
                                transform,
                                aliasing_threshold,
                                &mut self.strip_storage,
                            );
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_stroked_path(
                                s.elements(),
                                &stroke,
                                transform,
                                aliasing_threshold,
                                &mut self.strip_storage,
                            );
                        }
                    }

                    let strips: &[Strip] = &self.strip_storage.strips;

                    let coarse_command = CoarseTask::Render {
                        thread_id: self.thread_id,
                        strips: strips.into(),
                        paint,
                    };

                    task_buf.push(coarse_command);
                }
                RenderTask::PushLayer {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    fill_rule,
                    aliasing_threshold,
                } => {
                    let clip = if let Some((c, transform)) = clip_path {
                        self.strip_generator.generate_filled_path(
                            c,
                            fill_rule,
                            transform,
                            aliasing_threshold,
                            &mut self.strip_storage,
                        );

                        let strips: &[Strip] = &self.strip_storage.strips;

                        Some(strips.into())
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
                RenderTask::WideCommand {
                    strip_buf,
                    thread_idx,
                    paint,
                } => {
                    let coarse_command = CoarseTask::Render {
                        thread_id: thread_idx,
                        strips: strip_buf,
                        paint,
                    };

                    task_buf.push(coarse_command);
                }
            }
        }

        result_sender.send(task_idx as usize, task_buf).unwrap();
    }

    pub(crate) fn finalize(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.strip_storage.alphas)
    }
}
