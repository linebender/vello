use crate::Level;
use crate::dispatch::multi_threaded::{
    CoarseTask, CoarseCommandSender, OnceLockAlphaStorage, Path, RenderTask,
};
use crate::peniko::Fill;
use crate::strip_generator::StripGenerator;
use std::prelude::rust_2015::Vec;
use std::sync::Arc;
use vello_common::strip::Strip;

#[derive(Debug)]
pub(crate) struct Worker {
    strip_generator: StripGenerator,
    thread_id: u8,
    pub(crate) alpha_storage: Arc<OnceLockAlphaStorage>,
}

impl Worker {
    pub(crate) fn new(
        width: u16,
        height: u16,
        thread_id: u8,
        alpha_storage: Arc<OnceLockAlphaStorage>,
        level: Level,
    ) -> Self {
        let strip_generator = StripGenerator::new(width, height, level);

        Self {
            strip_generator,
            thread_id,
            alpha_storage,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.strip_generator.reset();
    }

    pub(crate) fn run_tasks(
        &mut self,
        task_idx: u32,
        tasks: Vec<RenderTask>,
        result_sender: &mut CoarseCommandSender,
    ) {
        let mut task_buf = Vec::with_capacity(tasks.len());

        for task in tasks {
            match task {
                RenderTask::FillPath {
                    path,
                    transform,
                    paint,
                    fill_rule,
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
                                .generate_filled_path(&b, fill_rule, transform, func);
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_filled_path(
                                s.elements(),
                                fill_rule,
                                transform,
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
                                .generate_stroked_path(&b, &stroke, transform, func);
                        }
                        Path::Small(s) => {
                            self.strip_generator.generate_stroked_path(
                                s.elements(),
                                &stroke,
                                transform,
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
                } => {
                    let clip = if let Some((c, transform)) = clip_path {
                        let mut strip_buf = &[][..];
                        self.strip_generator.generate_filled_path(
                            c,
                            fill_rule,
                            transform,
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

    pub(crate) fn place_alphas(&mut self) {
        let alpha_data = self.strip_generator.alpha_buf().to_vec();
        let _ = self.alpha_storage.store(self.thread_id, alpha_data);
    }
}
