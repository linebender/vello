// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, PathSeg, Point, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use crate::strip_generator::StripGenerator;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use crossbeam_channel::TryRecvError;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Barrier, OnceLock};
use thread_local::ThreadLocal;
use vello_common::coarse::{Cmd, Wide};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Fallback, Level, Simd};
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;

// TODO: Fine-tune this parameter.
const COST_THRESHOLD: f32 = 5.0;

type RenderTasksSender = crossbeam_channel::Sender<Vec<RenderTask>>;
type CoarseCommandSender = ordered_channel::Sender<CoarseCommand>;
type CoarseCommandReceiver = ordered_channel::Receiver<CoarseCommand>;

// TODO: In many cases, we pass a reference to an owned path in vello_common/vello_cpu, only
// to later clone it because the multi-threaded dispatcher needs owned access to the structs.
// Figure out whether there is a good way of solving this problem.
pub(crate) struct MultiThreadedDispatcher {
    wide: Wide,
    thread_pool: ThreadPool,
    task_batch: Vec<RenderTask>,
    batch_cost: f32,
    task_sender: Option<RenderTasksSender>,
    workers: Arc<ThreadLocal<RefCell<Worker>>>,
    result_receiver: Option<CoarseCommandReceiver>,
    alpha_storage: Arc<OnceLockAlphaStorage>,
    task_idx: usize,
    num_threads: u16,
    level: Level,
    flushed: bool,
}

impl MultiThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, num_threads: u16, level: Level) -> Self {
        let wide = Wide::new(width, height);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();
        let alpha_storage = Arc::new(OnceLockAlphaStorage::new(num_threads));
        let workers = Arc::new(ThreadLocal::new());
        let task_batch = vec![];

        {
            let alpha_storage = alpha_storage.clone();
            let thread_ids = Arc::new(AtomicU8::new(0));
            let workers = workers.clone();

            // Initialize all workers once in `new`, so that later on we can just call`.get().unwrap()`.
            thread_pool.spawn_broadcast(move |_| {
                let _ = workers.get_or(|| {
                    RefCell::new(Worker::new(
                        width,
                        height,
                        thread_ids.fetch_add(1, Ordering::SeqCst),
                        alpha_storage.clone(),
                        level,
                    ))
                });
            });
        }

        let cmd_idx = 0;
        let batch_cost = 0.0;
        let flushed = false;

        let mut dispatcher = Self {
            wide,
            thread_pool,
            task_batch,
            batch_cost,
            task_idx: cmd_idx,
            flushed,
            workers,
            task_sender: None,
            result_receiver: None,
            level,
            alpha_storage,
            num_threads,
        };

        dispatcher.init();

        dispatcher
    }

    fn init(&mut self) {
        let (render_task_sender, render_task_receiver) = crossbeam_channel::unbounded();
        let (coarse_command_sender, coarse_command_receiver) = ordered_channel::unbounded();
        let workers = self.workers.clone();

        self.task_sender = Some(render_task_sender);
        self.result_receiver = Some(coarse_command_receiver);

        self.thread_pool.spawn_broadcast(move |_| {
            let render_task_receiver = render_task_receiver.clone();
            let mut coarse_command_sender = coarse_command_sender.clone();
            let worker = workers.get().unwrap();
            let mut worker = worker.borrow_mut();

            while let Ok(tasks) = render_task_receiver.recv() {
                worker.run_tasks(tasks, &mut coarse_command_sender);
            }

            // If we reach this point, it means the task_sender has been dropped by the main thread
            // and no more tasks are available. So we are done, and just need to place the alphas
            // of the worker thread in the hashmap. Then, we drop the `coarse_command_sender`. Once all
            // worker thread have dropped `coarse_command_sender`, the main thread knows that all alphas
            // have been placed, and it's safe to proceed.
            worker.place_alphas();

            drop(coarse_command_sender);
        });
    }

    fn register_task(&mut self, task: RenderTask) {
        self.flushed = false;

        let cost = task.estimate_render_time();
        self.task_batch.push(task);
        self.batch_cost += cost;

        if self.batch_cost > COST_THRESHOLD {
            self.flush_tasks();
            self.batch_cost = 0.0;
        }
    }

    fn flush_tasks(&mut self) {
        let tasks = std::mem::replace(&mut self.task_batch, Vec::with_capacity(50));
        self.send_pending_tasks(tasks);
    }

    fn bump_task_idx(&mut self) -> usize {
        let idx = self.task_idx;
        self.task_idx += 1;
        idx
    }

    fn send_pending_tasks(&mut self, tasks: Vec<RenderTask>) {
        let task_sender = self.task_sender.as_mut().unwrap();
        task_sender.send(tasks).unwrap();
        self.run_coarse(true);
    }

    // Currently, we do coarse rasterization in two phases:
    //
    // The first phase is when we are still processing new draw commands from the client. After each
    // command, we check whether there are already any generated strips, and if so we do coarse
    // rasterization for them on the main thread. In this case, we want to abort in case there are
    // no more path strips available to process.
    //
    // The second phase is when we are flushing, in which case even if the queue is empty, we only
    // want to abort once all workers have closed the channel (and thus there won't be any more
    // new strips that will be generated.
    //
    // This is why we have the `abort_empty`flag.
    fn run_coarse(&mut self, abort_empty: bool) {
        let result_receiver = self.result_receiver.as_mut().unwrap();

        loop {
            match result_receiver.try_recv() {
                Ok(cmd) => match cmd {
                    CoarseCommand::Render {
                        thread_id,
                        strips,
                        fill_rule,
                        paint,
                    } => self.wide.generate(&strips, fill_rule, paint, thread_id),
                    CoarseCommand::PushLayer {
                        thread_id,
                        clip_path,
                        blend_mode,
                        mask,
                        opacity,
                    } => self
                        .wide
                        .push_layer(clip_path, blend_mode, mask, opacity, thread_id),
                    CoarseCommand::PopLayer => self.wide.pop_layer(),
                },
                Err(e) => match e {
                    TryRecvError::Empty => {
                        if abort_empty {
                            return;
                        }
                    }
                    TryRecvError::Disconnected => return,
                },
            }
        }
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        let mut buffer = Regions::new(width, height, buffer);
        let fines = ThreadLocal::new();
        let wide = &self.wide;
        let alpha_slots = self.alpha_storage.slots();

        self.thread_pool.install(|| {
            buffer.update_regions_par(|region| {
                let x = region.x;
                let y = region.y;

                let mut fine = fines
                    .get_or(|| RefCell::new(Fine::<S, F>::new(simd)))
                    .borrow_mut();

                let wtile = wide.get(x, y);
                fine.set_coords(x, y);

                fine.clear(wtile.bg);
                for cmd in &wtile.cmds {
                    let thread_idx = match cmd {
                        Cmd::AlphaFill(a) => Some(a.thread_idx),
                        Cmd::ClipStrip(a) => Some(a.thread_idx),
                        _ => None,
                    };

                    let alphas = thread_idx
                        .and_then(|i| alpha_slots[i as usize].get())
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    fine.run_cmd(cmd, alphas, encoded_paints);
                }

                fine.pack(region);
            });
        });
    }
}

impl Dispatcher for MultiThreadedDispatcher {
    fn wide(&self) -> &Wide {
        &self.wide
    }

    fn fill_path(&mut self, path: &BezPath, fill_rule: Fill, transform: Affine, paint: Paint) {
        let task_idx = self.bump_task_idx();

        self.register_task(RenderTask::FillPath {
            path: path.clone(),
            transform,
            paint,
            fill_rule,
            task_idx,
        });
    }

    fn stroke_path(&mut self, path: &BezPath, stroke: &Stroke, transform: Affine, paint: Paint) {
        let task_idx = self.bump_task_idx();

        self.register_task(RenderTask::StrokePath {
            path: path.clone(),
            transform,
            paint,
            stroke: stroke.clone(),
            task_idx,
        });
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
    ) {
        let task_idx = self.bump_task_idx();

        self.register_task(RenderTask::PushLayer {
            clip_path: clip_path.cloned().map(|c| (c, clip_transform)),
            blend_mode,
            opacity,
            mask,
            fill_rule,
            task_idx,
        });
    }

    fn pop_layer(&mut self) {
        let task_idx = self.bump_task_idx();

        self.register_task(RenderTask::PopLayer { task_idx });
    }

    fn reset(&mut self) {
        self.wide.reset();
        self.task_batch.clear();
        self.batch_cost = 0.0;
        self.task_idx = 0;
        self.flushed = false;
        self.task_sender = None;
        self.result_receiver = None;
        // TODO: We want to re-use the allocations of the storage somehow.
        self.alpha_storage = Arc::new(OnceLockAlphaStorage::new(self.num_threads));

        let workers = self.workers.clone();
        let alpha_storage = self.alpha_storage.clone();
        // + 1 since we also wait on the main thread.
        let barrier = Arc::new(Barrier::new(usize::from(self.num_threads) + 1));
        let t_barrier = barrier.clone();

        self.thread_pool.spawn_broadcast(move |_| {
            let worker = workers.get().unwrap();
            let mut borrowed = worker.borrow_mut();
            borrowed.reset();
            borrowed.alpha_storage = alpha_storage.clone();
            t_barrier.wait();
        });

        barrier.wait();

        self.init();
    }

    fn flush(&mut self) {
        self.flush_tasks();
        let sender = core::mem::take(&mut self.task_sender);
        // Note that dropping the sender will signal to the workers that no more new paths
        // can arrive.
        core::mem::drop(sender);
        self.run_coarse(false);

        self.flushed = true;
    }

    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        assert!(self.flushed, "attempted to rasterize before flushing");

        match render_mode {
            RenderMode::OptimizeSpeed => match self.level {
                #[cfg(all(feature = "std", target_arch = "aarch64"))]
                Level::Neon(n) => {
                    self.rasterize_with::<vello_common::fearless_simd::Neon, U8Kernel>(
                        n,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(w) => {
                    self.rasterize_with::<vello_common::fearless_simd::WasmSimd128, U8Kernel>(
                        w,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                _ => self.rasterize_with::<Fallback, U8Kernel>(
                    Fallback::new(),
                    buffer,
                    width,
                    height,
                    encoded_paints,
                ),
            },
            RenderMode::OptimizeQuality => match self.level {
                #[cfg(all(feature = "std", target_arch = "aarch64"))]
                Level::Neon(n) => {
                    self.rasterize_with::<vello_common::fearless_simd::Neon, F32Kernel>(
                        n,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(w) => {
                    self.rasterize_with::<vello_common::fearless_simd::WasmSimd128, F32Kernel>(
                        w,
                        buffer,
                        width,
                        height,
                        encoded_paints,
                    );
                }
                _ => self.rasterize_with::<Fallback, F32Kernel>(
                    Fallback::new(),
                    buffer,
                    width,
                    height,
                    encoded_paints,
                ),
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct OnceLockAlphaStorage {
    slots: Vec<OnceLock<Vec<u8>>>,
}

impl OnceLockAlphaStorage {
    pub(crate) fn new(num_threads: u16) -> Self {
        Self {
            slots: (0..num_threads).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Store alpha data for a specific thread (called once per thread)
    pub(crate) fn store(&self, thread_id: u8, data: Vec<u8>) -> Result<(), Vec<u8>> {
        self.slots[thread_id as usize].set(data)
    }

    fn slots(&self) -> &[OnceLock<Vec<u8>>] {
        &self.slots
    }

    // TODO: Probably worth adding a take all method and being able to reuse vec allocations somehow
}

impl Debug for MultiThreadedDispatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str("MultiThreadedDispatcher { .. }")
    }
}

#[derive(Debug)]
enum RenderTask {
    FillPath {
        path: BezPath,
        transform: Affine,
        paint: Paint,
        fill_rule: Fill,
        task_idx: usize,
    },
    StrokePath {
        path: BezPath,
        transform: Affine,
        paint: Paint,
        stroke: Stroke,
        task_idx: usize,
    },
    PushLayer {
        clip_path: Option<(BezPath, Affine)>,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        fill_rule: Fill,
        task_idx: usize,
    },
    PopLayer {
        task_idx: usize,
    },
}

impl RenderTask {
    fn estimate_render_time(&self) -> f32 {
        match self {
            Self::FillPath {
                path, transform, ..
            } => {
                let path_cost_data = PathCostData::new(path, *transform);
                estimate_runtime_in_micros(&path_cost_data, false)
            }
            Self::StrokePath {
                path, transform, ..
            } => {
                let path_cost_data = PathCostData::new(path, *transform);
                estimate_runtime_in_micros(&path_cost_data, true)
            }
            Self::PushLayer { clip_path, .. } => clip_path
                .as_ref()
                .map(|c| {
                    let path_cost_data = PathCostData::new(&c.0, c.1);
                    estimate_runtime_in_micros(&path_cost_data, false)
                })
                .unwrap_or(0.0),
            Self::PopLayer { .. } => 0.0,
        }
    }
}

enum CoarseCommand {
    Render {
        thread_id: u8,
        strips: Box<[Strip]>,
        fill_rule: Fill,
        paint: Paint,
    },
    PushLayer {
        thread_id: u8,
        clip_path: Option<(Box<[Strip]>, Fill)>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
    },
    PopLayer,
}

#[derive(Debug)]
struct Worker {
    strip_generator: StripGenerator,
    thread_id: u8,
    alpha_storage: Arc<OnceLockAlphaStorage>,
}

impl Worker {
    fn new(
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

    fn reset(&mut self) {
        self.strip_generator.reset();
    }

    fn run_tasks(&mut self, tasks: Vec<RenderTask>, result_sender: &mut CoarseCommandSender) {
        for task in tasks {
            match task {
                RenderTask::FillPath {
                    path,
                    transform,
                    paint,
                    fill_rule,
                    task_idx,
                } => {
                    let func = |strips: &[Strip]| {
                        let coarse_command = CoarseCommand::Render {
                            thread_id: self.thread_id,
                            strips: strips.into(),
                            fill_rule,
                            paint,
                        };
                        result_sender.send(task_idx, coarse_command).unwrap();
                    };

                    self.strip_generator
                        .generate_filled_path(&path, fill_rule, transform, func);
                }
                RenderTask::StrokePath {
                    path,
                    transform,
                    paint,
                    stroke,
                    task_idx,
                } => {
                    let func = |strips: &[Strip]| {
                        let coarse_command = CoarseCommand::Render {
                            thread_id: self.thread_id,
                            strips: strips.into(),
                            fill_rule: Fill::NonZero,
                            paint,
                        };
                        result_sender.send(task_idx, coarse_command).unwrap();
                    };

                    self.strip_generator
                        .generate_stroked_path(&path, &stroke, transform, func);
                }
                RenderTask::PushLayer {
                    clip_path,
                    blend_mode,
                    opacity,
                    mask,
                    fill_rule,
                    task_idx,
                } => {
                    let clip = if let Some((c, transform)) = clip_path {
                        let mut strip_buf = &[][..];
                        self.strip_generator.generate_filled_path(
                            &c,
                            fill_rule,
                            transform,
                            |strips| strip_buf = strips,
                        );

                        Some((strip_buf.into(), fill_rule))
                    } else {
                        None
                    };

                    let coarse_command = CoarseCommand::PushLayer {
                        thread_id: self.thread_id,
                        clip_path: clip,
                        blend_mode,
                        mask,
                        opacity,
                    };

                    result_sender.send(task_idx, coarse_command).unwrap();
                }
                RenderTask::PopLayer { task_idx } => {
                    result_sender
                        .send(task_idx, CoarseCommand::PopLayer)
                        .unwrap();
                }
            }
        }
    }

    fn place_alphas(&mut self) {
        let alpha_data = self.strip_generator.alpha_buf().to_vec();
        let _ = self.alpha_storage.store(self.thread_id, alpha_data);
    }
}

struct PathCostData {
    num_line_segments: u64,
    num_curve_segments: u64,
    path_length: f64,
}

impl PathCostData {
    fn new(path: &BezPath, transform: Affine) -> Self {
        let mut num_line_segments = 0;
        let mut num_curve_segments = 0;
        let mut path_length = 0.0;

        let mut register_path_length = |mut p0: Point, mut p1: Point| {
            p0 = transform * p0;
            p1 = transform * p1;
            // We don't sqrt here because it's too expensive, we just want a rough estimate.
            let dx = (p1.x - p0.x).abs();
            let dy = (p1.y - p0.y).abs();
            path_length += dx + dy;
        };

        for seg in path.segments() {
            match seg {
                PathSeg::Line(l) => {
                    num_line_segments += 1;

                    register_path_length(l.p0, l.p1);
                }
                PathSeg::Quad(q) => {
                    num_curve_segments += 1;

                    register_path_length(q.p0, q.p2);
                }
                PathSeg::Cubic(c) => {
                    num_curve_segments += 1;

                    register_path_length(c.p0, c.p3);
                }
            }
        }

        Self {
            num_line_segments,
            num_curve_segments,
            path_length,
        }
    }
}

// This formula was derived by recording benchmarks of how long paths with a certain set of properties
// take to render and then use "machine learning" to derive a formula which approximates the runtime.
// The result will be far from accurate, but all we need is a ballpark range, and it's important
// that the calculation is relatively fast, since it will be run on the main thread for each path.
// TODO: We will need to update the formula once SIMD + faster stroke expansion landed.
fn estimate_runtime_in_micros(path_cost_data: &PathCostData, is_stroke: bool) -> f32 {
    let line_segments = path_cost_data.num_line_segments as f64;
    let curve_segments = path_cost_data.num_curve_segments as f64;
    let path_length = path_cost_data.path_length;

    let line_log = (1.0 + line_segments).ln();
    let curve_log = (1.0 + curve_segments).ln();
    let path_log = (1.0 + path_length).ln();

    let complexity = 4.6223 - 3.5740 * line_log + 12.5549 * curve_log - 1.4774 * path_log
        + 1.1412 * line_log * path_log;

    let stroke_multiplier = if is_stroke { 6.8737 } else { 1.0 };
    let runtime = complexity * stroke_multiplier;

    runtime.max(0.0) as f32
}
