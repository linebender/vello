// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::dispatch::multi_threaded::cost::{COST_THRESHOLD, estimate_render_task_cost};
use crate::dispatch::multi_threaded::small_path::Path;
use crate::dispatch::multi_threaded::worker::Worker;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, PathSeg, Stroke};
use crate::peniko::{BlendMode, Fill};
use crate::region::Regions;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
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

mod cost;
mod small_path;
mod worker;

type RenderTasksSender = crossbeam_channel::Sender<(u32, Vec<RenderTask>)>;
type CoarseCommandSender = ordered_channel::Sender<Vec<CoarseTask>>;
type CoarseCommandReceiver = ordered_channel::Receiver<Vec<CoarseTask>>;

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
    task_idx: u32,
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
                worker.run_tasks(tasks.0, tasks.1, &mut coarse_command_sender);
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

        let cost = estimate_render_task_cost(&task);
        self.task_batch.push(task);
        self.batch_cost += cost;

        if self.batch_cost > COST_THRESHOLD {
            self.flush_tasks();
            self.batch_cost = 0.0;
        }
    }

    fn flush_tasks(&mut self) {
        let tasks = std::mem::replace(&mut self.task_batch, Vec::with_capacity(1));
        self.send_pending_tasks(tasks);
    }

    fn bump_task_idx(&mut self) -> u32 {
        let idx = self.task_idx;
        self.task_idx += 1;
        idx
    }

    fn send_pending_tasks(&mut self, tasks: Vec<RenderTask>) {
        let task_idx = self.bump_task_idx();
        let task_sender = self.task_sender.as_mut().unwrap();
        task_sender.send((task_idx, tasks)).unwrap();
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
                Ok(cmds) => {
                    for cmd in cmds {
                        match cmd {
                            CoarseTask::Render {
                                thread_id,
                                strips,
                                fill_rule,
                                paint,
                            } => self.wide.generate(&strips, fill_rule, paint, thread_id),
                            CoarseTask::PushLayer {
                                thread_id,
                                clip_path,
                                blend_mode,
                                mask,
                                opacity,
                            } => self
                                .wide
                                .push_layer(clip_path, blend_mode, mask, opacity, thread_id),
                            CoarseTask::PopLayer => self.wide.pop_layer(),
                        }
                    }
                }
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
        self.register_task(RenderTask::FillPath {
            path: Path::new(path),
            transform,
            paint,
            fill_rule,
        });
    }

    fn stroke_path(&mut self, path: &BezPath, stroke: &Stroke, transform: Affine, paint: Paint) {
        self.register_task(RenderTask::StrokePath {
            path: Path::new(path),
            transform,
            paint,
            stroke: stroke.clone(),
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
        self.register_task(RenderTask::PushLayer {
            clip_path: clip_path.cloned().map(|c| (c, clip_transform)),
            blend_mode,
            opacity,
            mask,
            fill_rule,
        });
    }

    fn pop_layer(&mut self) {
        self.register_task(RenderTask::PopLayer);
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
pub(crate) enum RenderTask {
    FillPath {
        path: Path,
        transform: Affine,
        paint: Paint,
        fill_rule: Fill,
    },
    StrokePath {
        path: Path,
        transform: Affine,
        paint: Paint,
        stroke: Stroke,
    },
    PushLayer {
        clip_path: Option<(BezPath, Affine)>,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        fill_rule: Fill,
    },
    PopLayer,
}

pub(crate) enum CoarseTask {
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
