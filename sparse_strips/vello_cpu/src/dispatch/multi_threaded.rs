// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::RenderMode;
use crate::dispatch::Dispatcher;
use crate::dispatch::multi_threaded::cost::{COST_THRESHOLD, estimate_render_task_cost};
use crate::dispatch::multi_threaded::worker::Worker;
use crate::fine::{F32Kernel, Fine, FineKernel, U8Kernel};
use crate::kurbo::{Affine, BezPath, PathEl, Stroke};
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
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Barrier, Mutex};
use thread_local::ThreadLocal;
use vello_common::clip::ClipContext;
use vello_common::coarse::{Cmd, MODE_CPU, Wide};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::strip_generator::{StripGenerator, StripStorage};

mod cost;
mod worker;

type RenderTaskSender = crossbeam_channel::Sender<RenderTask>;
type CoarseTaskSender = ordered_channel::Sender<CoarseTask>;
type CoarseTaskReceiver = ordered_channel::Receiver<CoarseTask>;

/// A dispatcher for multi-threaded rendering.
///
/// A small note for future contributors: Unfortunately, the logic of this dispatcher as well as
/// the lifecycle of the different fields of the dispatcher can be a bit hard to grasp.
/// The reason for this is that since we have to do a lot of communication across the thread boundary,
/// we have to work with lots of `Option` and `core::mem::take` operations, to ensure that we are
/// not needlessly cloning objects.
///
/// The below comments will hopefully help with understanding the overall structure and lifecycles
/// a bit better.
pub(crate) struct MultiThreadedDispatcher {
    /// The wide tile container.
    wide: Wide,
    clip_context: ClipContext,
    /// The thread pool that is used for dispatching tasks.
    thread_pool: ThreadPool,
    allocation_group: AllocationGroup,
    /// The cost of the current batch.
    batch_cost: f32,
    /// The sender used to dispatch new rendering tasks from the main thread.
    ///
    /// This field will be set once we call the `init` method.
    /// This field will be set back to `None` when running `flush` to drop the value and thus
    /// indicate to receivers that no more rendering tasks will be dispatched from that point onward.
    task_sender: Option<RenderTaskSender>,
    /// Contains one worker object for each thread.
    ///
    /// The workers will be initialized once when building the multi-threaded dispatcher via
    /// `MultiThreadedDispatcher::new`.
    workers: Arc<ThreadLocal<RefCell<Worker>>>,
    /// The receiver for coarse command tasks, used to do coarse rasterization on the main thread.
    ///
    /// Similarly to `task_sender`, this value is set to `None` initially, and will only be set once
    /// we actually call the `init` method (either when creating the dispatcher for the first time, or
    /// when resetting it).
    coarse_task_receiver: Option<CoarseTaskReceiver>,
    /// The storage for alpha values.
    ///
    /// Similarly to the single-threaded dispatcher, we want to be able to reuse the allocation holding
    /// the alpha values across multiple runs of `reset`. However, we have the problem that during path
    /// rendering, each thread needs to have its own allocation. We also need to be able to move
    /// the allocation back and forth between the threads (during path rendering) and the main thread
    /// (during fine rasterization). Because of this, we wrap it in this `MaybePresent` struct.
    ///
    /// During initialization, each thread will "take" the vector allocation out of its slot
    /// (the vector has a length of `num_threads`, so each thread has a slot belonging to itself)
    /// and will put it back to its slot after flushing. Then, during fine rasterization, we
    /// take all slots out of the `MaybePresent` object so that we can easily access each buffer
    /// when running the commands without having to go through the mutex. After fine rasterization,
    /// the slots are put back into the `MaybePresent` object.
    ///
    alpha_storage: MaybePresent<Vec<Vec<u8>>>,
    /// The task index that will be assigned to the next rendering task.
    ///
    /// Since we are rendering the paths on different threads, we need to make sure that they
    /// come back in the right order. The `task_idx` is used to keep track of that order.
    task_idx: u32,
    /// The number of threads active in the thread pool.
    num_threads: u16,
    /// The strip generator for the main thread, only used for recordings.
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    level: Level,
    flushed: bool,
    // So that we can reuse memory allocations across different runs.
    allocations: Allocations,
}

impl MultiThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, num_threads: u16, level: Level) -> Self {
        let wide = Wide::<MODE_CPU>::new(width, height);
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();
        // + 1 because the main thread also stores an alpha buffer, used for recordings.
        let alpha_storage = MaybePresent::new(vec![vec![]; usize::from(num_threads + 1)]);
        let workers = Arc::new(ThreadLocal::new());

        {
            // Start counting from 1, as thread_idx 0 is reserved for the main thread.
            let thread_ids = Arc::new(AtomicU8::new(1));
            let workers = workers.clone();

            // Create all workers once in `new`, so that later on we can just call`.get().unwrap()`.
            thread_pool.spawn_broadcast(move |_| {
                let thread_id = thread_ids.fetch_add(1, Ordering::SeqCst);
                let worker = Worker::new(width, height, thread_id, level);

                let _ = workers.get_or(|| RefCell::new(worker));
            });
        }

        let task_idx = 0;
        let batch_cost = 0.0;
        let flushed = false;

        let mut dispatcher = Self {
            wide,
            thread_pool,
            allocations: Default::default(),
            allocation_group: Default::default(),
            batch_cost,
            task_idx,
            flushed,
            workers,
            clip_context: ClipContext::new(),
            task_sender: None,
            coarse_task_receiver: None,
            strip_generator: StripGenerator::new(width, height, level),
            strip_storage: StripStorage::default(),
            level,
            alpha_storage,
            num_threads,
        };

        dispatcher.init();

        dispatcher
    }

    fn rasterize_f32(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, buffer, width, height, encoded_paints));
    }

    fn rasterize_u8(
        &self,
        buffer: &mut [u8],
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    ) {
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, buffer, width, height, encoded_paints));
    }

    fn init(&mut self) {
        let (render_task_sender, render_task_receiver) = crossbeam_channel::unbounded();
        let (coarse_task_sender, coarse_task_receiver) = ordered_channel::unbounded();
        let workers = self.workers.clone();
        let alpha_storage = self.alpha_storage.clone();

        self.task_sender = Some(render_task_sender);
        self.coarse_task_receiver = Some(coarse_task_receiver);

        // Spawn the loop for the worker threads.
        self.thread_pool.spawn_broadcast(move |_| {
            let render_task_receiver = render_task_receiver.clone();
            let mut coarse_task_sender = coarse_task_sender.clone();
            let worker = workers.get().unwrap();
            let mut worker = worker.borrow_mut();
            let thread_id = worker.thread_id();

            // Take out the allocation for alphas and store it in the worker.
            alpha_storage
                .with_inner(|alphas| worker.init(std::mem::take(&mut alphas[thread_id as usize])));

            while let Ok(task) = render_task_receiver.recv() {
                worker.run_render_task(task, &mut coarse_task_sender);
            }

            // If we reach this point, it means the `task_sender` has been dropped by the main thread
            // and no more tasks are available (since we flushed).
            // So we are done, and just need to place the alphas of the worker thread back into the
            // vector.

            alpha_storage.with_inner(|alphas| {
                alphas[thread_id as usize] = worker.finalize();
            });

            // Then, we drop the `coarse_task_sender`. Once all worker threads have
            // dropped their `coarse_task_sender`, the main thread knows that all workers are done
            // and all alphas have been placed, so it's safe to proceed.
            drop(coarse_task_sender);
        });
    }

    fn register_task(&mut self, task: RenderTaskType) {
        self.flushed = false;
        if self.task_sender.is_none() {
            self.init();
        }

        let cost = estimate_render_task_cost(&task, &self.allocation_group.path);
        self.allocation_group.render_tasks.push(task);
        self.batch_cost += cost;

        if self.batch_cost > COST_THRESHOLD {
            self.flush_tasks();
        }
    }

    fn flush_tasks(&mut self) {
        self.send_pending_tasks();

        self.batch_cost = 0.0;
    }

    fn bump_task_idx(&mut self) -> u32 {
        let idx = self.task_idx;
        self.task_idx += 1;
        idx
    }

    fn send_pending_tasks(&mut self) {
        let task_idx = self.bump_task_idx();
        let allocation_group =
            std::mem::replace(&mut self.allocation_group, self.allocations.get());
        let task_sender = self.task_sender.as_mut().unwrap();
        let clip_path = self.clip_context.get().map(|c| OwnedClip {
            strips: c.strips.into(),
            alphas: c.alphas.into(),
        });
        let task = RenderTask {
            idx: task_idx,
            clip_path,
            allocation_group,
        };
        task_sender.send(task).unwrap();
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
        let result_receiver = self.coarse_task_receiver.as_mut().unwrap();

        loop {
            match result_receiver.try_recv() {
                Ok(mut task) => {
                    let num_tasks = task.allocation_group.coarse_tasks.len();
                    for cmd in task.allocation_group.coarse_tasks.drain(0..num_tasks) {
                        match cmd {
                            CoarseTaskType::RenderPath {
                                strips: strip_range,
                                paint,
                                blend_mode,
                                thread_id,
                                mask,
                            } => self.wide.generate(
                                &task.allocation_group.strips
                                    [strip_range.start as usize..strip_range.end as usize],
                                paint.clone(),
                                blend_mode,
                                thread_id,
                                mask,
                            ),
                            CoarseTaskType::RenderWideCommand {
                                strips,
                                blend_mode,
                                paint,
                                thread_id,
                                mask,
                            } => self.wide.generate(
                                &strips,
                                paint.clone(),
                                blend_mode,
                                thread_id,
                                mask,
                            ),
                            CoarseTaskType::PushLayer {
                                thread_id,
                                clip_path,
                                blend_mode,
                                mask,
                                opacity,
                            } => {
                                let clip_path = clip_path.map(|strip_range| {
                                    &task.allocation_group.strips
                                        [strip_range.start as usize..strip_range.end as usize]
                                });

                                self.wide
                                    .push_layer(clip_path, blend_mode, mask, opacity, thread_id);
                            }
                            CoarseTaskType::PopLayer => self.wide.pop_layer(),
                        }
                    }

                    // Put the allocation group back so it can be reused in future iterations!
                    self.allocations.put(task.allocation_group);
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
        let alpha_slots = self.alpha_storage.take();

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
                        .map(|i| alpha_slots[i as usize].as_slice())
                        .unwrap_or(&[]);
                    fine.run_cmd(cmd, alphas, encoded_paints);
                }

                fine.pack(region);
            });
        });

        // Don't forget to put back the alpha buffers, so that they can be re-used in
        // the next path rendering iteration!
        self.alpha_storage.init(alpha_slots);
    }
}

impl Dispatcher for MultiThreadedDispatcher {
    fn wide(&self) -> &Wide {
        &self.wide
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let start = self.allocation_group.path.len() as u32;
        self.allocation_group.path.extend(path);
        let end = self.allocation_group.path.len() as u32;
        self.register_task(RenderTaskType::FillPath {
            path_range: start..end,
            transform,
            paint,
            fill_rule,
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let start = self.allocation_group.path.len() as u32;
        self.allocation_group.path.extend(path);
        let end = self.allocation_group.path.len() as u32;
        self.register_task(RenderTaskType::StrokePath {
            path_range: start..end,
            transform,
            paint,
            stroke: stroke.clone(),
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let mapped_clip = clip_path.map(|c| {
            let start = self.allocation_group.path.len() as u32;
            self.allocation_group.path.extend(c);
            let end = self.allocation_group.path.len() as u32;
            (start..end, clip_transform)
        });

        self.register_task(RenderTaskType::PushLayer {
            clip_path: mapped_clip,
            blend_mode,
            opacity,
            mask,
            fill_rule,
            aliasing_threshold,
        });
    }

    fn pop_layer(&mut self) {
        self.register_task(RenderTaskType::PopLayer);
    }

    fn reset(&mut self) {
        self.wide.reset();
        self.clip_context.reset();
        self.allocation_group.clear();
        self.batch_cost = 0.0;
        self.task_idx = 0;
        self.flushed = false;
        self.task_sender = None;
        self.coarse_task_receiver = None;
        self.strip_generator.reset();
        self.strip_storage.clear();
        self.alpha_storage.with_inner(|alphas| {
            for alpha in alphas {
                alpha.clear();
            }
        });

        let workers = self.workers.clone();
        // + 1 since we also wait on the main thread.
        let barrier = Arc::new(Barrier::new(usize::from(self.num_threads) + 1));
        let t_barrier = barrier.clone();

        self.thread_pool.spawn_broadcast(move |_| {
            let worker = workers.get().unwrap();
            let mut borrowed = worker.borrow_mut();
            borrowed.reset();
            t_barrier.wait();
        });

        barrier.wait();

        self.init();
    }

    fn flush(&mut self) {
        if self.flushed {
            return;
        }

        self.flush_tasks();
        let sender = core::mem::take(&mut self.task_sender);
        // Note that dropping the sender will signal to the workers that no more new paths
        // can arrive.
        drop(sender);
        self.run_coarse(false);

        self.alpha_storage.with_inner(|alphas| {
            // The main thread stores the alphas that are produced by playing a recording.
            // It is important we reserve the thread id 0 for this as the implementation for
            // `Recordable` uses this thread ID when generating the commands for coarse rasterization.
            alphas[0] = std::mem::take(&mut self.strip_storage.alphas);
        });

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
            RenderMode::OptimizeSpeed => self.rasterize_u8(buffer, width, height, encoded_paints),
            RenderMode::OptimizeQuality => {
                self.rasterize_f32(buffer, width, height, encoded_paints);
            }
        }
    }

    fn generate_wide_cmd(&mut self, strip_buf: &[Strip], paint: Paint, blend_mode: BlendMode) {
        // Note that we are essentially round-tripping here: The wide container is inside of the
        // main thread, but we first send a render task to a child thread which basically just
        // forwards it back to the main thread again. We cannot apply the wide command directly
        // here because there might be other paths that are currently being processed in a child
        // thread that should be rendered _before_ this wide command. Therefore, we treat it like
        // any other render task so that we can utilize the same mechanism of assigning a task ID
        // to ensure that they are executed in order.
        self.register_task(RenderTaskType::WideCommand {
            strip_buf: strip_buf.into(),
            // Recordings are currently always built on the main thread and thus have a `thread_idx`
            // of 0.
            thread_idx: 0,
            paint,
            blend_mode,
        });
    }

    fn strip_storage_mut(&mut self) -> &mut StripStorage {
        &mut self.strip_storage
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.flush_tasks();
        self.clip_context.push_clip(
            path,
            &mut self.strip_generator,
            fill_rule,
            transform,
            aliasing_threshold,
        );
    }

    fn pop_clip_path(&mut self) {
        self.flush_tasks();
        self.clip_context.pop_clip();
    }
}

impl Debug for MultiThreadedDispatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str("MultiThreadedDispatcher { .. }")
    }
}

#[derive(Debug)]
pub(crate) struct OwnedClip {
    strips: Box<[Strip]>,
    alphas: Box<[u8]>,
}

/// A structure that allows storing and fetching existing allocations.
struct AllocationManager<T> {
    entries: Vec<Vec<T>>,
}

impl<T> AllocationManager<T> {
    /// Get a new vector allocation.
    ///
    /// The vector is guaranteed to have been cleared before.
    fn get(&mut self) -> Vec<T> {
        self.entries.pop().unwrap_or_default()
    }

    /// Insert a new allocation in the store.
    fn put(&mut self, mut allocation: Vec<T>) {
        allocation.clear();
        self.entries.push(allocation);
    }
}

impl<T> Default for AllocationManager<T> {
    fn default() -> Self {
        Self { entries: vec![] }
    }
}

/// A structure to keep track of allocations that will be done while rendering with
/// multi-threading.
#[derive(Default)]
struct Allocations {
    /// The render tasks of a batch. They will be filled by the main thread as new fill/stroke commands
    /// come in and consumed by worker threads as they process them.
    render_tasks: AllocationManager<RenderTaskType>,
    /// The path store of a batch. It will be filled by the main thread as new commands come in
    /// and be used by the worker thread to generate the strips of a path.
    paths: AllocationManager<PathEl>,
    /// Stores allocations that are used by the worker thread to produce strips. They will be
    /// sent back to the main thread which then uses them for coarse rasterization.
    strips: AllocationManager<Strip>,
    /// The coarse tasks produced by a worker thread, which will be processed by the main thread.
    coarse_tasks: AllocationManager<CoarseTaskType>,
}

impl Allocations {
    /// Return a new allocation group.
    ///
    /// The group is guaranteed to have been cleared.
    fn get(&mut self) -> AllocationGroup {
        let render_tasks = self.render_tasks.get();
        let path = self.paths.get();
        let strips = self.strips.get();
        let coarse_tasks = self.coarse_tasks.get();

        AllocationGroup {
            path,
            render_tasks,
            coarse_tasks,
            strips,
        }
    }

    fn put(&mut self, allocation: AllocationGroup) {
        self.render_tasks.put(allocation.render_tasks);
        self.paths.put(allocation.path);
        self.strips.put(allocation.strips);
        self.coarse_tasks.put(allocation.coarse_tasks);
    }
}

#[derive(Default, Debug)]
pub(crate) struct AllocationGroup {
    pub(crate) path: Vec<PathEl>,
    pub(crate) render_tasks: Vec<RenderTaskType>,
    pub(crate) strips: Vec<Strip>,
    pub(crate) coarse_tasks: Vec<CoarseTaskType>,
}

impl AllocationGroup {
    fn clear(&mut self) {
        self.path.clear();
        self.render_tasks.clear();
        self.strips.clear();
        self.coarse_tasks.clear();
    }
}

#[derive(Debug)]
pub(crate) struct RenderTask {
    pub(crate) idx: u32,
    pub(crate) clip_path: Option<OwnedClip>,
    pub(crate) allocation_group: AllocationGroup,
}

#[derive(Debug, Clone)]
pub(crate) enum RenderTaskType {
    FillPath {
        path_range: Range<u32>,
        transform: Affine,
        paint: Paint,
        fill_rule: Fill,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    WideCommand {
        strip_buf: Box<[Strip]>,
        thread_idx: u8,
        paint: Paint,
        blend_mode: BlendMode,
    },
    StrokePath {
        path_range: Range<u32>,
        transform: Affine,
        paint: Paint,
        stroke: Stroke,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    PushLayer {
        clip_path: Option<(Range<u32>, Affine)>,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
    },
    PopLayer,
}

pub(crate) struct CoarseTask {
    allocation_group: AllocationGroup,
}

#[derive(Debug)]
pub(crate) enum CoarseTaskType {
    RenderPath {
        thread_id: u8,
        strips: Range<u32>,
        blend_mode: BlendMode,
        paint: Paint,
        mask: Option<Mask>,
    },
    RenderWideCommand {
        thread_id: u8,
        strips: Box<[Strip]>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    PushLayer {
        thread_id: u8,
        clip_path: Option<Range<u32>>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
    },
    PopLayer,
}

/// An object that might hold a certain value (behind a mutex), and panics if we attempt
/// to access it when it's not initialized.
#[derive(Clone)]
pub(crate) struct MaybePresent<T: Default> {
    present: Arc<AtomicBool>,
    value: Arc<Mutex<T>>,
}

impl<T: Default> MaybePresent<T> {
    pub(crate) fn new(val: T) -> Self {
        Self {
            present: Arc::new(AtomicBool::new(true)),
            value: Arc::new(Mutex::new(val)),
        }
    }

    pub(crate) fn init(&self, value: T) {
        let mut locked = self.value.lock().unwrap();
        *locked = value;
        self.present.store(true, Ordering::SeqCst);
    }

    pub(crate) fn with_inner(&self, mut func: impl FnMut(&mut T)) {
        assert!(
            self.present.load(Ordering::SeqCst),
            "Tried to access `MaybePresent` before initialization."
        );

        let mut lock = self.value.lock().unwrap();
        func(&mut lock);
    }

    pub(crate) fn take(&self) -> T {
        assert!(
            self.present.load(Ordering::SeqCst),
            "Tried to access `MaybePresent` before initialization."
        );

        let mut locked = self.value.lock().unwrap();
        self.present.store(false, Ordering::SeqCst);
        std::mem::take(&mut *locked)
    }
}

#[cfg(test)]
mod tests {
    use crate::Level;
    use crate::color::palette::css::BLUE;
    use crate::dispatch::Dispatcher;
    use crate::dispatch::multi_threaded::MultiThreadedDispatcher;
    use crate::kurbo::{Affine, Rect, Shape};
    use crate::peniko::{BlendMode, Fill};
    use vello_common::paint::{Paint, PremulColor};

    /// Ensure we don't cause a memory leak.
    #[test]
    fn allocations() {
        let mut dispatcher = MultiThreadedDispatcher::new(100, 100, 4, Level::new());
        for _ in 0..20 {
            dispatcher.fill_path(
                &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
                Fill::NonZero,
                Affine::IDENTITY,
                Paint::Solid(PremulColor::from_alpha_color(BLUE)),
                BlendMode::default(),
                None,
                None,
            );
            dispatcher.flush();
        }

        assert_eq!(dispatcher.allocations.paths.entries.len(), 1);
        assert_eq!(dispatcher.allocations.strips.entries.len(), 1);
        assert_eq!(dispatcher.allocations.render_tasks.entries.len(), 1);
        assert_eq!(dispatcher.allocations.coarse_tasks.entries.len(), 1);
    }
}
