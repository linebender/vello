// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared timing primitives and state for WebGL renderer initialization and probes.

use core::time::Duration;

#[derive(Debug, Clone, Copy)]
struct Timestamp(f64);

impl Timestamp {
    fn now() -> Self {
        Self(web_sys::window().unwrap().performance().unwrap().now())
    }

    fn elapsed(self) -> Duration {
        self.elapsed_until(Self::now())
    }

    fn elapsed_until(self, end: Self) -> Duration {
        Duration::from_secs_f64((end.0 - self.0) / 1_000.0)
    }
}

#[derive(Debug, Default)]
struct PhaseTimer(Duration);

impl PhaseTimer {
    fn measure<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        self.measure_from(Timestamp::now(), operation)
    }

    fn measure_from<T>(&mut self, started_at: Timestamp, operation: impl FnOnce() -> T) -> T {
        self.measure_with_end_from(started_at, operation).0
    }

    #[cfg(feature = "probe")]
    fn measure_with_end<T>(&mut self, operation: impl FnOnce() -> T) -> (T, Timestamp) {
        self.measure_with_end_from(Timestamp::now(), operation)
    }

    fn measure_with_end_from<T>(
        &mut self,
        started_at: Timestamp,
        operation: impl FnOnce() -> T,
    ) -> (T, Timestamp) {
        let result = operation();
        let completed_at = Timestamp::now();
        self.0 += started_at.elapsed_until(completed_at);
        (result, completed_at)
    }

    fn duration(&self) -> Duration {
        self.0
    }
}

#[derive(Debug)]
struct PendingTimer {
    started_at: Timestamp,
    poll_count: u32,
}

impl PendingTimer {
    fn start() -> Self {
        Self::start_at(Timestamp::now())
    }

    fn start_at(started_at: Timestamp) -> Self {
        Self {
            started_at,
            poll_count: 0,
        }
    }

    fn record_poll(&mut self) {
        self.poll_count = self.poll_count.saturating_add(1);
    }

    fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Time spent constructing a WebGL renderer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebGlRendererInitTimings {
    /// CPU time spent configuring and obtaining the WebGL2 context and validating its capabilities.
    pub context_creation: Duration,
    /// CPU time spent detecting parallel shader compilation support and issuing shader compilation
    /// and program linking calls.
    pub shader_submission: Duration,
    /// CPU time spent creating persistent renderer caches and WebGL textures, buffers,
    /// framebuffers, and vertex array objects.
    pub resource_initialization: Duration,
    /// Elapsed time from completing synchronous setup until shader compilation and linking were
    /// first observed as complete.
    ///
    /// This includes delays between asynchronous polls. When
    /// [`WebGlRendererInit::finish`](crate::WebGlRendererInit::finish) is
    /// called before shader completion has been observed, this is zero and any blocking wait is
    /// included in [`Self::finalization`] instead.
    pub shader_completion: Duration,
    /// CPU time spent validating linked programs, resolving their interfaces, and constructing the
    /// final renderer.
    ///
    /// This can include time blocked on shader compilation and linking when initialization is
    /// finished synchronously.
    pub finalization: Duration,
    /// Elapsed time from starting
    /// [`WebGlRenderer::begin_with`](crate::WebGlRenderer::begin_with) until renderer construction
    /// completed.
    ///
    /// When initialization is polled asynchronously, this includes delays between polls.
    pub total: Duration,
}

#[derive(Debug)]
pub(super) struct RendererInitTiming {
    initialization_started_at: Timestamp,
    context_creation: Duration,
    shader_submission: PhaseTimer,
    resource_initialization: PhaseTimer,
    initial_resource_initialization_started_at: Option<Timestamp>,
}

impl RendererInitTiming {
    pub(super) fn start() -> Self {
        Self {
            initialization_started_at: Timestamp::now(),
            context_creation: Duration::ZERO,
            shader_submission: PhaseTimer::default(),
            resource_initialization: PhaseTimer::default(),
            initial_resource_initialization_started_at: None,
        }
    }

    pub(super) fn context_created(&mut self) {
        let context_created_at = Timestamp::now();
        self.context_creation = self
            .initialization_started_at
            .elapsed_until(context_created_at);
        self.initial_resource_initialization_started_at = Some(context_created_at);
    }

    pub(super) fn measure_shader_submission<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        self.shader_submission.measure(operation)
    }

    pub(super) fn measure_resource_initialization<T>(
        &mut self,
        operation: impl FnOnce() -> T,
    ) -> T {
        if let Some(started_at) = self.initial_resource_initialization_started_at.take() {
            self.resource_initialization
                .measure_from(started_at, operation)
        } else {
            self.resource_initialization.measure(operation)
        }
    }

    pub(super) fn setup_complete(self) -> PendingRendererInitTiming {
        PendingRendererInitTiming {
            initialization_started_at: self.initialization_started_at,
            context_creation: self.context_creation,
            shader_submission: self.shader_submission.duration(),
            resource_initialization: self.resource_initialization.duration(),
            shader_completion: PendingTimer::start(),
            shader_completion_duration: None,
        }
    }
}

#[derive(Debug)]
pub(super) struct PendingRendererInitTiming {
    initialization_started_at: Timestamp,
    context_creation: Duration,
    shader_submission: Duration,
    resource_initialization: Duration,
    shader_completion: PendingTimer,
    shader_completion_duration: Option<Duration>,
}

impl PendingRendererInitTiming {
    pub(super) fn record_poll(&mut self) {
        self.shader_completion.record_poll();
    }

    pub(super) fn shader_completion_observed(&mut self) {
        self.shader_completion_duration = Some(self.shader_completion.elapsed());
    }

    pub(super) fn begin_finalization(self) -> RendererFinalizationTiming {
        RendererFinalizationTiming {
            timing: self,
            finalization_started_at: Timestamp::now(),
        }
    }
}

pub(super) struct RendererFinalizationTiming {
    timing: PendingRendererInitTiming,
    finalization_started_at: Timestamp,
}

impl RendererFinalizationTiming {
    pub(super) fn finish(self) -> WebGlRendererInitTimings {
        let initialization_completed_at = Timestamp::now();
        WebGlRendererInitTimings {
            context_creation: self.timing.context_creation,
            shader_submission: self.timing.shader_submission,
            resource_initialization: self.timing.resource_initialization,
            shader_completion: self
                .timing
                .shader_completion_duration
                .unwrap_or(Duration::ZERO),
            finalization: self
                .finalization_started_at
                .elapsed_until(initialization_completed_at),
            total: self
                .timing
                .initialization_started_at
                .elapsed_until(initialization_completed_at),
        }
    }
}

/// Time spent in each phase of a WebGL probe.
#[cfg(feature = "probe")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGlProbeTimings {
    /// CPU time spent allocating resources, uploading the probe image, and building the scene.
    pub setup: Duration,
    /// CPU time spent executing the probe's internal `render_scene` call.
    pub render_submission: Duration,
    /// CPU time spent allocating the pixel-pack buffer, queuing the pixel readback, creating its
    /// fence, and flushing WebGL commands.
    pub readback_submission: Duration,
    /// Time from submitting the pixel readback until its fence was first observed as signaled.
    ///
    /// This includes GPU rendering and readback as well as delays between polls.
    pub completion_latency: Duration,
    /// CPU time spent reading the completed pixel buffer back, copying it into WASM memory, and
    /// flipping it vertically.
    pub readback: Duration,
    /// Wall-clock time from starting the probe until its result was produced.
    pub total: Duration,
}

#[cfg(feature = "probe")]
#[derive(Debug)]
pub(super) struct ProbeTiming {
    probe_started_at: Timestamp,
    setup_started_at: Timestamp,
    setup: Duration,
    render_submission: PhaseTimer,
    readback_submission: PhaseTimer,
    completion: Option<PendingTimer>,
    completion_latency: Option<Duration>,
    readback: PhaseTimer,
}

#[cfg(feature = "probe")]
impl ProbeTiming {
    pub(super) fn start() -> Self {
        let probe_started_at = Timestamp::now();
        Self {
            probe_started_at,
            setup_started_at: probe_started_at,
            setup: Duration::ZERO,
            render_submission: PhaseTimer::default(),
            readback_submission: PhaseTimer::default(),
            completion: None,
            completion_latency: None,
            readback: PhaseTimer::default(),
        }
    }

    pub(super) fn setup_complete(&mut self) {
        self.setup = self.setup_started_at.elapsed();
    }

    pub(super) fn measure_render_submission<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        self.render_submission.measure(operation)
    }

    pub(super) fn measure_readback_submission<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        let (result, submitted_at) = self.readback_submission.measure_with_end(operation);
        self.completion = Some(PendingTimer::start_at(submitted_at));
        result
    }

    pub(super) fn record_poll(&mut self) {
        self.completion_mut().record_poll();
    }

    pub(super) fn completion_observed(&mut self) {
        self.completion_latency = Some(self.completion_mut().elapsed());
    }

    pub(super) fn measure_readback<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        self.readback.measure(operation)
    }

    pub(super) fn finish(self) -> (WebGlProbeTimings, u32) {
        let probe_completed_at = Timestamp::now();
        let completion = self
            .completion
            .expect("pixel readback must be submitted before completing the probe");
        (
            WebGlProbeTimings {
                setup: self.setup,
                render_submission: self.render_submission.duration(),
                readback_submission: self.readback_submission.duration(),
                completion_latency: self
                    .completion_latency
                    .expect("pixel readback completion must be observed before finishing"),
                readback: self.readback.duration(),
                total: self.probe_started_at.elapsed_until(probe_completed_at),
            },
            completion.poll_count,
        )
    }

    fn completion_mut(&mut self) -> &mut PendingTimer {
        self.completion
            .as_mut()
            .expect("pixel readback must be submitted before polling the probe")
    }
}
