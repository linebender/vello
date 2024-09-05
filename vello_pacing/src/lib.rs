use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
        Arc,
    },
    time::{Duration, Instant},
};

use wgpu::Buffer;

/// Docs used to type out how this is being reasoned about.
///
/// We manage six clocks:
/// 1) The CPU side clock. This is treated as an opaque nanosecond value, used only(?) for sleeping.
/// 2) The GPU "render complete clock". We wait for rendering work to complete to schedule the "present" of the *next* frame, to avoid stuttering.
///    On Android, we calibrate this to the "latching" clock using `presentMargin`.
///    Additionally, if a frame "failed", we choose not to present the next frame.
/// 3) The GPU render start clock, used to get a quick estimate for likely stutters.
/// 4) The GPU "latching" clock. This is only implied by `presentMargin` in relation to the render complete clock.
/// 5) The display "present clock". We assume that this is a potentially variable, but generally fixed amount of
///    time from the "latching" clock (due to compositing time).
/// 6) The CPU side present clock, which is the time we think presentation will happen.
///    These are always a fixed number of nanoseconds away from the "true presentation time" (this difference can be
///    calculated using <https://docs.rs/wgpu/latest/wgpu/struct.Adapter.html#method.get_presentation_timestamp>
///    and the true presentation time).
///
/// There are three cases for when we want to render:
/// 1) Active rendering
///      - Active Passive, e.g. playing an animation (/a video)
///      - Active Active, e.g. scrolling using a touch screen
/// - These have slightly different characteristics, because animation smoothness in in the passive case is more important.
///   Whereas latency reduction is more important in the active case.
/// 2) Rendering "once", i.e. in response to a keypress. Note that scrolling will likely be treated
///    as animation, and is an important case for latency minimisation.
/// 3) Stopping active rendering, i.e. the last frame.
///
/// Note also that we need to be aware of <https://github.com/linebender/vello/pull/606> in this design.
///
/// We observe that `earliest_present_time` can be earlier than `actual_present_time` because of earlier presents "getting in the way".
///
/// In the rendering once case, we just render as soon as possible.
/// It *could* be possible to coalesce these updates.
/// There are two phases to the multiple renders case:
/// 1) Up-to-date control loop
/// 2) Outdated control loop.
///
/// The outdated control loop case is due to the Presentation Time API having ~5 frames of latency.
/// (see <https://themaister.net/blog/2018/11/09/experimenting-with-vk_google_display_timing-taking-control-over-the-swap-chain/>,
/// the "Android 8.0, presentation timing latency" heading).
///
/// At the start of rendering, we are in the outdated control loop case, because of the above mentioned latency. That involves:
/// 1) We render the first frame immediately. This uses a best-effort estimated present time.
///    This does *not* have an requestedPresentationTime.
/// 2) We then set about rendering the second frame, which uses an estimated present time *one* interval
///    of display refresh rate after the first presentation. We start the rendering work for this frame
///    (tunable) ~20% sooner than one refresh rate interval after the start of rendering work for the first frame.
///    - If work has already finished on the first frame, then
///    - A potential option here is to cancel this render (GPU side) if the previous frame took (significantly) longer
///      than one refresh interval. It is future work to reason about this.
///      We foresee this as potentially advantageous because the CPU side work would be a relatively small part of
///      a frame, and so it would probably lead to "better" animation smoothness.
/// 3) Once the first frame finishes rendering, we present the second frame. We use "render end time" - "render start time" (`T_R`) to
///    estimate how many refresh durations (`N`) this render took, with a (tunable) ~25% "grace reduction" in the value.
///    This grace amount is to account for a likely ramping up of GPU power states, so future frames will probably be faster.
///    We also use `T_R` to estimate how many refresh durations the second frame will take to render (`N_2`), for scheduling the
///    present of the second frame. This does not use the grace reduction, but does have a grace subtraction of ~45% of the refresh cycle to
///    count as one frame. This grace period is to account for the fact that we don't know when in the refresh cycle we started rendering,
///    so there's a chance that even if both frames took longer than one refresh duration to render, we can "fake" smoothness.
///    Additionally, this slightly helps account for the GPU power state ramp-up.
/// 4) We present the second frame with a time which is "render end time" + previous "present clock - latching clock" + `N_2`  * refresh durations.
/// 5) We start rendering on the third frame, either `N_2 + N` or `2*N` (TBD) refresh durations after the first frame rendering started.
///    This will have an estimated present time of the same `N_2 + N` or `2*N` refresh durations from the estimated present time.
///
/// Using a statistical model for the variable time from starting rendering from event `1` to `2`.
///
/// TODO: Reason about multi-window/multi-display?
pub struct Thinking;

pub struct FrameRenderRequest {
    scene: vello::Scene,
    frame: FrameId,
    expected_present: u64,
    // In general, if touch is held, will need the next frame.
    needs_next_frame: bool, // TODO: No, LatencyOptimised, ConsistencyOptimised?
    present_immediately: bool,
    paint_start: Instant,
}

pub struct FrameStats {
    /// When the rendering work started on the GPU, in nanoseconds.
    ///
    /// Used to estimate how long frames are taking to render, to get
    /// faster feedback on whether we should request a slower (or faster) display mode.
    /// (We choose to be more conservative in requesting a faster display mode, but
    /// act quickly on requesting a slower one.)
    render_start: u64,
    /// When the rendering work finished on the GPU, in nanoseconds.
    ///
    /// Used to:
    /// - estimate the amount of time compositing takes
    /// - for estimating the expected presentation times before up-to-date timestamps become available
    render_end: u64,
    /// The time we told the application that this frame would be rendered at.
    ///
    /// In early frames, this was a best-effort guess, but we should be consistent in the use of this.
    ///
    /// For most users, the estimated present is only useful in terms of "how long between presents".
    /// However, an absolute timestamp is useful for calibrating Audio with Video
    estimated_present: u64,
    /// The time which we started the entire painting for this frame, including scene construction, etc.
    ///
    /// This is used for:
    /// 1) Providing the time at which rendering should start, generally
    ///    1 refresh duration later than the previous frame +- some margin.
    paint_start: Instant,
    // /// The time at which the frame pacing controller received the frame to be rendered.
    // ///
    // /// This is used to predict a possible frame deadline miss early
    // paint_end: Instant,
    /// The information we received from `SurfaceFlinger`, which is severely outdated by design.
    ///
    /// We might not get this information, so should be ready to work without it.
    presentation_time: Option<ash::vk::PastPresentationTimingGOOGLE>,
}

pub struct InFlightStatus {}

pub struct InFlightFrame {
    download_buffer: Buffer,
}

/// The state of the frame pacing controller thread.
pub struct VelloPacing {
    rx: Receiver<FrameRenderRequest>,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    /// Stats from previous frames, stored in a ring buffer (max capacity ~10?).
    stats: VecDeque<(FrameId, FrameStats)>,
    /// The refresh rate reported by the system.
    refresh_rate: u64,
    presenting_frame: InFlightFrame,
    gpu_working_frame: InFlightFrame,
}

/// A sketch of the expected API.
impl VelloPacing {
    pub fn new() -> Self {
        todo!()
    }

    pub fn launch(self) {
        std::thread::spawn(|| self.run());
    }

    /// Run a rendering task until presentation. Useful on macOS for resizing.
    pub fn present_synchronously(&mut self) {
        let token: () = self.present_immediately();
        self.wait_on_present(token);
    }

    pub fn present_immediately(&mut self) {}

    fn wait_on_present(&mut self, (): ()) {}

    pub fn stop(&mut self) {}

    fn run(mut self) {
        loop {
            let timeout = if self.frame_in_flight() {
                Duration::from_millis(4)
            } else {
                // If there is no frame in flight, then we can
                // keep ticking the device, but it isn't really needed
                Duration::from_millis(100)
            };
            match self.rx.recv_timeout(timeout) {
                Ok(frame_request) => {
                    self.paint_frame();
                    if frame_request.needs_next_frame {}
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    unreachable!("The main thread has stopped without telling rendering to stop")
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.poll_frame();
        }
    }

    fn paint_frame(&mut self) {
        // Prepare command buffers, etc.

        // If the previous frame returned
        self.poll_frame();
    }

    fn poll_frame(&mut self) {
        self.device.poll(wgpu::Maintain::Poll);
        if self.presented_frame_work_done() {
            if self.presented_frame_failed() {}
        }
    }

    fn presented_frame_work_done(&self) -> bool {
        false
    }

    fn presented_frame_failed(&self) -> bool {
        false
    }
    // fn presented_frame_status(&self) -> FrameStatus {}
    fn frame_in_flight(&self) -> bool {
        false
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FrameId(u32);

impl FrameId {
    pub fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

impl Default for VelloPacing {
    fn default() -> Self {
        Self::new()
    }
}
