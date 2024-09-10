use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender},
        Arc,
    },
    time::{Duration, Instant},
};

use vello::Scene;
use wgpu::{Buffer, SubmissionIndex, SurfaceTexture};

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
    /// Whether the source paint request wanted there to be a next frame.
    next_frame_continued: bool,
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
    download_map_buffer: Buffer,
    work_complete: Arc<AtomicBool>,
    paint_start: Instant,
    id: FrameId,
    submission_index: SubmissionIndex,
    /// Information needed to perform a presentation, and not before.
    required_to_present: Option<(SurfaceTexture, Scene)>,
    estimated_present: u64,
    next_frame_expected: bool,
}

pub enum VelloControl {
    Frame(FrameRenderRequest),
    Stop,
    /// A resize request. This might skip rendering a previous frame
    /// if it arrives before that frame is presented.
    Resize(FrameRenderRequest, (u32, u32), Sender<FrameId>),
}

/// The state of the frame pacing controller thread.
pub struct VelloPacing {
    rx: Receiver<VelloControl>,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    google_display_timing_ext_devices: Vec<Option<ash::google::display_timing::Device>>,
    adapter: Arc<wgpu::Adapter>,
    surface: wgpu::Surface<'static>,
    /// Stats from previous frames, stored in a ring buffer (max capacity ~10?).
    stats: VecDeque<(FrameId, FrameStats)>,
    /// The refresh rate reported "by the system".
    refresh_rate: u64,
    mapped_unused_download_buffers: Vec<(Buffer, Arc<AtomicBool>)>,
    mapped_unused_download_buffers_scratch: Vec<(Buffer, Arc<AtomicBool>)>,
    // TODO: Smallvec?
    free_download_buffers: Vec<(Buffer, Arc<AtomicBool>)>,
    /// Details about the previous frame, which has already been presented.
    presenting_frame: Option<InFlightFrame>,
    /// Details about the frame whose work has been submitted to the.
    gpu_working_frame: Option<InFlightFrame>,
}

/// A sketch of the expected API.
impl VelloPacing {
    pub fn new() -> Self {
        todo!()
    }

    pub fn launch(self) {
        std::thread::spawn(move || self.run());
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
                Ok(command) => {
                    match command {
                        VelloControl::Frame(request) => {
                            self.poll_frame();
                            // TODO: Error handling
                            let texture = self.surface.get_current_texture().unwrap();
                            self.paint_frame(request.scene, texture);
                            self.poll_frame();
                        }
                        VelloControl::Stop => {
                            // TODO:
                            if let Some(mut old_frame) = self.gpu_working_frame.take() {
                                // This frame will never be presented
                                drop(old_frame.required_to_present.take());
                            }
                            // self.device.poll(wgpu::MaintainBase::Wait);
                            // What do we need to be careful about dropping?
                            // Do we need to run the GPU
                            break;
                        }
                        #[expect(
                            unreachable_code,
                            unused_variables,
                            reason = "We stub out the unused variables"
                        )]
                        VelloControl::Resize(request, (_, _), done) => {
                            // Cancel the frame which hasn't been scheduled for presentation.
                            if let Some(mut old_frame) = self.gpu_working_frame.take() {
                                // This frame will never be presented
                                drop(old_frame.required_to_present.take());
                                self.abandon(old_frame);
                            }
                            // Make sure any easily detectable needed reallocation happens
                            // TODO: What do we want to do if:
                            // 1) The previous frame didn't succeed
                            // 2) We are trying to resize
                            // We choose not to address this presently, because it is a.
                            self.poll_frame();
                            // self.surface
                            //     .configure(&self.device, SurfaceConfiguration { width, height });
                            unimplemented!();

                            let texture = self.surface.get_current_texture().unwrap();
                            let frame = self.paint_frame_inner(&request.scene, &texture);
                            texture.present();
                            if let Some(old_presenting) = self.presenting_frame.take() {
                                self.abandon(old_presenting);
                            }
                            // TODO: Maybe: self.abandoned_frames.extend(self.presenting_frame.take());
                            self.presenting_frame = Some(frame);
                            if let Err(e) = done.send(request.frame) {
                                tracing::error!("Failed to send present result {e}");
                            };
                        }
                    }

                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // TODO: Is this just the implicit stop command?
                    tracing::error!(
                        "The main thread has stopped without telling rendering to stop"
                    );
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Fall through intentionally
                }
            }
            self.poll_frame();
        }
    }

    fn abandon(&mut self, frame: InFlightFrame) {
        // This is accurate since we never make a `Weak` for the `work_complete` buffers
        // If that became untrue, the only risk is that the buffer and value would be dropped instead of reused.
        if Arc::strong_count(&frame.work_complete) == 1 {
            Self::handle_completed_mapping_finished(
                &mut self.free_download_buffers,
                frame.download_map_buffer,
                frame.work_complete,
            );
        } else {
            self.mapped_unused_download_buffers
                .push((frame.download_map_buffer, frame.work_complete));
        }
    }

    fn paint_frame(&mut self, scene: Scene, to: SurfaceTexture) {
        // TODO: It could happen that frames get sent not in response to our handling code (e.g. as an immediate frame)
        // Do we just always want to abandon the current working frame?
        assert!(self.gpu_working_frame.is_none());
        let mut res = self.paint_frame_inner(&scene, &to);
        res.required_to_present = Some((to, scene));
        self.gpu_working_frame = Some(res);
    }

    #[must_use]
    #[expect(unused_variables, reason = "Not yet implemented")]
    fn paint_frame_inner(&mut self, scene: &Scene, to: &SurfaceTexture) -> InFlightFrame {
        // Prepare command buffers, etc.

        todo!();
    }

    fn poll_frame(&mut self) {
        self.device.poll(wgpu::Maintain::Poll);
        let mut failed = false;
        let mut desired_present_time: Option<u64> = None;
        if let Some(presenting) = &mut self.presenting_frame {
            if let Some(value) = Arc::get_mut(&mut presenting.work_complete) {
                // Reset the value to false, because we're about to reuse it.
                if std::mem::take(value.get_mut()) {
                    let presenting = self
                        .presenting_frame
                        .take()
                        .expect("We know this value is present");

                    {
                        let map_result =
                            presenting.download_map_buffer.slice(..).get_mapped_range();
                        eprintln!("Downloaded: {map_result:?}");
                        // Will be calculated from `map_result`, as part of https://github.com/linebender/vello/pull/606
                        failed = false;
                        if failed {
                            // Perform reallocation. Will be part of https://github.com/linebender/vello/pull/606.
                        }
                        // The time which we should not present before.
                        // If `map_result` indicates we really badly overflowed the available time,
                        // then this will be later than otherwise expected, to avoid a stutter.
                        // See https://developer.android.com/games/sdk/frame-pacing
                        desired_present_time = None;
                        self.stats.push_back((
                            presenting.id,
                            FrameStats {
                                // TODO: Make optional because some backends don't support timestamp queries?
                                render_start: 0,
                                render_end: 0,
                                estimated_present: presenting.estimated_present,
                                paint_start: presenting.paint_start,
                                presentation_time: None,
                                next_frame_continued: presenting.next_frame_expected,
                            },
                        ));
                    }
                    Self::handle_completed_mapping_finished(
                        &mut self.free_download_buffers,
                        presenting.download_map_buffer,
                        presenting.work_complete,
                    );
                } else {
                    unreachable!(
                        "Buffer mapping/work complete callback dropped without being called."
                    )
                }
            } else {
                // The presenting frame's work isn't complete. Probably nothing to do?
            }
        }
        if self.presenting_frame.is_none() {
            if let Some(mut working_frame) = self.gpu_working_frame.take() {
                let (texture, scene) = working_frame.required_to_present.take().unwrap();
                if failed {
                    // Then redo the paint; we know the previous attempted frame will have been cancalled, so won't be expensive.
                    // We choose to present here immediately, to minimise likely latency costs.
                    // We know that the.
                    let new_inner = self.paint_frame_inner(&scene, &texture);
                    let old_working = std::mem::replace(&mut working_frame, new_inner);
                    self.abandon(old_working);
                } else if working_frame.work_complete.load(Ordering::Relaxed) {
                    // I don't think there's actually anything interesting to do here, but might need to be reasoned about.
                };
                // We run a display timing request after the present occurs, but we only need to get the Vulkan swapchain once.
                let mut swc = None;
                if let Some(desired_present_time) =
                    // Pseudocode for the or-else case. This is to handle the scenario where the previous frame finished
                    // *before* we.
                    desired_present_time
                        .or_else(|| Some(self.stats.front()?.1.estimated_present))
                {
                    swc = unsafe {
                        // SAFETY: We do not "manually drop" the raw handle, i.e. call "vkDestroySurfaceKHR"
                        self.surface
                            .as_hal::<wgpu::hal::vulkan::Api, _, _>(|surface| {
                                if let Some(surface) = surface {
                                    surface.set_next_present_time(ash::vk::PresentTimeGOOGLE {
                                        desired_present_time,
                                        present_id: working_frame.id.0,
                                    });
                                    Some(surface.raw_swapchain())
                                } else {
                                    None
                                }
                            })
                            .flatten()
                            .flatten()
                    };
                }
                texture.present();
                if let Some(_swc) = swc {
                    // Load past present timing information
                }
                if working_frame.next_frame_expected {
                    // Do the maths for when we should ask the main thread for the next frame
                    // TODO: Do we need our own timer thread for just that, or should something else happen?
                }
                self.presenting_frame = Some(working_frame);
            }
        }
        assert!(self.mapped_unused_download_buffers_scratch.is_empty());
        for (buffer, work_complete) in self.mapped_unused_download_buffers.drain(..) {
            // We know that the `work_complete` will not change again.
            if Arc::strong_count(&work_complete) == 1 {
                Self::handle_completed_mapping_finished(
                    &mut self.free_download_buffers,
                    buffer,
                    work_complete,
                );
            } else {
                self.mapped_unused_download_buffers_scratch
                    .push((buffer, work_complete));
            }
        }
        std::mem::swap(
            &mut self.mapped_unused_download_buffers_scratch,
            &mut self.mapped_unused_download_buffers,
        );
    }

    fn frame_in_flight(&self) -> bool {
        self.gpu_working_frame.is_some()
    }

    /// Handle a completed buffer mapping for reuse.
    /// `buffer` should be finished with, but mapped.
    ///
    /// Associated function for borrow checker purposes
    fn handle_completed_mapping_finished(
        unmapped_download_buffers: &mut Vec<(Buffer, Arc<AtomicBool>)>,
        buffer: Buffer,
        mut work_complete: Arc<AtomicBool>,
    ) {
        if let Some(value) = Arc::get_mut(&mut work_complete) {
            let value = value.get_mut();
            if !*value {
                tracing::error!("Tried to unmap buffer which was never assigned for mapping?");
            } else {
                buffer.unmap();
            }
            *value = false;
            if unmapped_download_buffers.len() < 4 {
                unmapped_download_buffers.push((buffer, work_complete));
            }
        }
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
