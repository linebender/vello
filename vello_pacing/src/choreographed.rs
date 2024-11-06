#![allow(unused)]
#![warn(unused_variables)]

use std::{
    cell::RefCell,
    collections::VecDeque,
    mem::ManuallyDrop,
    ops::{Add, Mul},
    os::fd::{AsFd, AsRawFd},
    rc::{self, Rc},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    time::Duration,
};

use ndk::{
    choreographer::{self, Choreographer},
    looper::{self, FdEvent, ForeignLooper},
};
use nix::{
    sys::timerfd::{self, TimerFd},
    time::ClockId,
};
use vello::Scene;

/// A slightly tweaked version of the thinking, now that we have an understanding of `AChoreographer`.
///
/// Observations:
/// 1) A frame drop can happen because CPU or GPU work overruns.
/// 2) GPU work starts ~a fixed time after the end of the [`wgpu::Queue::submit`] call.
/// 3) We can predict a frame drop due to CPU time problems due to this case.
/// 4) Just dropping a frame is better than stuttering when we do so.
///
/// We need to think about three cases:
///
/// 1) The CPU-side work to prepare a scene is longer than one vsync. We ignore this case due to parallel command construction.
/// 2) The GPU-side work to prepare a frame is longer than one vsync.
/// 3) The total GPU+CPU side work is longer than one vsync (but individually each is shorter)
/// 4) Neither is the case.
///
/// For the first draft of this feature, we focus only on the third of these.
/// I.e. both the CPU and GPU side work of a frame are relatively cheap.
/// N.B. we do `Scene` preparation on the main thread, but it is included in
/// this CPU-side work.
///
/// The rendering loop goes as follows:
///
/// 1) We submit frame A.
/// 2) We start the CPU side work for frame B at the estimated start time.
/// 3) We race wait until the *deadline* for frame A with the CPU side work for frame B.
/// 4) If the deadline happened first (normal case, GPU work less than 1 frame), we compare
///    the timestamp of the end of the blit pass with the deadline.
///
/// For the moment, we ignore the possibility of a frame failing.
/// That doesn't actually change any behaviour here, because we render with [`Mailbox`](wgpu::PresentMode::Mailbox).
pub struct Thinking;

enum PacingCommand {}

/// The controller for a frame pacing thread.
pub struct PacingChannel {
    waker: ForeignLooper,
    channel: ManuallyDrop<mpsc::Sender<PacingCommand>>,
}

impl PacingChannel {
    fn send_command(&self, command: PacingCommand) {
        self.channel.send(command);
        // We add to the channel before waking, so that the event will be
        // received by the right wake.
        self.waker.wake();
    }
}

impl Drop for PacingChannel {
    fn drop(&mut self) {
        // Safety: We don't use `self.channel` after this line.
        // We drop the value before performing the wake, so that we the controller
        // thread knows the drop has happened.
        unsafe { ManuallyDrop::drop(&mut self.channel) };
        self.waker.wake();
        // TODO: Block on the thread itself finishing.
        // This makes using the Android NDK context in that thread *safer*.
    }
}

// We generally want to be thinking about two frames at a time, except for some statistical modelling.

/// A timestamp in `CLOCK_MONOTONIC`
///
/// For simplicity, all timestamps are treated as happening in the `CLOCK_MONOTONIC` timebase.
/// We'll validate that GPU performance counter timestamps meet this expectation as it becomes relevant.
///
/// This might not actually be true - the timebase of the return values from [`choreographer::ChoreographerFrameCallbackData`]
/// aren't documented by anything to be `CLOCK_MONOTONIC`, and I suspect we'll need to use [`ash::khr::calibrated_timestamps`] to get
/// the proper results.
// TODO: Does `Eq` make sense?
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
struct Timestamp(i64);

impl Add<Duration> for Timestamp {
    type Output = Timestamp;

    fn add(self, rhs: Duration) -> Self::Output {
        Self::from_nanos(
            self.as_nanos()
                .checked_add(rhs.as_nanos().try_into().unwrap())
                .unwrap(),
        )
    }
}

impl Mul<i64> for Timestamp {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Timestamp {
    /// Create a timespec for `nanos` nanoseconds since the origin of `CLOCK_MONOTIC`.
    const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// Create a timespec for `micros` microseconds since the origin of `CLOCK_MONOTIC`.
    const fn from_micros(micros: i64) -> Self {
        Self::from_nanos(micros * 1_000)
    }

    /// Create a timespec for `millis` milliseconds since the origin of `CLOCK_MONOTIC`.
    const fn from_millis(millis: i64) -> Self {
        Self::from_nanos(millis * 1_000_000)
    }

    /// Create a timestamp from a `Duration`, assuming that the duration is from
    /// `CLOCK_MONOTONIC`'s zero point.
    ///
    /// This matches the behaviour of values in [choreographer].
    fn from_ndk_crate(duration: Duration) -> Self {
        Self::from_nanos(duration.as_nanos().try_into().unwrap())
    }

    /// Get the number of nanoseconds since the origin of `CLOCK_MONOTIC` for this time.
    fn as_nanos(&self) -> i64 {
        self.0
    }

    /// Get the current time in `CLOCK_MONOTONIC`.
    fn now() -> Self {
        let spec = nix::time::clock_gettime(ClockId::CLOCK_MONOTONIC).unwrap();
        Self(spec.tv_sec() * 1_000_000_000 + spec.tv_nsec())
    }
}

/// A margin of latency which we *always* render against for safety.
const DEADLINE_MARGIN: Timestamp = Timestamp::from_millis(3);

/// A margin of latency before the deadline, which if we aren't before, we assume that the
/// frame probably missed the deadline.
///
/// In those cases, we bring future frames forward to try and avoid a cascading stutter
/// (and instead maintain only a dropped frame).
///
/// Note that this frame might still have technically counted as hitting the deadline.
/// However, we think a prolonged timing mismatch is worse than one dropped frame.
const DEADLINE_ASSUME_FAILED: Timestamp = Timestamp::from_micros(500);

/// The time within which we expect our estimations to be correct.
const EXPECTED_CONSISTENCY: Timestamp = Timestamp::from_millis(3);

struct OngoingFrame {
    /// The vertical blanking interval we're aiming to be presented at.
    ///
    /// We aim to finish GPU work [`DEADLINE_MARGIN`] before `deadline`.
    target_vsync: UpcomingVsync,

    /// The time at which we wanted to start this frame.
    ///
    /// `cpu_start_time` should try to be `requested_cpu_start_time - EPSILON`,
    /// but if this is far off, we know early that we might drop this frame (and so should request
    /// the next frame super early).
    /// If this is significantly off, then we will likely drop this frame to avoid stuttering.
    requested_cpu_start_time: Timestamp,

    /// The time at which `Scene` [rendering](`vello::Renderer::render_to_texture`) began.
    ///
    /// TODO: Does this include `Scene` construction time?
    cpu_start_time: Timestamp,

    /// The time at which [`wgpu::Queue::submit`] finished for this frame.
    ///
    /// If this is "much" later than expected (TODO: What would be expected?)
    /// then we will start the CPU side work for the next frame early.
    cpu_submit_time: Timestamp,

    /// The time at which work for this frame on the GPU started happening.
    ///
    /// Note: This assumes that
    gpu_start_time: Timestamp,
    /// The time at which work on the GPU finished.
    ///
    /// This should be before `target_deadline`.
    /// `gpu_finish_time` - `cpu_start_time` is used to estimate how long a total frame takes
    /// (and `gpu_finished_time` - `cpu_submit_time`) is used to estimate if a submission has
    /// missed a deadline.
    ///
    /// There is some really interesting trickery we can do here; the *next* frame
    /// on the GPU can definitely know this value, and can compare it against the deadline.
    /// If we know that the submitted frame will miss the deadline, then we can.
    gpu_finish_time: Timestamp,
}

/// The *shared* state of the pacing controller.
///
/// This is the state which *must* be available to callback based APIs.
struct VelloPacingController {
    // Resources used to implement the controller, which must be shared.
    choreographer: Choreographer,
    looper: looper::ThreadLooper,
    timer: TimerFd,

    // Communication primitives
    command_rx: Receiver<PacingCommand>,
    /// The duration of each frame, as reported by the system.
    ///
    /// In most cases, this can be inferred from the active frame timelines.
    /// However, for a short time after the refresh rate changes, `AChoreographer` is
    /// giving us incorrect future vsyncs.
    /// In particular, when we get a refresh rate callback, we know that frames after the
    /// next vsync will be this duration long.
    ///
    /// There are three cases where this can happen:
    /// 1) Where we ask for a slower framerate. We would do that when we are
    ///    consistently missing the target frame time.
    /// 2) When we ask for a faster framerate.
    /// 3) Where some other app (such as an overlay) asks for a faster framerate.
    ///    In that case, we need to catch up as quickly as possible.
    ///
    /// TODO: How does refresh rate interact with Poll and non-animated versions?
    ///
    /// Correct behaviour in this case is currently out-of-scope.
    upcoming_refresh_rate: Option<Duration>,
    /// The time at which the most recent vsync happened.
    latest_vsync_time: Option<Timestamp>,
    /// Upcoming vsyncs, (probably?) ordered by present time.
    upcoming_vsyncs: VecDeque<UpcomingVsync>,
}

struct UpcomingVsync {
    /// The [present time][choreographer::FrameTimeline::expected_presentation_time] of this vsync.
    ///
    /// That is, the time of the actual "flip".
    /// Once this time has passed, the vsync is historical.
    present_time: Timestamp,
    /// The [vsync id][choreographer::FrameTimeline::vsync_id] we're aiming for.
    vsync_id: i64,
    /// The [deadline][choreographer::FrameTimeline::deadline] which a frame needs to meet to be rendered at `present_time`.
    ///
    /// Once this time has passed, the vsync is largely academic, ad
    deadline: Timestamp,
}

enum GpuCommand {
    Render(Scene),
    Resize(u32, u32, Scene),
}

pub fn launch_pacing() -> PacingChannel {
    let (channel_tx, channel_rx) = std::sync::mpsc::sync_channel(0);
    // TODO: Give thread a name
    std::thread::spawn(|| {
        let looper = looper::ThreadLooper::prepare();
        let waker = looper.as_foreign().clone();
        let (command_tx, command_rx) = std::sync::mpsc::channel();
        channel_tx.send(PacingChannel {
            waker,
            channel: ManuallyDrop::new(command_tx),
        });
        drop(channel_tx);
        let choreographer = Choreographer::instance().expect("We just made the `Looper`");

        let timer = TimerFd::new(
            timerfd::ClockId::CLOCK_MONOTONIC,
            timerfd::TimerFlags::TFD_CLOEXEC,
        )
        // Something is much more badly wrong if this fails
        .expect("Could create a timer file descriptor");
        looper
            .as_foreign()
            .add_fd(timer.as_fd(), 0, FdEvent::INPUT, std::ptr::null_mut());

        let state = VelloPacingController {
            choreographer,
            looper: looper::ThreadLooper::for_thread().unwrap(),
            timer,
            command_rx,
            upcoming_refresh_rate: None,
            latest_vsync_time: None,
            upcoming_vsyncs: Default::default(),
        };
        /// We need to use a shared pacing controller here because of callback based APIs.
        let state = Rc::new(RefCell::new(state));
        {
            // Since our pointer will not be deallocated/freed unless unregister is called,
            // and we can't call unregister, we use a Weak.
            // Note that if `state` has been dropped, this thread will be ending,
            // so the looper will never be polled again.
            let callback_state = Rc::downgrade(&state);
            let state = state.borrow();
            state
                .choreographer
                .register_refresh_rate_callback(Box::new(move |rate| {
                    // Note: The refresh rate might be a multiple of some supported mode with interval
                    // higher than our.
                    if let Some(state) = callback_state.upgrade() {
                        let mut state = state.borrow_mut();
                        state.upcoming_refresh_rate = Some(rate);
                    } else {
                        tracing::error!(
                            "Kept getting refresh rate callbacks from Android despite thread ending."
                        );
                    }
                }));
        }
        {
            let callback_state = Rc::downgrade(&state);
            let state = state.borrow();
            fn vsync_callback(
                data: &choreographer::ChoreographerFrameCallbackData,
                callback_state: rc::Weak<RefCell<VelloPacingController>>,
            ) {
                if let Some(state) = callback_state.upgrade() {
                    let mut state = state.borrow_mut();
                    state
                        .choreographer
                        .post_vsync_callback(Box::new(move |data| {
                            vsync_callback(data, callback_state);
                        }));
                    let this_vsync_time = Timestamp::from_ndk_crate(data.frame_time());
                    state.clear_historical_vsyncs(this_vsync_time);
                    if let Some(_frame_time) = state.upcoming_refresh_rate {
                        // We might need special handling for upcoming refresh rates here.
                        // Primarily, checking if the refresh rate change has trickled into
                        // Choreographer yet.
                    }
                    // TODO: What significance (if any) does "preferred frame timeline" have?
                    for timeline in data.frame_timelines() {
                        let vsync_id = timeline.vsync_id();
                        // TODO: More efficient check here using ordering properties?
                        if state
                            .upcoming_vsyncs
                            .iter()
                            .any(|it| it.vsync_id == vsync_id)
                        {
                            continue;
                        }
                        let deadline = Timestamp::from_ndk_crate(timeline.deadline());
                        let present_time =
                            Timestamp::from_ndk_crate(timeline.expected_presentation_time());
                        state.upcoming_vsyncs.push_back(UpcomingVsync {
                            present_time,
                            vsync_id,
                            deadline,
                        });
                    }
                }
            }
            state
                .choreographer
                .post_vsync_callback(Box::new(move |data| vsync_callback(data, callback_state)));
        }
        let (gpu_tx, gpu_rx) = std::sync::mpsc::channel::<GpuCommand>();
        // TODO: Give thread a name
        std::thread::spawn(move || {
            // We perform all rendering work on this thread
            // Since submitting work and polling the GPU are mutually exclusive,
            // we do them on the same thread?
            loop {
                let command = gpu_rx.recv_timeout(Duration::from_millis(5));
            }
        });

        loop {
            // TODO: Ideally, we'd have the GPU polling happen through this looper.
            let poll = looper
                .poll_once()
                .expect("'Unrecoverable' error should not occur");
            // Outside of the looper polling operation, so no chance of overlap.
            let state = state.borrow_mut();
            match poll {
                // Fallthrough to checking the command channel
                looper::Poll::Wake | looper::Poll::Callback => {}
                looper::Poll::Timeout => {
                    unreachable!("Timeout reached, but we didn't set a timeout")
                }
                looper::Poll::Event {
                    ident,
                    fd,
                    events,
                    data: _,
                } => {
                    if fd.as_raw_fd() == state.timer.as_fd().as_raw_fd() {
                        // TODO: A Hangup might be expected for a timer?
                        if events.contains(FdEvent::ERROR | FdEvent::INVALID | FdEvent::HANGUP) {
                            panic!("Got an error from the timer file descriptor {events:?}")
                        }
                        assert!(ident == 0);
                        // Clear out the existing timer value, so that we won't immediately retrigger.
                        state.timer.wait().unwrap();
                        // We should now do whatever we set the timer for.
                        // Presumably, that is start rendering?
                    }
                }
            }

            match state.command_rx.try_recv() {
                Ok(command) => {
                    // Action `command`
                    match command {}
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Continue
                }
            }
        }
    });
    channel_rx
        .recv_timeout(Duration::from_millis(100))
        .expect("Could not create pacing controller")
}

impl VelloPacingController {
    /// Clear the historical vsyncs which are definitely no longer relevant, given a
    /// time that a vsync recently occurred.
    fn clear_historical_vsyncs(&mut self, recent_vsync: Timestamp) {
        // We assume FPS to be less than 1000; this gives flexibility here.
        let vsync_cutoff = recent_vsync + Duration::from_millis(1);
        while let Some(front) = self.upcoming_vsyncs.front() {
            if front.present_time < vsync_cutoff {
                self.upcoming_vsyncs.pop_front();
            } else {
                break;
            }
        }
    }
}
