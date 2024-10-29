#![allow(unused)]
#![warn(unused_variables)]

use std::{
    cell::RefCell,
    mem::ManuallyDrop,
    ops::Mul,
    rc::Rc,
    sync::mpsc::{self, Receiver},
    time::Duration,
};

use ndk::{
    choreographer::Choreographer,
    looper::{self, ForeignLooper},
};
use nix::time::ClockId;
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
/// That doesn't actually change any behaviour here, because we render with `MailBox`.
pub struct Thinking;

enum PacingCommand {}

pub struct PacingChannel {
    waker: ForeignLooper,
    channel: ManuallyDrop<mpsc::Sender<PacingCommand>>,
}

impl PacingChannel {
    fn send_command(&self, command: PacingCommand) {
        self.channel.send(command);
        // We add to the channel before waking, so that the event will be received by the right wake.
        self.waker.wake();
    }
}

impl Drop for PacingChannel {
    fn drop(&mut self) {
        // Safety: We don't use `self.channel` after this line.
        // We drop the value before performing the wake, so that we instantly know that the drop has happened.
        unsafe { ManuallyDrop::drop(&mut self.channel) };
        self.waker.wake();
    }
}

// We generally want to be thinking about two frames at a time, except for some statistical modelling.

/// A timestamp in `CLOCK_MONOTONIC`
///
/// For simplicity, all timestamps are treated as happening in the `CLOCK_MONOTONIC` timebase.
/// We'll validate that GPU performance counter timestamps meet this expectation as it becomes relevant.
///
/// This might not actually be true - the timebase of the return values from [`ndk::choreographer::ChoreographerFrameCallbackData`]
/// aren't documented by anything to be `CLOCK_MONOTONIC`, and I suspect we'll need to use [`ash::khr::calibrated_timestamps`] to get
/// the proper results.
struct Timestamp(i64);

impl Mul<i64> for Timestamp {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Timestamp {
    const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    const fn from_micros(micros: i64) -> Self {
        Self::from_nanos(micros * 1_000)
    }

    const fn from_millis(millis: i64) -> Self {
        Self::from_nanos(millis * 1_000_000)
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
    /// The [present time][ndk::choreographer::ChoreographerFrameCallbackData::frame_timeline_expected_presentation_time] we
    /// expect this frame to be displayed at.
    target_present_time: Timestamp,
    /// The [vsync id][ndk::choreographer::ChoreographerFrameCallbackData::frame_timeline_vsync_id] we're aiming for.
    target_vsync_id: i64,
    /// The [deadline][ndk::choreographer::ChoreographerFrameCallbackData::frame_timeline_deadline] which this frame needs to meet to be rendered at `target_present_time`.
    ///
    /// We aim for a time [`DEADLINE_MARGIN`] before the deadline.
    target_deadline: Timestamp,

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
    /// If this is "much" later than
    cpu_submit_time: Timestamp,

    /// The time at which work on the GPU started.
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

struct VelloPacingController {
    choreographer: Choreographer,
    command_rx: Receiver<PacingCommand>,
    /// The duration of each frame, as reported by the system.
    ///
    /// For a short time, we don't have the refresh rate.
    ///
    /// This is used to detect the case where `AChoreographer` is giving us incorrect future vsyncs.
    refresh_rate: Option<Duration>,
    looper: looper::ThreadLooper,
}

/// We need to use a shared
type SharedPacing = Rc<RefCell<VelloPacingController>>;

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

        let state = VelloPacingController {
            choreographer,
            command_rx,
            refresh_rate: None,
            looper: looper::ThreadLooper::for_thread().unwrap(),
        };
        let state = Rc::new(RefCell::new(state));
        {
            let callback_state = Rc::clone(&state);
            let state = state.borrow();
            state
                .choreographer
                .register_refresh_rate_callback(Box::new(move |rate| {
                    let mut state = callback_state.borrow_mut();
                    state.refresh_rate = Some(rate);
                }));
        }
        let (gpu_tx, gpu_rx) = std::sync::mpsc::channel::<GpuCommand>();
        // TODO: Give thread a name
        std::thread::spawn(|| {
            // We perform all GPU work on this thread
            // Since submitting work and polling the GPU are mutually exclusive, we
        });
        loop {
            let poll = looper.poll_once().expect("'Unrecoverable' error");
            assert!(
                !matches!(poll, looper::Poll::Timeout | looper::Poll::Event { .. }),
                "Impossible poll results from our use of Looper APIs."
            );
            let state = state.borrow_mut();
            match state.command_rx.try_recv() {
                Ok(command) => {
                    // Action `command`
                }
                Err(mpsc::TryRecvError::Disconnected) => {}
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    });
    channel_rx
        .recv_timeout(Duration::from_millis(100))
        .expect("Could not create pacing controller")
}
