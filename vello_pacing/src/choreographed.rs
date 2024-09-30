use std::ops::Mul;

use nix::time::ClockId;

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

// We generally want to be thinking about two frames at a time, except for some statistical modelling.

/// A timestamp in `CLOCK_MONOTONIC`
///
/// For simplicity, all timestamps are treated as happening in the `CLOCK_MONOTONIC` timebase.
/// We'll validate that GPU performance counter timestamps meet this expectation as it becomes relevant.
///
/// This might not actually be true - the timebase of the return values from [`ndk::choreographer::ChoreographerFrameCallbackData`]
/// aren't documented by
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
    ///
    /// TODO: This assumed the returned value is not negative.
    /// Hopefully that's fine?
    fn now() -> Self {
        let spec = nix::time::clock_gettime(ClockId::CLOCK_MONOTONIC).unwrap();
        Self(spec.tv_sec() * 1_000_000_000 + spec.tv_nsec())
    }
}

/// A margin of latency which we *always* render against for safety.
const DEADLINE_MARGIN: Timestamp = Timestamp::from_millis(2);

/// A margin of latency before the deadline, which if we aren't before, we assume that the
/// frame probably missed the deadline.
///
/// In those cases, we bring future frames forward.
const DEADLINE_ASSUME_FAILED: Timestamp = Timestamp::from_micros(500);

struct OngoingFrame {
    /// The [present time][ndk::choreographer::ChoreographerFrameCallbackData::frame_timeline_expected_presentation_time].
    target_present_time: Timestamp,
    /// The [vsync id][ndk::choreographer::ChoreographerFrameCallbackData::frame_timeline_vsync_id] we're aiming for.
    target_vsync_id: i64,
    /// The deadline which this frame needs to meet to be rendered at `target_present_time`.
    ///
    /// We aim for a time [`DEADLINE_MARGIN`] before the deadline.
    target_deadline: Timestamp,

    /// The time at which we wanted to start this frame.
    ///
    /// `start_time` should try to be `requested_start_time - EPSILON`,
    /// but if this is far off, we know early that we might drop this frame (and so should request
    /// the next frame super early).
    /// If this is significantly off, then we will likely drop this frame to avoid stuttering.
    requested_start_time: Timestamp,

    /// The time at which `Scene` [rendering](`vello::Renderer::render_to_texture`) began.
    ///
    /// TODO: Does this include `Scene` construction time?
    start_time: Timestamp,

    /// The time at which [`wgpu::Queue::submit`] finished for this frame.
    submit_time: Timestamp,
}
