/// A slightly tweaked version of the thinking, now that we have an understanding of `AChoreographer`.
///
/// We need to think about three cases:
///
/// 1) The CPU-side work to prepare a scene is longer than one vsync.
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
/// 1) We submit frame a
/// 2) We start the CPU side work for frame b at the estimated start time.
/// 3) We race wait until the *deadline* for frame a with the CPU side work for frame b.
/// 4) If the deadline happened first (normal case, GPU work less than 1 frame), we compare
///    the timestamp of the end of the blit pass with the deadline.
/// 5)
pub struct Thinking;
