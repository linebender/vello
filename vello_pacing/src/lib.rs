pub struct VelloPacing {}

/// Docs used to type out how this is being reasoned about.
///
/// We manage six clocks:
/// 1) The CPU side clock. This is treated as an opaque nanosecond value, used only for sleeping.
/// 2) The GPU "render complete clock". We wait for rendering work to complete to schedule the "present" of the *next* frame, to avoid stuttering.
///    On Android, we calibrate this to the "latching" clock using `presentMargin`.
///    Additionally, if a frame "failed", we choose not to present the next frame.
/// 3) The GPU render start clock, used to get a quick estimate for likely stutters.
/// 4) The GPU "latching" clock. This is only implied by `presentMargin` in relation to the render complete clock.
/// 5) The display "present clock". We assume that this is a potentially variable, but generally fixed amount of
///    time from the "latching" clock (due to compositing time).
/// 6) The CPU side present clocks, which have an arbitrary base.
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
/// In the outdated control loop case, we do a best-effort attempt. That involves:
/// 1) We render the first frame immediately. This uses a best-effort estimated present time.
///    This does *not* have an requestedPresentationTime.
/// 2) We then set about rendering the second frame, which uses an estimated present time *one* interval
///    of display refresh rate after the first presentation. We start the rendering work for this frame
///    (tunable) ~35% sooner than one refresh rate interval after the start of rendering work for the first frame.
///    - A potential option here is to cancel this render (GPU side) if the previous frame took (significantly) longer
///      than one refresh interval. It is future work to reason about this.
///      We foresee this as potentially advantageous because the CPU side work would be a relatively small part of
///      a frame, and so it would probably lead to "better" animation smoothness.
/// 3) Once the first frame finishes rendering, we present the second frame. We use "render end time" - "render start time" (`T_R`) to
///    estimate how many refresh durations (`N`) this render took, with a (tunable) ~30% "grace reduction" in the value.
///    This grace amount is to account for a likely ramping up of GPU power states, so future frames will probably be faster.
///    We also use `T_R` to estimate how many refresh durations this took (`N_2`), for scheduling the present of the second frame.
///    This does not use the grace reduction, but does have a grace period of ~50% of the refresh cycle to count as one frame.
///    This grace period is to account for the fact that we don't know when in the refresh cycle we started rendering,
///    so there's a chance that even if both frames took longer than one refresh duration to render, we can "fake" smoothness.
///    Additionally, this helps account for the GPU power state ramp-up.
/// 4) We present the second frame with a time which is "render end time" + previous "present clock - latching clock" + `N_2` refresh durations.
/// 5) We start rendering on the third frame, `N_2 + N` refresh durations after the first frame.
/// 1) We use the stored "real present time - latching clock" from when we last
pub struct Thinking;

/// A sketch of the expected API.
impl VelloPacing {
    pub fn new() -> Self {
        Self {}
    }

    pub fn launch() {}

    /// Run a rendering task until presentation. Useful on macOS for resizing.
    pub fn present_synchronously(&mut self) {
        let token: () = self.present_immediately();
        self.wait_on_present(token);
    }

    pub fn present_immediately(&mut self) {}

    fn wait_on_present(&mut self, (): ()) {}

    pub fn stop(&mut self) {}
}

impl Default for VelloPacing {
    fn default() -> Self {
        Self::new()
    }
}
