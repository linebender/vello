pub struct VelloPacing {}

/// Docs used to type out how this is being reasoned about.
///
/// We manage four clocks:
/// 1) The CPU side clock. This is treated as an opaque nanosecond value, used only for sleeping.
/// 2) The GPU "render complete clock". We wait for rendering work to complete to schedule the "present" of the *next* frame, to avoid stuttering.
///    On Android, we calibrate this to the "latching" clock using `presentMargin`.
///    Additionally, if a frame "failed", we.
/// 3) The GPU "latching" clock. This is the
/// 4) The display "present clock".
///
/// We observe that `earliest_present_time` can be earlier than `actual_present_time` because.
///
/// There are three cases for when we want to render:
/// 1) Active rendering, i.e. animation
/// 2) Rendering "once", i.e. in response to a keypress. Note that scrolling will likely be treated as animation, and is an important case for latency minimisation.
/// 3) Stopping active rendering, i.e. the last frame
///
/// Note also that we need to be aware of <https://github.com/linebender/vello/pull/606> in this design.
///
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
