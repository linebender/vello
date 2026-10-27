//! Temporary instrumentation: accumulates the time spent in the phases of rendering.

extern crate std;

use core::cell::RefCell;
use std::thread_local;

/// The accumulated time per rendering phase, in milliseconds, since the last [`take`].
#[derive(Clone, Debug, Default)]
pub struct RenderPhaseTimings {
    pub renders: u32,
    pub before_render_ms: f64,
    pub encode_paints_ms: f64,
    pub schedule_ms: f64,
    pub intermediate_textures_ms: f64,
    pub upload_ms: f64,
    pub execute_ms: f64,
    pub after_render_ms: f64,
    pub strips: u64,
    pub alpha_bytes: u64,
    pub encoded_paints: u64,
}

impl core::fmt::Display for RenderPhaseTimings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} renders: before_render {:.2}ms, encode_paints {:.2}ms, schedule {:.2}ms, \
             intermediate_textures {:.2}ms, upload {:.2}ms, execute {:.2}ms, after_render {:.2}ms \
             | per render: {} strips, {} KiB alphas, {} paints",
            self.renders,
            self.before_render_ms,
            self.encode_paints_ms,
            self.schedule_ms,
            self.intermediate_textures_ms,
            self.upload_ms,
            self.execute_ms,
            self.after_render_ms,
            self.strips / u64::from(self.renders.max(1)),
            self.alpha_bytes / u64::from(self.renders.max(1)) / 1024,
            self.encoded_paints / u64::from(self.renders.max(1)),
        )
    }
}

thread_local! {
    static TIMINGS: RefCell<RenderPhaseTimings> = RefCell::new(RenderPhaseTimings::default());
}

/// Returns the timings accumulated on this thread and resets them.
pub fn take() -> RenderPhaseTimings {
    TIMINGS.with(|timings| core::mem::take(&mut *timings.borrow_mut()))
}

pub(crate) fn record(f: impl FnOnce(&mut RenderPhaseTimings)) {
    TIMINGS.with(|timings| f(&mut timings.borrow_mut()));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub(crate) fn now_ms() -> f64 {
    let performance = js_sys::Reflect::get(&js_sys::global(), &"performance".into()).unwrap();
    let now = js_sys::Reflect::get(&performance, &"now".into()).unwrap();
    js_sys::Function::from(now)
        .call0(&performance)
        .unwrap()
        .as_f64()
        .unwrap()
}

#[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
pub(crate) fn now_ms() -> f64 {
    0.0
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}
