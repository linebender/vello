// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::probe::{ALL_PROBE_ELEMENTS, Probe};
use vello_gpu::{WebGlPendingProbe, WebGlProbeReport, WebGlProbeStatus, WebGlRenderer};
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

const MAX_PROBE_POLLS: u32 = 60;
const PROBE_POLL_INTERVAL_MS: f64 = 100.0;

pub(crate) struct ProbeIndicator {
    pending: Option<WebGlPendingProbe>,
    poll_count: u32,
    last_poll_at: Option<f64>,
    container: HtmlElement,
    message: HtmlElement,
}

impl ProbeIndicator {
    pub(crate) fn new(renderer: &mut WebGlRenderer) -> Self {
        let document = web_sys::window().unwrap().document().unwrap();
        let container = document
            .create_element("aside")
            .unwrap()
            .dyn_into::<HtmlElement>()
            .unwrap();
        container.set_attribute("role", "status").unwrap();
        container.set_attribute("aria-live", "polite").unwrap();

        let style = container.style();
        for (property, value) in [
            ("position", "fixed"),
            ("top", "56px"),
            ("right", "12px"),
            ("z-index", "1000"),
            ("min-width", "210px"),
            ("max-width", "320px"),
            ("padding", "10px 12px"),
            ("border", "1px solid rgba(96, 165, 250, 0.55)"),
            ("border-left-width", "4px"),
            ("border-radius", "8px"),
            ("background", "rgba(15, 23, 42, 0.92)"),
            ("box-shadow", "0 8px 24px rgba(0, 0, 0, 0.3)"),
            ("color", "#f8fafc"),
            ("font-family", "system-ui, sans-serif"),
            ("font-size", "13px"),
            ("line-height", "1.35"),
            ("backdrop-filter", "blur(8px)"),
        ] {
            style.set_property(property, value).unwrap();
        }

        let label = document
            .create_element("div")
            .unwrap()
            .dyn_into::<HtmlElement>()
            .unwrap();
        label.set_inner_text("Native WebGL renderer probe");
        label.style().set_property("font-weight", "650").unwrap();

        let message = document
            .create_element("div")
            .unwrap()
            .dyn_into::<HtmlElement>()
            .unwrap();
        message.set_inner_text(&format!(
            "Checking {} rendering features · poll 0/{MAX_PROBE_POLLS}",
            ALL_PROBE_ELEMENTS.len(),
        ));
        let message_style = message.style();
        message_style.set_property("margin-top", "2px").unwrap();
        message_style.set_property("color", "#bfdbfe").unwrap();
        message_style.set_property("font-size", "12px").unwrap();

        container.append_child(&label).unwrap();
        container.append_child(&message).unwrap();
        document.body().unwrap().append_child(&container).unwrap();

        let mut indicator = Self {
            pending: None,
            poll_count: 0,
            last_poll_at: None,
            container,
            message,
        };
        match renderer.probe(ALL_PROBE_ELEMENTS) {
            Ok(pending) => indicator.pending = Some(pending),
            Err(error) => {
                indicator.show_failure(
                    &format!("Could not start the probe: {error}"),
                    "The renderer probe could not be started.",
                );
            }
        }
        indicator
    }

    pub(crate) fn poll(&mut self, timestamp: f64) {
        if self.pending.is_none() {
            return;
        }
        let Some(last_poll_at) = self.last_poll_at else {
            self.last_poll_at = Some(timestamp);
            return;
        };
        if timestamp - last_poll_at < PROBE_POLL_INTERVAL_MS {
            return;
        }
        self.last_poll_at = Some(timestamp);

        let Some(pending) = self.pending.take() else {
            return;
        };
        self.poll_count += 1;

        match pending.try_finish() {
            Ok(WebGlProbeStatus::Pending(_)) if self.poll_count >= MAX_PROBE_POLLS => {
                self.show_failure(
                    &format!("Probe timed out after {} polls", self.poll_count),
                    "The GPU readback did not complete within the probe poll limit.",
                );
            }
            Ok(WebGlProbeStatus::Pending(pending)) => {
                self.message.set_inner_text(&format!(
                    "Checking {} rendering features · poll {}/{MAX_PROBE_POLLS}",
                    ALL_PROBE_ELEMENTS.len(),
                    self.poll_count,
                ));
                self.pending = Some(pending);
            }
            Ok(WebGlProbeStatus::Complete(report)) => self.show_report(report),
            Err(error) => {
                self.show_failure(
                    &format!(
                        "Probe readback failed after {} {}: {error}",
                        self.poll_count,
                        poll_count_label(self.poll_count),
                    ),
                    "The browser could not read the rendered probe back from the GPU.",
                );
            }
        }
    }

    fn show_report(&self, report: WebGlProbeReport) {
        let poll_count = report.poll_count;
        match report.outcome {
            Probe::Success => {
                self.container
                    .style()
                    .set_property("border-color", "rgba(34, 197, 94, 0.75)")
                    .unwrap();
                self.message
                    .style()
                    .set_property("color", "#bbf7d0")
                    .unwrap();
                self.message.set_inner_text(&format!(
                    "Passed all {} rendering checks after {} {}",
                    ALL_PROBE_ELEMENTS.len(),
                    poll_count,
                    poll_count_label(poll_count),
                ));
                self.container
                    .set_attribute("title", "The rendered probe matched its reference output.")
                    .unwrap();
            }
            Probe::Error(result) => {
                let failed_checks = result
                    .statistics
                    .iter()
                    .filter(|statistics| statistics.different_pixel_count > 0)
                    .count();
                let different_pixels = result
                    .statistics
                    .iter()
                    .map(|statistics| statistics.different_pixel_count)
                    .sum::<u32>();
                self.show_failure(
                    &format!(
                        "{failed_checks}/{} checks failed after {} {} · {different_pixels} pixels differ",
                        ALL_PROBE_ELEMENTS.len(),
                        poll_count,
                        poll_count_label(poll_count),
                    ),
                    &format!("Probe differences: {:?}", result.statistics),
                );
            }
            Probe::RenderError(error) => {
                self.show_failure(
                    &format!(
                        "Probe rendering failed after {} {}: {error}",
                        poll_count,
                        poll_count_label(poll_count),
                    ),
                    "The renderer returned an error while drawing the probe.",
                );
            }
        }
    }

    fn show_failure(&self, message: &str, details: &str) {
        self.container
            .style()
            .set_property("border-color", "rgba(248, 113, 113, 0.8)")
            .unwrap();
        self.message
            .style()
            .set_property("color", "#fecaca")
            .unwrap();
        self.message.set_inner_text(message);
        self.container.set_attribute("title", details).unwrap();
    }
}

fn poll_count_label(count: u32) -> &'static str {
    if count == 1 { "poll" } else { "polls" }
}
