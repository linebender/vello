// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![no_main]

mod compare;
mod config;
mod harness;
mod images;
mod replay;
mod scene;
mod snapshot_test;

use arbitrary::{Arbitrary, Unstructured};
use harness::DifferentialHarness;
use libfuzzer_sys::fuzz_target;
use scene::FuzzScene;
use std::cell::RefCell;

thread_local! {
    // Leaking the renderer is intentional. On some backends, dropping wgpu resources from a
    // thread-local destructor accesses wgpu's own TLS after it has already been destroyed.
    static HARNESS: RefCell<Option<&'static mut DifferentialHarness>> = const { RefCell::new(None) };
}

fuzz_target!(|data: &[u8]| {
    let Ok(scene) = FuzzScene::arbitrary(&mut Unstructured::new(data)) else {
        return;
    };
    let decoded = snapshot_test::write(&scene);
    let diff_stem = config::DIFF_OUTPUT.as_ref();
    if decoded && diff_stem.is_none() {
        return;
    }
    HARNESS.with(|slot| {
        let mut slot = slot.borrow_mut();
        let harness = slot.get_or_insert_with(|| Box::leak(Box::new(DifferentialHarness::new())));
        match diff_stem {
            Some(stem) => harness.write_diff(&scene, stem),
            None => harness.run(data, &scene),
        }
    });
});
