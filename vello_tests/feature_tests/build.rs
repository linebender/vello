// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Select the reference image for the enabled shader features.

fn main() {
    let features = ["blurred_rounded_rect", "image_bicubic", "gradient_sweep"];
    let enabled = features
        .iter()
        .filter(|feature| {
            let variable = format!("CARGO_FEATURE_{}", feature.to_uppercase());
            println!("cargo:rerun-if-env-changed={variable}");
            std::env::var_os(variable).is_some()
        })
        .copied()
        .collect::<Vec<_>>();
    let name = match enabled.len() {
        0 => "none".to_owned(),
        3 => "all".to_owned(),
        _ => enabled.join("_"),
    };
    println!("cargo:rustc-env=PROBE_SNAPSHOT_NAME=feature_tests_probe_{name}");
}
