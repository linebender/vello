// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build step.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=VELLO_CI_GPU_SUPPORT");
    println!("cargo:rustc-check-cfg=cfg(skip_gpu_tests)");
    if let Ok(mut value) = env::var("VELLO_CI_GPU_SUPPORT") {
        value.make_ascii_lowercase();
        match &*value {
            "yes" | "y" => {}
            "no" | "n" => {
                println!("cargo:rustc-cfg=skip_gpu_tests");
            }
            _ => {
                println!("cargo:cargo:warning=VELLO_CI_GPU_SUPPORT should be set to yes/y or no/n");
            }
        }
    }
}
