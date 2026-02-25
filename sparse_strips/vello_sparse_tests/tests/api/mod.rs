// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Ports of the Sparse Strip tests to use [Vello API](vello_api).
//!
//! There are two types of tests in this folder; "Scene-only" tests, and tests using a renderer.
//! The tests using a renderer are there to test things like image handling, etc.
//!
//! We also check that you have all required SIMD levels, so that tests aren't silently ignored.

mod basic;
mod example;
mod infra;

// This message is adapted from https://github.com/linebender/fearless_simd/pull/126
const UNSUPPORTED_LEVEL_MESSAGE: &str = "This means that some of the other tests in this run may be false positives, that is, they have been marked as succeeding even though they would actually fail if they could run.\n\
    When these tests are run on CI, any false positives should be caught.\n\
    However, please open a thread in the #vello channel on the Linebender Zulip if you see this message.\n\
    That would allow us to know whether it's worth us setting up the tests to run on an emulated system (such as using QEMU)";

#[test]
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    ignore = "x86 specific"
)]
fn supports_all_simd_levels_x86() {
    assert!(
        infra::parse_level("fallback").is_some(),
        "Fallback always supported."
    );
    if !(std::env::var("CI").is_ok_and(|it| !it.is_empty())
        || std::env::var("SKIP_UNSUPPORTED_SIMD").is_ok_and(|it| !it.is_empty()))
    {
        assert!(
            infra::parse_level("sse42").is_some(),
            "Your machine does not support SSE4.2.\n{UNSUPPORTED_LEVEL_MESSAGE}"
        );
        assert!(
            infra::parse_level("avx2").is_some(),
            "Your machine does not support AVX2 and/or FMA.\n{UNSUPPORTED_LEVEL_MESSAGE}"
        );
    }
}

#[test]
#[cfg_attr(not(any(target_arch = "aarch64")), ignore = "aarch64 specific")]
fn supports_all_simd_levels_aarch64() {
    assert!(
        infra::parse_level("fallback").is_some(),
        "Fallback always supported."
    );
    if !(std::env::var("CI").is_ok_and(|it| !it.is_empty())
        || std::env::var("SKIP_UNSUPPORTED_SIMD").is_ok_and(|it| !it.is_empty()))
    {
        assert!(
            infra::parse_level("neon").is_some(),
            "Your machine does not support Neon.\n{UNSUPPORTED_LEVEL_MESSAGE}"
        );
    }
}
