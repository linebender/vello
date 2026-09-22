// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Settings read from the command line.
//!
//! libFuzzer owns the process arguments and uses single-dash flags itself, but it ignores flags
//! that start with `--`, which leaves those free for the target.

use std::ffi::OsString;
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::LazyLock;

// The regular snapshot suite allows a default GPU tolerance of one. Generated scenes exercise
// many anti-aliased boundaries at once, so this target permits a slightly larger local difference
// and a handful of isolated outliers while still catching structural rendering failures.
const DEFAULT_CHANNEL_TOLERANCE: u8 = 2;
const DEFAULT_MAX_OUTLIER_PIXELS: usize = 4;

/// Thresholds for the CPU/GPU image comparison.
pub(crate) struct Tolerance {
    /// Largest per-channel difference a pixel may have without counting as an outlier.
    pub(crate) channel: u8,
    /// Number of outlier pixels an image pair may contain and still match.
    pub(crate) max_outlier_pixels: usize,
}

pub(crate) static TOLERANCE: LazyLock<Tolerance> = LazyLock::new(|| Tolerance {
    channel: target_flag("channel-tolerance").unwrap_or(DEFAULT_CHANNEL_TOLERANCE),
    max_outlier_pixels: target_flag("max-outlier-pixels").unwrap_or(DEFAULT_MAX_OUTLIER_PIXELS),
});

/// `--diff` writes `<artifact>.png` and `<artifact>.json`.
pub(crate) static DIFF_OUTPUT: LazyLock<Option<PathBuf>> =
    LazyLock::new(|| diagnostic_output("diff", ""));

/// `--decode-to-rs` writes `<artifact>.rs`.
pub(crate) static DECODE_OUTPUT: LazyLock<Option<PathBuf>> =
    LazyLock::new(|| diagnostic_output("decode-to-rs", ".rs"));

/// With `--dedup`, continue mode saves only the first finding with a given signature: the set of
/// differing pixels for a mismatch, the panic location for a crash. Later ones are reported as
/// duplicates without writing artifacts.
pub(crate) static DEDUP: LazyLock<bool> = LazyLock::new(|| has_target_flag("dedup"));

/// `--image-quality=all` lets image paints use `Medium` and `High` sampling as well as `Low`.
/// Those are known to differ between the backends, so the default keeps them out of the
/// generated scenes to avoid drowning other findings.
pub(crate) static IMAGE_QUALITY_ALL: LazyLock<bool> =
    LazyLock::new(|| match target_flag::<String>("image-quality").as_deref() {
        None | Some("low") => false,
        Some("all") => true,
        Some(value) => {
            eprintln!("invalid value {value:?} for --image-quality: expected `low` or `all`");
            std::process::exit(2);
        }
    });

/// With `--continue` a mismatch is recorded as `mismatch-<hash>` and a renderer panic as
/// `crash-<hash>` in libFuzzer's artifact directory, and fuzzing carries on.
pub(crate) static CONTINUE_ARTIFACT_PREFIX: LazyLock<Option<OsString>> = LazyLock::new(|| {
    if !has_target_flag("continue") {
        return None;
    }
    // Like libFuzzer, the last `-artifact_prefix` wins; cargo-fuzz adds its own before user flags.
    let prefix = std::env::args_os()
        .skip(1)
        .filter_map(|argument| {
            argument
                .to_str()?
                .strip_prefix("-artifact_prefix=")
                .map(OsString::from)
        })
        .next_back()
        .unwrap_or_else(|| OsString::from("fuzz/artifacts/cpu_gpu_differential/"));
    Some(prefix)
});

/// Whether a bare `--<name>` argument is present.
fn has_target_flag(name: &str) -> bool {
    let flag = format!("--{name}");
    std::env::args_os()
        .skip(1)
        .any(|argument| argument.to_str() == Some(flag.as_str()))
}

/// Parses the last `--<name>=<value>` argument, exiting with a message on a malformed value.
fn target_flag<T>(name: &str) -> Option<T>
where
    T: FromStr,
    T::Err: Display,
{
    let prefix = format!("--{name}=");
    let value = std::env::args_os()
        .skip(1)
        .filter_map(|argument| argument.to_str()?.strip_prefix(&prefix).map(str::to_owned))
        .next_back()?;
    match value.parse() {
        Ok(parsed) => Some(parsed),
        Err(error) => {
            eprintln!("invalid value {value:?} for --{name}: {error}");
            std::process::exit(2);
        }
    }
}

/// Output path for a diagnostic mode flag, derived from the artifact path on the command line.
fn diagnostic_output(flag: &str, extension: &str) -> Option<PathBuf> {
    if !has_target_flag(flag) {
        return None;
    }
    let mut output = std::env::args_os()
        .skip(1)
        .find(|argument| !argument.to_string_lossy().starts_with('-'))
        .unwrap_or_else(|| {
            eprintln!("--{flag} requires an artifact path");
            std::process::exit(2);
        });
    output.push(extension);
    Some(output.into())
}
