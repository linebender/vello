// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scratch module for findings of the CPU/GPU differential fuzz target.
//!
//! `fuzz/validate.sh <artifact>` adds a test here (see `fuzz/README.md`) so it runs against every
//! CPU and GPU variant of the snapshot suite. Once a finding is understood, move the test into the
//! module matching its topic (or `issues.rs`) under a descriptive name, rename its reference image
//! in `snapshots/` to match, and reset this file with `git checkout`. Nothing added here is meant
//! to be committed.
