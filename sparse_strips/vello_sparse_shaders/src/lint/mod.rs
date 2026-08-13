// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lints for the WGSL output linked from WESL shaders.
//!
//! [`lint`] is the entry point. Each individual pass lives in its own submodule
//! under `lint::*` and exposes a `check(module: &Module) -> Option<LintReport>`.

use naga::Module;

mod no_integer_vertex_inputs;
mod no_structs_in_fragment;

/// Diagnostic produced by a single lint pass when it finds violations.
struct LintReport {
    /// One-line summary of what the lint guards against.
    summary: &'static str,
    /// Multi-paragraph context: why the lint exists, how to fix it.
    explanation: &'static str,
    /// Specific places in the shader that violate the lint.
    violations: Vec<String>,
}

/// Runs every WGSL shader lint over `module` and panics with a single aggregated
/// message (prefixed by `shader_name`) if any lint reports violations.
pub(crate) fn lint(shader_name: &str, module: &Module) {
    let reports: Vec<LintReport> = [
        no_structs_in_fragment::check(module),
        no_integer_vertex_inputs::check(module),
    ]
    .into_iter()
    .flatten()
    .collect();

    if reports.is_empty() {
        return;
    }

    let mut message = format!("`{shader_name}.wesl` failed shader lints:\n");
    for report in &reports {
        use std::fmt::Write as _;
        write!(
            message,
            "\n{}\n\n{}\n\nViolations:\n",
            report.summary, report.explanation,
        )
        .unwrap();
        for violation in &report.violations {
            message.push_str("  - ");
            message.push_str(violation);
            message.push('\n');
        }
    }
    panic!("{message}");
}
