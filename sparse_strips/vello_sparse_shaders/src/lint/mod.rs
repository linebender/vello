// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lints for the WGSL shaders.
//!
//! [`lint`] is the entry point. Each individual pass lives in its own submodule
//! under `lint::*` and exposes a `check(module: &Module) -> Option<LintReport>`.

use naga::Module;

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
    let reports: Vec<LintReport> = [no_structs_in_fragment::check(module)]
        .into_iter()
        .flatten()
        .collect();

    if reports.is_empty() {
        return;
    }

    let mut message = format!("`{shader_name}.wgsl` failed shader lints:\n");
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

#[cfg(test)]
mod tests {
    use naga::front::wgsl;

    use super::*;

    #[test]
    fn every_shipped_shader_passes_the_lint() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let shader_dir = manifest_dir.join("shaders");
        let shaders =
            crate::shader_info::load_shader_infos(&shader_dir).expect("load WGSL shaders");
        assert!(
            !shaders.is_empty(),
            "expected at least one shader in {shader_dir:?}"
        );
        for shader in shaders {
            let module = wgsl::parse_str(&shader.wgsl_source).expect("WGSL parses");
            lint(&shader.name, &module);
        }
    }
}
