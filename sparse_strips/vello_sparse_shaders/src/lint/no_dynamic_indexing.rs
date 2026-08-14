// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lint pass: shaders must not use dynamically computed indices.

use naga::{Expression, Function, Module};

use super::LintReport;

pub(super) fn check(module: &Module) -> Option<LintReport> {
    let mut violations = Vec::new();

    for (_, function) in module.functions.iter() {
        let name = function.name.as_deref().unwrap_or("<unnamed>");
        check_function(function, &format!("function `{name}`"), &mut violations);
    }
    for entry_point in &module.entry_points {
        check_function(
            &entry_point.function,
            &format!("entry point `{}`", entry_point.name),
            &mut violations,
        );
    }

    (!violations.is_empty()).then_some(LintReport {
        summary: "contains one or more dynamic index expressions.",
        explanation: "Dynamic indexing has caused correctness and performance problems on older \
                      GPU drivers. Replace dynamically indexed arrays, vectors, or matrices with \
                      direct constant accesses, vector operations, or explicit control flow.",
        violations,
    })
}

fn check_function(function: &Function, function_label: &str, violations: &mut Vec<String>) {
    for (_, expression) in function.expressions.iter() {
        if matches!(expression, Expression::Access { .. }) {
            violations.push(format!("{function_label}: dynamic index expression"));
        }
    }
}

#[cfg(test)]
mod tests {
    use naga::front::wgsl;

    use super::super::lint;

    fn lint_source(source: &str) {
        lint(
            "test_shader",
            &wgsl::parse_str(source).expect("WGSL parses"),
        );
    }

    #[test]
    fn accepts_static_indexing() {
        lint_source(
            r#"
fn select() -> u32 {
    let values = array<u32, 3>(1u, 2u, 3u);
    return values[1];
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(select()));
}
"#,
        );
    }

    #[test]
    #[should_panic(expected = "function `select`: dynamic index expression")]
    fn rejects_dynamic_indexing_in_helper_function() {
        lint_source(
            r#"
fn select(index: u32) -> u32 {
    let values = array<u32, 3>(1u, 2u, 3u);
    return values[index];
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(select(1u)));
}
"#,
        );
    }

    #[test]
    #[should_panic(expected = "entry point `vs_main`: dynamic index expression")]
    fn rejects_dynamic_indexing_in_entry_point() {
        lint_source(
            r#"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec4<f32>, 2>(vec4<f32>(0.0), vec4<f32>(1.0));
    return positions[index];
}
"#,
        );
    }
}
