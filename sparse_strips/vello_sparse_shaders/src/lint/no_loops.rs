// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lint pass: shaders must not contain loops.

use naga::{Block, Module, Statement};

use super::LintReport;

pub(super) fn check(module: &Module) -> Option<LintReport> {
    let mut violations = Vec::new();

    for (_, function) in module.functions.iter() {
        let name = function.name.as_deref().unwrap_or("<unnamed>");
        check_block(
            &function.body,
            &format!("function `{name}`"),
            &mut violations,
        );
    }
    for entry_point in &module.entry_points {
        check_block(
            &entry_point.function.body,
            &format!("entry point `{}`", entry_point.name),
            &mut violations,
        );
    }

    (!violations.is_empty()).then_some(LintReport {
        summary: "contains one or more loops.",
        explanation: "Loops have caused correctness and performance problems on older GPU drivers. \
                      Unroll the loop or replace it with vector operations. This rule applies to \
                      `for`, `while`, and `loop` statements in every shader stage and helper function.",
        violations,
    })
}

fn check_block(block: &Block, function: &str, violations: &mut Vec<String>) {
    for statement in block.iter() {
        match statement {
            Statement::Block(block) => check_block(block, function, violations),
            Statement::If { accept, reject, .. } => {
                check_block(accept, function, violations);
                check_block(reject, function, violations);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    check_block(&case.body, function, violations);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                violations.push(format!("{function}: loop statement"));
                check_block(body, function, violations);
                check_block(continuing, function, violations);
            }
            _ => {}
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
    fn accepts_shader_without_loops() {
        lint_source(
            r#"
fn choose(value: bool) -> f32 {
    if value { return 1.0; }
    return 0.0;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(choose(true));
}
"#,
        );
    }

    #[test]
    #[should_panic(expected = "function `sum`: loop statement")]
    fn rejects_for_loop_in_helper_function() {
        lint_source(
            r#"
fn sum() -> u32 {
    var result = 0u;
    for (var i = 0u; i < 3u; i++) {
        result += i;
    }
    return result;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(sum()));
}
"#,
        );
    }

    #[test]
    #[should_panic(expected = "entry point `fs_main`: loop statement")]
    fn rejects_while_loop_in_entry_point() {
        lint_source(
            r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    var i = 0u;
    while i < 3u {
        i++;
    }
    return vec4<f32>(f32(i));
}
"#,
        );
    }

    #[test]
    #[should_panic(expected = "entry point `vs_main`: loop statement")]
    fn rejects_loop_in_vertex_entry_point() {
        lint_source(
            r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    var i = 0u;
    loop {
        i++;
        if i == 3u { break; }
    }
    return vec4<f32>(f32(i));
}
"#,
        );
    }
}
