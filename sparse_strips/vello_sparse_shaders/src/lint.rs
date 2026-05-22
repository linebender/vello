// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lints for the WGSL shaders.
//!
//! See [`lint`] for the entry point and the individual `check_*` passes for what
//! each lint enforces.

use std::collections::{BTreeSet, VecDeque};

use naga::{Block, Expression, Function, Handle, Module, ShaderStage, Statement, Type, TypeInner};

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
    let reports: Vec<LintReport> = [check_no_structs_in_fragment_shader(module)]
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

/// Detects user-defined struct values reachable from the `@fragment` entry point.
///
/// A row of older Adreno GPU drivers silently downgrade the numeric precision of
/// structs in fragment shaders to 16 bits, regardless of the precision the shader
/// actually requested. The workaround is to flatten all data into scalar or vector
/// primitives. See <https://github.com/linebender/vello/pull/1604> and the issues
/// linked from it.
///
/// Struct-typed uniform globals (e.g. `Config`) and vertex IO structs are allowed
/// because their fields are only ever read as scalars/vectors inside the fragment
/// shader — the struct itself is never materialised in fragment-shader storage.
///
/// Returns `None` for modules without a fragment entry point or when the rule is
/// upheld.
fn check_no_structs_in_fragment_shader(module: &Module) -> Option<LintReport> {
    let entry_point = module
        .entry_points
        .iter()
        .find(|ep| ep.stage == ShaderStage::Fragment)?;

    let mut violations = Vec::new();
    check_function(
        module,
        &format!("fragment entry point `{}`", entry_point.name),
        &entry_point.function,
        &mut violations,
    );

    let mut reachable: BTreeSet<Handle<Function>> = BTreeSet::new();
    let mut queue: VecDeque<Handle<Function>> = VecDeque::new();
    collect_called_functions(&entry_point.function.body, &mut |handle| {
        if reachable.insert(handle) {
            queue.push_back(handle);
        }
    });
    while let Some(handle) = queue.pop_front() {
        let func = &module.functions[handle];
        let name = func
            .name
            .clone()
            .unwrap_or_else(|| format!("<fn {handle:?}>"));
        check_function(module, &format!("function `{name}`"), func, &mut violations);

        collect_called_functions(&func.body, &mut |handle| {
            if reachable.insert(handle) {
                queue.push_back(handle);
            }
        });
    }

    if violations.is_empty() {
        return None;
    }

    Some(LintReport {
        summary: "uses one or more struct values in code reachable from its fragment entry point.",
        explanation: "A row of Adreno GPU drivers silently downgrades the precision of struct \
                      values in fragment shaders to 16 bits, which corrupts paint encodings and \
                      coordinates. Flatten the data into scalar or vector primitives instead \
                      (e.g. unpack a `vec4<u32>` texel directly with top-level `get_*` helpers).\n\
                      See https://github.com/linebender/vello/pull/1604 for context.",
        violations,
    })
}

fn check_function(
    module: &Module,
    function_label: &str,
    func: &Function,
    violations: &mut Vec<String>,
) {
    for (idx, arg) in func.arguments.iter().enumerate() {
        if is_struct(module, arg.ty) {
            violations.push(format!(
                "{function_label}: argument #{idx} (`{}`) has struct type `{}`",
                arg.name.as_deref().unwrap_or("<unnamed>"),
                type_name(module, arg.ty),
            ));
        }
    }
    if let Some(result) = &func.result
        && is_struct(module, result.ty)
    {
        violations.push(format!(
            "{function_label}: return value has struct type `{}`",
            type_name(module, result.ty),
        ));
    }
    for (_, local) in func.local_variables.iter() {
        if is_struct(module, local.ty) {
            violations.push(format!(
                "{function_label}: local variable `{}` has struct type `{}`",
                local.name.as_deref().unwrap_or("<unnamed>"),
                type_name(module, local.ty),
            ));
        }
    }
    for (_, expr) in func.expressions.iter() {
        match expr {
            Expression::Compose { ty, .. } if is_struct(module, *ty) => {
                violations.push(format!(
                    "{function_label}: struct constructor for `{}`",
                    type_name(module, *ty),
                ));
            }
            Expression::ZeroValue(ty) if is_struct(module, *ty) => {
                violations.push(format!(
                    "{function_label}: `zero_value` of struct type `{}`",
                    type_name(module, *ty),
                ));
            }
            Expression::Load { pointer } => {
                if let Some((kind, name, ty)) = base_variable_load(module, func, *pointer)
                    && is_struct(module, ty)
                {
                    violations.push(format!(
                        "{function_label}: load of {kind} `{name}` of struct type `{}`",
                        type_name(module, ty),
                    ));
                }
            }
            _ => {}
        }
    }
}

/// If `pointer` is a direct load of a `GlobalVariable` or `LocalVariable` (with no
/// `Access` / `AccessIndex` reducing it to a field), returns the variable's kind,
/// name, and type. Otherwise returns `None` — partial loads like `config.width`
/// correctly fall through.
fn base_variable_load(
    module: &Module,
    func: &Function,
    pointer: Handle<Expression>,
) -> Option<(&'static str, String, Handle<Type>)> {
    match func.expressions[pointer] {
        Expression::GlobalVariable(handle) => {
            let global = &module.global_variables[handle];
            Some((
                "global variable",
                global
                    .name
                    .clone()
                    .unwrap_or_else(|| "<unnamed>".to_string()),
                global.ty,
            ))
        }
        Expression::LocalVariable(handle) => {
            let local = &func.local_variables[handle];
            Some((
                "local variable",
                local
                    .name
                    .clone()
                    .unwrap_or_else(|| "<unnamed>".to_string()),
                local.ty,
            ))
        }
        _ => None,
    }
}

fn collect_called_functions(block: &Block, on_call: &mut impl FnMut(Handle<Function>)) {
    for stmt in block.iter() {
        match stmt {
            Statement::Block(b) => collect_called_functions(b, on_call),
            Statement::If { accept, reject, .. } => {
                collect_called_functions(accept, on_call);
                collect_called_functions(reject, on_call);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_called_functions(&case.body, on_call);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_called_functions(body, on_call);
                collect_called_functions(continuing, on_call);
            }
            Statement::Call { function, .. } => on_call(*function),
            _ => {}
        }
    }
}

fn is_struct(module: &Module, ty: Handle<Type>) -> bool {
    matches!(module.types[ty].inner, TypeInner::Struct { .. })
}

fn type_name(module: &Module, ty: Handle<Type>) -> String {
    module.types[ty]
        .name
        .clone()
        .unwrap_or_else(|| "<unnamed type>".to_string())
}

#[cfg(test)]
mod tests {
    use naga::front::wgsl;

    use super::*;

    const SHADER_WITH_STRUCT_LOCAL: &str = r#"
struct Config { width: u32 }
struct MyPair { a: u32, b: u32 }

@group(0) @binding(0) var<uniform> config: Config;

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    var pair: MyPair;
    pair.a = config.width;
    return vec4<f32>(f32(pair.a), f32(pair.b), 0.0, 1.0);
}
"#;

    const SHADER_WITH_STRUCT_HELPER: &str = r#"
struct Pair { a: u32, b: u32 }

fn make_pair() -> Pair {
    return Pair(1u, 2u);
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let p = make_pair();
    return vec4<f32>(f32(p.a), f32(p.b), 0.0, 1.0);
}
"#;

    const SHADER_WITH_STRUCT_FRAGMENT_INPUT: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coord: vec2<f32>,
}

@vertex
fn vs_main() -> VertexOutput {
    var out: VertexOutput;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.coord, 0.0, 1.0);
}
"#;

    const SHADER_WITHOUT_STRUCTS: &str = r#"
struct Config { width: u32 }
struct VertexOutput { @builtin(position) position: vec4<f32> }

@group(0) @binding(0) var<uniform> config: Config;

@vertex
fn vs_main() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(f32(config.width));
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(config.width), 0.0, 0.0, 1.0);
}
"#;

    // Loading a whole uniform struct into a `let` materialises the struct value
    // in fragment-shader storage and triggers the same Adreno precision bug as
    // declaring a struct local does.
    const SHADER_LOADS_UNIFORM_STRUCT: &str = r#"
struct Config { width: u32, height: u32 }

@group(0) @binding(0) var<uniform> config: Config;

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    let copied = config;
    return vec4<f32>(f32(copied.width), f32(copied.height), 0.0, 1.0);
}
"#;

    fn parse(source: &str) -> Module {
        wgsl::parse_str(source).expect("WGSL parses")
    }

    #[test]
    fn accepts_fragment_shader_without_struct_values() {
        lint("ok_shader", &parse(SHADER_WITHOUT_STRUCTS));
    }

    #[test]
    #[should_panic(expected = "uses one or more struct values")]
    fn rejects_struct_local_variable_in_fragment_shader() {
        lint("with_struct_local", &parse(SHADER_WITH_STRUCT_LOCAL));
    }

    #[test]
    #[should_panic(expected = "uses one or more struct values")]
    fn rejects_helper_function_returning_struct() {
        lint("with_struct_helper", &parse(SHADER_WITH_STRUCT_HELPER));
    }

    // The same Adreno bug also fires when the fragment entry point takes a struct as
    // its input parameter (every field of `in` gets silently downgraded to 16-bit).
    #[test]
    #[should_panic(expected = "argument #0 (`in`) has struct type `VertexOutput`")]
    fn rejects_struct_as_fragment_entry_point_input() {
        lint(
            "with_struct_fragment_input",
            &parse(SHADER_WITH_STRUCT_FRAGMENT_INPUT),
        );
    }

    // Whole-struct load of a uniform global into a `let`.
    #[test]
    #[should_panic(expected = "load of global variable `config` of struct type `Config`")]
    fn rejects_whole_uniform_struct_load_in_fragment_shader() {
        lint("loads_uniform_struct", &parse(SHADER_LOADS_UNIFORM_STRUCT));
    }

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
            lint(&shader.name, &parse(&shader.wgsl_source));
        }
    }
}
