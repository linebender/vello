// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build-time shader identifier and whitespace minification.

#![allow(
    dead_code,
    reason = "WGSL minification is used by build.rs; GLSL whitespace minification is used by the library."
)]

use std::collections::BTreeMap;

use naga::{
    Function, Module, TypeInner,
    back::wgsl,
    front::wgsl as wgsl_front,
    valid::{Capabilities, ValidationFlags, Validator},
};

/// Minified WGSL together with names needed to preserve the reflection API.
pub(crate) struct MinifiedWgsl {
    /// Minified, validated WGSL ready to embed.
    pub(crate) source: String,
    /// Maps minified WGSL global names back to their authored names.
    pub(crate) original_global_names: BTreeMap<String, String>,
}

/// Renames every non-entry-point declaration in linked WGSL and removes redundant whitespace.
///
/// For example, `fn helper(value: u32)` may become `fn A(B:u32)`, while `vs_main` is preserved.
pub(crate) fn minify_wgsl(source: &str) -> MinifiedWgsl {
    let mut module =
        wgsl_front::parse_str(source).expect("linked WGSL should parse before minifying");
    let original_global_names = rename_module(&mut module);
    let info = validate(&module);
    let source = wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())
        .expect("renamed WGSL should serialize");

    MinifiedWgsl {
        source: minify_whitespace(&source, false),
        original_global_names,
    }
}

/// Removes comments and whitespace that are unnecessary between shader tokens.
///
/// For example, `let x = 1u;` becomes `let x=1u;`; GLSL preprocessor directives remain on
/// separate lines when `preserve_preprocessor_lines` is enabled.
pub(crate) fn minify_whitespace(source: &str, preserve_preprocessor_lines: bool) -> String {
    if !preserve_preprocessor_lines {
        return minify_code(source);
    }

    let mut output = String::with_capacity(source.len());
    let mut code = String::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            append_minified_code(&mut output, &code);
            code.clear();
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(trimmed);
            output.push('\n');
        } else {
            code.push_str(line);
            code.push('\n');
        }
    }
    append_minified_code(&mut output, &code);

    output
}

/// Appends minified code without merging tokens across the append boundary.
///
/// For example, appending `value;` after `return` inserts the required space.
fn append_minified_code(output: &mut String, source: &str) {
    let minified = minify_code(source);
    if minified.is_empty() {
        return;
    }
    if !output.is_empty()
        && !output.ends_with('\n')
        && needs_separator(last_char(output), first_char(&minified))
    {
        output.push(' ');
    }
    output.push_str(&minified);
}

/// Assigns short names throughout a module and returns the global-name reverse map.
///
/// For example, `Config` and global `config` may become `A` and `B`, with `B -> config` recorded.
fn rename_module(module: &mut Module) -> BTreeMap<String, String> {
    let mut names = NameAllocator::default();

    let type_handles = module
        .types
        .iter()
        .map(|(handle, _)| handle)
        .collect::<Vec<_>>();
    for handle in type_handles {
        let original = module.types[handle].clone();
        let mut ty = original.clone();
        rename_optional(&mut ty.name, &mut names);
        if let TypeInner::Struct { members, .. } = &mut ty.inner {
            for member in members {
                rename_optional(&mut member.name, &mut names);
            }
        }
        if ty != original {
            module.types.replace(handle, ty);
        }
    }

    for (_, constant) in module.constants.iter_mut() {
        rename_optional(&mut constant.name, &mut names);
    }
    for (_, override_) in module.overrides.iter_mut() {
        rename_optional(&mut override_.name, &mut names);
    }

    let mut original_global_names = BTreeMap::new();
    for (_, global) in module.global_variables.iter_mut() {
        if let Some(original) = global.name.take() {
            let minified = names.next();
            original_global_names.insert(minified.clone(), original);
            global.name = Some(minified);
        }
    }

    for (_, function) in module.functions.iter_mut() {
        rename_function(function, &mut names);
    }
    for entry_point in &mut module.entry_points {
        // Entry-point names are part of the renderer's pipeline contract.
        rename_function(&mut entry_point.function, &mut names);
    }

    original_global_names
}

/// Renames a function declaration, its arguments, locals, and named expressions.
///
/// For example, `fn shade(color)` may become `fn A(B)`, including short names for its local lets.
fn rename_function(function: &mut Function, names: &mut NameAllocator) {
    rename_optional(&mut function.name, names);
    for argument in &mut function.arguments {
        rename_optional(&mut argument.name, names);
    }
    for (_, local) in function.local_variables.iter_mut() {
        rename_optional(&mut local.name, names);
    }
    for name in function.named_expressions.values_mut() {
        *name = names.next();
    }
}

/// Replaces an optional Naga IR declaration name with the next short name, leaving unnamed
/// declarations (`None`) unchanged.
///
/// For example, `Some("width")` may become `Some("A")`.
fn rename_optional(name: &mut Option<String>, names: &mut NameAllocator) {
    if name.is_some() {
        *name = Some(names.next());
    }
}

/// Validates the renamed module and produces the metadata required by Naga's writer.
///
/// This does not change the module; invalid bindings, types, or entry points fail the build.
fn validate(module: &Module) -> naga::valid::ModuleInfo {
    Validator::new(ValidationFlags::all(), Capabilities::default())
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(module)
        .expect("renamed WGSL should validate")
}

/// Deterministically allocates compact uppercase identifiers.
#[derive(Default)]
struct NameAllocator {
    next: usize,
}

impl NameAllocator {
    /// Returns the next spreadsheet-style name: `A` through `Z`, then `AA`, `AB`, and so on.
    fn next(&mut self) -> String {
        let mut value = self.next;
        self.next += 1;
        let mut bytes = Vec::with_capacity(3);

        loop {
            bytes.push(b'A' + u8::try_from(value % 26).expect("base-26 digit should fit in u8"));
            value /= 26;
            if value == 0 {
                break;
            }
            value -= 1;
        }
        bytes.reverse();
        String::from_utf8(bytes).expect("generated identifiers should be ASCII")
    }
}

/// Removes comments and optional whitespace while preserving shader token boundaries.
///
/// For example, `a = b + c; // sum` becomes `a=b+c;`.
fn minify_code(source: &str) -> String {
    let chars = source.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(source.len());
    let mut index = 0;
    let mut pending_separator = false;

    while index < chars.len() {
        let current = chars[index];
        let next = chars.get(index + 1).copied();

        if current.is_whitespace() {
            pending_separator = true;
            index += 1;
            continue;
        }
        if current == '/' && next == Some('/') {
            pending_separator = true;
            index += 2;
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }
        if current == '/' && next == Some('*') {
            pending_separator = true;
            index += 2;
            while index + 1 < chars.len() && !(chars[index] == '*' && chars[index + 1] == '/') {
                index += 1;
            }
            index = (index + 2).min(chars.len());
            continue;
        }

        if pending_separator && !output.is_empty() && needs_separator(last_char(&output), current) {
            output.push(' ');
        }
        pending_separator = false;

        if current == '"' || current == '\'' {
            index = copy_quoted(&chars, index, &mut output);
        } else {
            output.push(current);
            index += 1;
        }
    }

    output
}

/// Copies one quoted value verbatim, including escaped characters.
///
/// For example, `"a // b"` remains unchanged rather than treating `//` as a comment.
fn copy_quoted(chars: &[char], start: usize, output: &mut String) -> usize {
    let quote = chars[start];
    let mut index = start;
    while index < chars.len() {
        let current = chars[index];
        output.push(current);
        index += 1;
        if current == '\\' && index < chars.len() {
            output.push(chars[index]);
            index += 1;
        } else if current == quote && index > start + 1 {
            break;
        }
    }
    index
}

/// Returns the first character of a known non-empty source section.
///
/// For example, `first_char("abc")` returns `a`.
fn first_char(source: &str) -> char {
    source.chars().next().expect("source should not be empty")
}

/// Returns the last character of a known non-empty source section.
///
/// For example, `last_char("abc")` returns `c`.
fn last_char(source: &str) -> char {
    source
        .chars()
        .next_back()
        .expect("source should not be empty")
}

/// Reports whether removing whitespace would merge two adjacent shader tokens.
///
/// For example, `return` followed by `value` needs a separator, as do `+` and `+`.
fn needs_separator(left: char, right: char) -> bool {
    let identifier = |character: char| character.is_ascii_alphanumeric() || character == '_';
    (identifier(left) && identifier(right))
        || matches!(
            (left, right),
            ('+', '+')
                | ('-', '-')
                | ('-', '>')
                | ('/', '/')
                | ('/', '*')
                | ('<', '<')
                | ('<', '=')
                | ('>', '>')
                | ('>', '=')
                | ('=', '=')
                | ('!', '=')
                | ('&', '&')
                | ('|', '|')
                | (':', ':')
        )
        || (left.is_ascii_digit() && right == '.')
        || (left == '.' && right.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_short_deterministic_names() {
        let mut names = NameAllocator::default();
        assert_eq!(names.next(), "A");
        for _ in 1..26 {
            names.next();
        }
        assert_eq!(names.next(), "AA");
        assert_eq!(names.next(), "AB");
    }

    #[test]
    fn minifies_root_and_imported_declarations() {
        let source = r#"
struct Configuration {
    width: u32,
}

@group(0) @binding(0)
var<uniform> configuration: Configuration;

fn verbose_helper(value: u32) -> u32 {
    let incremented = value + 1u;
    return incremented;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let result = verbose_helper(vertex_index) + configuration.width;
    return vec4<f32>(f32(result));
}
"#;

        let first = minify_wgsl(source);
        let second = minify_wgsl(source);

        assert_eq!(first.source, second.source);
        assert!(first.source.contains("fn vs_main("));
        assert!(!first.source.contains("Configuration"));
        assert!(!first.source.contains("verbose_helper"));
        assert!(!first.source.contains("incremented"));
        assert!(
            first
                .original_global_names
                .values()
                .any(|name| name == "configuration")
        );
        wgsl_front::parse_str(&first.source).expect("minified WGSL should parse");
    }

    #[test]
    fn keeps_required_token_separators() {
        assert_eq!(
            minify_whitespace(
                "let value = left + +right; // comment\nreturn value;",
                false
            ),
            "let value=left+ +right;return value;"
        );
        assert_eq!(
            minify_whitespace("#version 300 es\n\nprecision highp float;\n", true),
            "#version 300 es\nprecision highp float;"
        );
    }
}
