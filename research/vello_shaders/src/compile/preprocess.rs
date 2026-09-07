// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::{fs, vec};

pub fn get_imports(shader_dir: &Path) -> HashMap<String, String> {
    let mut imports = HashMap::new();
    let imports_dir = shader_dir.join("shared");
    for entry in imports_dir
        .read_dir()
        .expect("Can read shader import directory")
        .filter_map(move |e| {
            e.ok()
                .filter(|e| e.path().extension().map(|e| e == "wgsl").unwrap_or(false))
        })
    {
        let file_name = entry.file_name();
        if let Some(name) = file_name.to_str() {
            let suffix = ".wgsl";
            if let Some(import_name) = name.strip_suffix(suffix) {
                let contents = fs::read_to_string(imports_dir.join(&file_name))
                    .unwrap_or_else(|_| panic!("Couldn't read shader {import_name} contents"));
                imports.insert(import_name.to_owned(), contents);
            }
        }
    }
    imports
}

pub struct StackItem {
    active: bool,
    else_passed: bool,
}

pub fn preprocess(
    input: &str,
    shader_name: &str,
    defines: &HashSet<String>,
    imports: &HashMap<String, String>,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut stack = vec![];
    'all_lines: for (line_number, mut line) in input.lines().enumerate() {
        loop {
            if line.is_empty() {
                break;
            }
            let hash_index = line.find('#');
            let comment_index = line.find("//");
            let hash_index = match (hash_index, comment_index) {
                (Some(hash_index), None) => hash_index,
                (Some(hash_index), Some(comment_index)) if hash_index < comment_index => hash_index,
                // Add this line to the output - all directives are commented out or there are no directives
                _ => break,
            };
            let directive_start = &line[hash_index + '#'.len_utf8()..];
            let directive_len = directive_start
                // The first character which can't be part of the directive name marks the end of the directive
                // In practise this should always be whitespace, but in theory a 'unit' directive
                // could be added
                .find(|c: char| !c.is_alphanumeric())
                .unwrap_or(directive_start.len());
            let directive = &directive_start[..directive_len];
            let directive_is_at_start = line.trim_start().starts_with('#');

            match directive {
                item @ ("ifdef" | "ifndef" | "else" | "endif" | "enable")
                    if !directive_is_at_start =>
                {
                    log::warn!(
                        "#{item} directives must be the first non_whitespace items on \
                               their line, ignoring (line {line_number} of {shader_name}.wgsl)"
                    );
                    break;
                }
                def_test @ ("ifdef" | "ifndef") => {
                    let def = directive_start[directive_len..].trim();
                    let exists = defines.contains(def);
                    let mode = def_test == "ifdef";
                    stack.push(StackItem {
                        active: mode == exists,
                        else_passed: false,
                    });
                    // Don't add this line to the output; instead process the next line
                    continue 'all_lines;
                }
                "else" => {
                    let item = stack.last_mut();
                    if let Some(item) = item {
                        if item.else_passed {
                            log::warn!(
                                "Second else for same ifdef/ifndef (line {line_number} of {shader_name}.wgsl); \
                                       ignoring second else"
                            );
                        } else {
                            item.else_passed = true;
                            item.active = !item.active;
                        }
                    }
                    let remainder = directive_start[directive_len..].trim();
                    if !remainder.is_empty() {
                        log::warn!(
                            "#else directives don't take an argument. `{remainder}` will not \
                                   be in output (line {line_number} of {shader_name}.wgsl)"
                        );
                    }
                    // Don't add this line to the output; it should be empty (see warning above)
                    continue 'all_lines;
                }
                "endif" => {
                    if stack.pop().is_none() {
                        log::warn!("Mismatched endif (line {line_number} of {shader_name}.wgsl)");
                    }
                    let remainder = directive_start[directive_len..].trim();
                    if !remainder.is_empty() && !remainder.starts_with("//") {
                        log::warn!(
                            "#endif directives don't take an argument. `{remainder}` will \
                                   not be in output (line {line_number} of {shader_name}.wgsl)"
                        );
                    }
                    // Don't add this line to the output; it should be empty (see warning above)
                    continue 'all_lines;
                }
                "import" => {
                    output.push_str(&line[..hash_index]);
                    let directive_end = &directive_start[directive_len..];
                    let import_name_start = if let Some(import_name_start) =
                        directive_end.find(|c: char| !c.is_whitespace())
                    {
                        import_name_start
                    } else {
                        log::warn!(
                            "#import needs a non_whitespace argument (line {line_number} of {shader_name}.wgsl)"
                        );
                        continue 'all_lines;
                    };
                    let import_name_start = &directive_end[import_name_start..];
                    let import_name_end_index = import_name_start
                        // The first character which can't be part of the import name marks the end of the import
                        .find(|c: char| !(c == '_' || c.is_alphanumeric()))
                        .unwrap_or(import_name_start.len());
                    let import_name = &import_name_start[..import_name_end_index];
                    line = &import_name_start[import_name_end_index..];
                    let import = imports.get(import_name);
                    if let Some(import) = import {
                        // In theory, we can cache this until the top item of the stack changes
                        // However, in practise there will only ever be at most 2 stack items, so
                        // it's reasonable to just recompute it every time
                        if stack.iter().all(|item| item.active) {
                            output.push_str(&preprocess(import, shader_name, defines, imports));
                        }
                    } else {
                        log::warn!(
                            "Unknown import `{import_name}` (line {line_number} of {shader_name}.wgsl)"
                        );
                    }
                    continue;
                }
                "enable" => {
                    // Turn this directive into a comment. It will be handled as part in
                    // postprocess.
                    if stack.iter().all(|item| item.active) {
                        output.push_str("//__");
                        output.push_str(line);
                        output.push('\n');
                    }
                    continue 'all_lines;
                }
                val => {
                    log::warn!(
                        "Unknown preprocessor directive `{val}` (line {line_number} of {shader_name}.wgsl)"
                    );
                }
            }
        }
        if stack.iter().all(|item| item.active) {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}
