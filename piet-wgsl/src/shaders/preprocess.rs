use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    vec,
};

pub fn get_imports(shader_dir: &Path) -> HashMap<String, String> {
    let mut imports = HashMap::new();
    let imports_dir = shader_dir.join("shared");
    for entry in imports_dir
        .read_dir()
        .expect("Can read shader import directory")
    {
        let entry = entry.expect("Can continue reading shader import directory");
        if entry.file_type().unwrap().is_file() {
            let file_name = entry.file_name();
            if let Some(name) = file_name.to_str() {
                let suffix = ".wgsl";
                if name.ends_with(suffix) {
                    let import_name = name[..(name.len() - suffix.len())].to_owned();
                    let contents = fs::read_to_string(imports_dir.join(file_name))
                        .expect("Could read shader {import_name} contents");
                    imports.insert(import_name, contents);
                }
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
    defines: &HashSet<String>,
    imports: &HashMap<String, String>,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut stack = vec![];
    'outer: for (line_number, mut line) in input.lines().enumerate() {
        loop {
            if line.is_empty() {
                break;
            }
            let hash_index = line.find('#');
            let comment_index = line.find("//");
            let hash_index = match (hash_index, comment_index) {
                (Some(hash_index), None) => hash_index,
                (Some(hash_index), Some(comment_index)) if hash_index < comment_index => hash_index,
                _ => break,
            };
            let directive_start = &line[hash_index + '#'.len_utf8()..];
            let directive_len = directive_start
                .chars()
                .take_while(|char| char.is_alphanumeric())
                .map(char::len_utf8)
                .sum();
            let directive = &directive_start[..directive_len];
            let directive_is_at_start = line.trim_start().starts_with('#');

            match directive {
                if_item @ ("ifdef" | "ifndef" | "else" | "endif") if !directive_is_at_start => {
                    eprintln!("#{if_item} directives must be the first non_whitespace items on their line, ignoring (line {line_number})");
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
                    continue 'outer;
                }
                "else" => {
                    let item = stack.last_mut();
                    if let Some(item) = item {
                        if item.else_passed {
                            eprintln!("Second else for same ifdef/ifndef (line {line_number}); ignoring second else")
                        } else {
                            item.else_passed = true;
                            item.active = !item.active;
                        }
                    }
                    let remainder = directive_start[directive_len..].trim();
                    if !remainder.is_empty() {
                        eprintln!("#else directives don't take an argument. `{remainder}` will not be in output (line {line_number})");
                    }
                    continue 'outer;
                }
                "endif" => {
                    if let None = stack.pop() {
                        eprintln!("Mismatched endif (line {line_number})");
                    }
                    continue 'outer;
                }
                "import" => {
                    output.push_str(&line[..hash_index]);
                    let directive_end = &directive_start[directive_len..];
                    let import_name_start = if let Some(import_name_start) =
                        directive_end.find(|c: char| !c.is_whitespace())
                    {
                        import_name_start
                    } else {
                        eprintln!("#import needs a non_whitespace argument (line {line_number})");
                        continue 'outer;
                    };
                    let import_name_start = &directive_end[import_name_start..];
                    let import_name_end_index = import_name_start
                        .find(|c| !(matches!(c, '_') || c.is_alphanumeric()))
                        .unwrap_or(import_name_start.len());
                    let import_name = &import_name_start[..import_name_end_index];
                    line = &import_name_start[import_name_end_index..];
                    let import = imports.get(import_name);
                    if let Some(import) = import {
                        if stack.last().map(|x| x.active).unwrap_or(true) {
                            output.push_str(&preprocess(import, defines, imports));
                        }
                    } else {
                        eprintln!("Unknown import `{import_name}` (line {line_number})");
                    }
                    continue;
                }
                val => {
                    eprintln!("Unknown preprocessor directive `{val}` (line {line_number})");
                }
            }
        }
        if stack.last().map(|x| x.active).unwrap_or(true) {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}
