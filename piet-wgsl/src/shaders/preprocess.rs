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

    for (line_number, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("#") {
            let val_idx = trimmed
                .chars()
                .take_while(|char| char.is_alphanumeric())
                .map(char::len_utf8)
                .sum();
            let arg = trimmed[val_idx..].trim();
            match &trimmed[..val_idx] {
                x @ ("ifdef" | "ifndef") => {
                    let exists = defines.contains(arg);
                    let mode = x == "ifdef";
                    stack.push(StackItem {
                        active: mode == exists,
                        else_passed: false,
                    });
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
                }
                "endif" => {
                    if let None = stack.pop() {
                        eprintln!("Mismatched endif (line {line_number})");
                    }
                }
                "import" => {
                    let import = imports.get(arg);
                    if let Some(import) = import {
                        output.push_str(&preprocess(import, defines, imports));
                    } else {
                        eprintln!("Unkown import `{arg}` (line {line_number})");
                    }
                }
                val => eprintln!("Unknown preprocessor directive `{val}` (line {line_number})"),
            }
        } else {
            if stack.last().map(|x| x.active).unwrap_or(true) {
                output.push_str(line);
            }
        }
    }
    output
}
