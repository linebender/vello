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
    mut input: &str,
    defines: &HashSet<String>,
    imports: &HashMap<String, String>,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut stack: Vec<StackItem> = vec![];

    while let Some(hash_or_slash_index) = input.find(|char| matches!(char, '#' | '/')) {
        // N.B. `#` is a single ascii character
        let previous = &input[..hash_or_slash_index];
        if stack.last().map(|x| x.active).unwrap_or(true) {
            output.push_str(&previous);
        }

        let remaining = &input[hash_or_slash_index..];
        if remaining.starts_with("//") {
            let len = remaining
                .find('\n')
                .map_or(remaining.len(), |val| val + '\n'.len_utf8());
            output.push_str(&remaining[..len]);
            input = &remaining[len..];
            continue;
        } else if remaining.starts_with('/') {
            // Unary divide, ignore
            output.push('/');
            input = &remaining['/'.len_utf8()..];
            continue;
        }
        assert!(remaining.starts_with('#'));
        let input_start_directive = &remaining[('#'.len_utf8())..];

        let can_be_if = previous.len() == 0
            || previous
                // Don't bother with carriage return support, or any of the fancier line breaks
                .trim_end_matches(|char| !matches!(char, '\n') && char.is_whitespace())
                .ends_with('\n');

        let directive_len = input_start_directive
            .chars()
            .take_while(|char| char.is_alphanumeric())
            .map(char::len_utf8)
            .sum();
        let input_end_directive = &input_start_directive[directive_len..];
        input = &input_end_directive;
        let arg = || {
            let first_arg_char = input_end_directive
                .find(char::is_alphanumeric)
                .unwrap_or(input_end_directive.len());
            let input_arg_start = &input_end_directive[first_arg_char..];
            let arg_len = input_arg_start
                .chars()
                .take_while(|c| c.is_alphabetic() || matches!(c, '_'))
                .map(char::len_utf8)
                .sum();
            let arg = &input_arg_start[..arg_len];
            let input_arg_end = &input_arg_start[arg_len..];
            (arg, input_arg_end)
        };
        match &input_start_directive[..directive_len] {
            if_type @ ("endif" | "ifdef" | "ifndef") if !can_be_if => {
                panic!("Preprocessor directive {if_type} must be the first non-whitespace symbols in a line");
            }
            ifdef_symbol @ ("ifdef" | "ifndef") => {
                let (arg, new_input) = arg();
                let exists = defines.contains(arg);
                let mode = ifdef_symbol == "ifdef";
                stack.push(StackItem {
                    active: mode == exists,
                    else_passed: false,
                });
                input = new_input;
            }
            "else" => {
                let item = stack.last_mut();
                if let Some(item) = item {
                    if item.else_passed {
                        eprintln!("Second else for same ifdef/ifndef; ignoring second else")
                    } else {
                        item.else_passed = true;
                        item.active = !item.active;
                    }
                }
            }
            "endif" => {
                if let None = stack.pop() {
                    eprintln!("Mismatched endif");
                }
            }
            "import" => {
                let (arg, new_input) = arg();
                let import = imports.get(arg);
                if let Some(import) = import {
                    output.push_str(&preprocess(import, defines, imports));
                } else {
                    eprintln!("Unkown import `{arg}`");
                }
                input = new_input;
            }
            val => {
                eprintln!("Unknown preprocessor directive `{val}`")
            }
        }
    }
    assert!(stack.len() == 0, "ifdef and ifndef must be balanced");
    output.push_str(&input);
    // println!("{output} \n END OF SHADER \n");
    output
}
