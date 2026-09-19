// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;

#[derive(Debug)]
pub struct Permutation {
    /// The new name for the permutation
    pub name: String,
    /// Set of defines to apply for the permutation
    pub defines: Vec<String>,
}

pub fn parse(source: &str) -> HashMap<String, Vec<Permutation>> {
    let mut map: HashMap<String, Vec<Permutation>> = HashMap::default();
    let mut current_source: Option<String> = None;
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(line) = line.strip_prefix('+') {
            if let Some(current_source) = &current_source {
                let mut parts = line.split(':').map(|s| s.trim());
                let Some(name) = parts.next() else {
                    continue;
                };
                let mut defines = vec![];
                if let Some(define_list) = parts.next() {
                    defines.extend(define_list.split(' ').map(|s| s.trim().to_string()));
                }
                map.entry(current_source.to_string())
                    .or_default()
                    .push(Permutation {
                        name: name.to_string(),
                        defines,
                    });
            }
        } else {
            current_source = Some(line.to_string());
        }
    }
    map
}
