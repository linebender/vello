// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShaderInfo {
    pub(crate) name: String,
    pub(crate) wgsl_source: String,
}

pub(crate) fn load_shader_infos(shader_dir: &Path) -> io::Result<Vec<ShaderInfo>> {
    let shader_paths = fs::read_dir(shader_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "wgsl" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut shader_infos = Vec::new();
    for path in shader_paths {
        let file_stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let source = fs::read_to_string(&path)?;
        shader_infos.push(ShaderInfo {
            name: file_stem,
            wgsl_source: source,
        });
    }
    shader_infos.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(shader_infos)
}
