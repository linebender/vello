// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs::read_dir;
use std::path::{Path, PathBuf};

use anyhow::{Ok, Result};
use instant::Instant;

use vello::{
    kurbo::{Affine, Stroke, Vec2},
    peniko::Fill,
    Scene,
};

use crate::{ExampleScene, SceneParams, SceneSet};

pub fn scene_from_files(files: &[PathBuf]) -> Result<SceneSet> {
    scene_from_files_inner(files, || ())
}

pub fn default_scene(command: impl FnOnce() -> clap::Command) -> Result<SceneSet> {
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../assets/")
        .canonicalize()?;
    let mut has_empty_directory = false;
    let result = scene_from_files_inner(
        &[
            assets_dir.join("Ghostscript_Tiger.svg"),
            assets_dir.join("downloads"),
        ],
        || has_empty_directory = true,
    )?;
    if has_empty_directory {
        let mut command = command();
        command.build();
        println!(
            "No test files have been downloaded. Consider downloading some using the subcommand:"
        );
        let subcmd = command.find_subcommand_mut("download").unwrap();
        subcmd.print_help()?;
    }
    Ok(result)
}

fn scene_from_files_inner(
    files: &[PathBuf],
    mut empty_dir: impl FnMut(),
) -> std::result::Result<SceneSet, anyhow::Error> {
    let mut scenes = Vec::new();
    for path in files {
        if path.is_dir() {
            let mut count = 0;
            let start_index = scenes.len();
            for file in read_dir(path)? {
                let entry = file?;
                if let Some(extension) = Path::new(&entry.file_name()).extension() {
                    if extension == "svg" {
                        count += 1;
                        scenes.push(example_scene_of(entry.path()));
                    }
                }
            }
            // Ensure a consistent order within directories
            scenes[start_index..].sort_by_key(|scene| scene.config.name.to_lowercase());
            if count == 0 {
                empty_dir();
            }
        } else {
            scenes.push(example_scene_of(path.to_owned()));
        }
    }
    Ok(SceneSet { scenes })
}

fn example_scene_of(file: PathBuf) -> ExampleScene {
    let name = file
        .file_stem()
        .map(|it| it.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    ExampleScene {
        function: Box::new(svg_function_of(name.clone(), move || {
            std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("failed to read svg file {file:?}: {e}"))
        })),
        config: crate::SceneConfig {
            animated: false,
            name,
        },
    }
}

pub fn svg_function_of<R: AsRef<str>>(
    name: String,
    contents: impl FnOnce() -> R + Send + 'static,
) -> impl FnMut(&mut Scene, &mut SceneParams) {
    fn render_svg_contents(name: &str, contents: &str) -> (Scene, Vec2) {
        use crate::pico_svg::*;
        let mut start = Instant::now();
        let mut new_scene = Scene::new();
        let mut resolution = Vec2::new(420 as f64, 420 as f64);
        match PicoSvg::load(contents, 1.0) {
            std::result::Result::Ok(PicoSvg {
                items,
                origin,
                size,
            }) => {
                eprintln!("Parsed svg {name} in {:?}", start.elapsed());
                start = Instant::now();
                resolution = size.to_vec2();
                let transform = Affine::translate(origin.to_vec2() * -1.0);
                for item in items {
                    match item {
                        Item::Fill(fill) => {
                            new_scene.fill(Fill::NonZero, transform, fill.color, None, &fill.path);
                        }
                        Item::Stroke(stroke) => {
                            new_scene.stroke(
                                &Stroke::new(stroke.width),
                                transform,
                                stroke.color,
                                None,
                                &stroke.path,
                            );
                        }
                    }
                }
            }
            std::result::Result::Err(e) => {
                eprintln!("{:?}", e);
            }
        }

        eprintln!("Encoded svg {name} in {:?}", start.elapsed());
        (new_scene, resolution)
    }
    let mut cached_scene = None;
    #[cfg(not(target_arch = "wasm32"))]
    let (tx, rx) = std::sync::mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    let mut tx = Some(tx);
    #[cfg(not(target_arch = "wasm32"))]
    let mut has_started_parse = false;
    let mut contents = Some(contents);
    move |scene, params| {
        if let Some((scene_frag, resolution)) = cached_scene.as_mut() {
            scene.append(scene_frag, None);
            params.resolution = Some(*resolution);
            return;
        }
        if cfg!(target_arch = "wasm32") || !params.interactive {
            let contents = contents.take().unwrap();
            let contents = contents();
            let (scene_frag, resolution) = render_svg_contents(&name, contents.as_ref());
            scene.append(&scene_frag, None);
            params.resolution = Some(resolution);
            cached_scene = Some((scene_frag, resolution));
            #[cfg_attr(target_arch = "wasm32", allow(clippy::needless_return))]
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut timeout = std::time::Duration::from_millis(10);
            if !has_started_parse {
                has_started_parse = true;
                // Prefer jank over loading screen for first time
                timeout = std::time::Duration::from_millis(75);
                let tx = tx.take().unwrap();
                let contents = contents.take().unwrap();
                let name = name.clone();
                std::thread::spawn(move || {
                    let contents = contents();
                    tx.send(render_svg_contents(&name, contents.as_ref()))
                        .unwrap();
                });
            }
            let recv = rx.recv_timeout(timeout);
            use std::sync::mpsc::RecvTimeoutError;
            match recv {
                Result::Ok((scene_frag, resolution)) => {
                    scene.append(&scene_frag, None);
                    params.resolution = Some(resolution);
                    cached_scene = Some((scene_frag, resolution));
                }
                Err(RecvTimeoutError::Timeout) => params.text.add(
                    scene,
                    None,
                    48.,
                    None,
                    vello::kurbo::Affine::translate((110.0, 600.0)),
                    &format!("Loading {name}"),
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!()
                }
            }
        };
    }
}
