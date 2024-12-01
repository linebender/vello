// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs::read_dir;
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use anyhow::Result;
use vello::{
    kurbo::{Affine, Rect, Stroke, Vec2},
    peniko::{color::palette, Fill},
    Scene,
};

use crate::{ExampleScene, SceneParams, SceneSet};

pub fn scene_from_files(files: &[PathBuf]) -> Result<SceneSet> {
    scene_from_files_inner(files)
}

pub fn default_scene() -> Result<SceneSet> {
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../assets/")
        .canonicalize()?;
    scene_from_files_inner(&[
        assets_dir.join("Ghostscript_Tiger.svg"),
        assets_dir.join("downloads"),
    ])
}

fn scene_from_files_inner(files: &[PathBuf]) -> std::result::Result<SceneSet, anyhow::Error> {
    let mut scenes = Vec::new();
    for path in files {
        if path.is_dir() {
            let start_index = scenes.len();
            for file in read_dir(path)? {
                let entry = file?;
                if let Some(extension) = Path::new(&entry.file_name()).extension() {
                    if extension == "svg" {
                        scenes.push(example_scene_of(entry.path()));
                    }
                }
            }
            // Ensure a consistent order within directories
            scenes[start_index..].sort_by_key(|scene| scene.config.name.to_lowercase());
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

fn render_svg_rec(items: &[crate::pico_svg::Item]) -> Scene {
    let mut scene = Scene::new();
    for item in items {
        use crate::pico_svg::Item;
        match item {
            Item::Fill(fill) => {
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    fill.color,
                    None,
                    &fill.path,
                );
            }
            Item::Stroke(stroke) => {
                scene.stroke(
                    &Stroke::new(stroke.width),
                    Affine::IDENTITY,
                    stroke.color,
                    None,
                    &stroke.path,
                );
            }
            Item::Group(group) => {
                let child_scene = render_svg_rec(&group.children);
                scene.append(&child_scene, Some(group.affine));
            }
        }
    }
    scene
}

pub fn svg_function_of<R: AsRef<str>>(
    name: String,
    contents: impl FnOnce() -> R + Send + 'static,
) -> impl FnMut(&mut Scene, &mut SceneParams) {
    fn render_svg_contents(name: &str, contents: &str) -> (Scene, Vec2) {
        use crate::pico_svg::*;
        let start = Instant::now();
        match PicoSvg::load(contents, 1.0) {
            Ok(svg) => {
                eprintln!("Parsed svg {name} in {:?}", start.elapsed());
                let start = Instant::now();
                let scene = render_svg_rec(&svg.items);
                eprintln!("Encoded svg {name} in {:?}", start.elapsed());
                (scene, svg.size.to_vec2())
            }
            Err(e) => {
                eprintln!("Failed to load svg: {e}");
                let mut error_scene = Scene::new();
                error_scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    palette::css::FUCHSIA,
                    None,
                    &Rect::new(0.0, 0.0, 1.0, 1.0),
                );
                (error_scene, Vec2::new(1.0, 1.0))
            }
        }
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
                    Affine::translate((110.0, 600.0)),
                    &format!("Loading {name}"),
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!()
                }
            }
        };
    }
}
