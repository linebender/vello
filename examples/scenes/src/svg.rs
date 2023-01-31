use std::{
    fs::read_dir,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Ok, Result};
use vello::{kurbo::Vec2, SceneBuilder, SceneFragment};
use vello_svg::usvg;

use crate::{ExampleScene, SceneSet};

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
                        scenes.push(example_scene_of(entry.path()))
                    }
                }
            }
            // Ensure a consistent order within directories
            scenes[start_index..].sort_by_key(|scene| scene.config.name.to_lowercase());
            if count == 0 {
                empty_dir();
            }
        } else {
            scenes.push(example_scene_of(path.to_owned()))
        }
    }
    Ok(SceneSet { scenes })
}

fn example_scene_of(file: PathBuf) -> ExampleScene {
    let name = file
        .file_stem()
        .map(|it| it.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let name_stored = name.clone();
    let mut cached_scene = None;
    ExampleScene {
        function: Box::new(move |builder, params| {
            let (scene_frag, resolution) = cached_scene.get_or_insert_with(|| {
                let start = Instant::now();
                let contents = std::fs::read_to_string(&file).expect("failed to read svg file");
                let svg = usvg::Tree::from_str(&contents, &usvg::Options::default())
                    .expect("failed to parse svg file");
                eprintln!(
                    "Parsing SVG {name_stored} took {:?} (file `{file:?}`",
                    start.elapsed()
                );
                let mut new_scene = SceneFragment::new();
                let mut builder = SceneBuilder::for_fragment(&mut new_scene);
                vello_svg::render_tree(&mut builder, &svg);
                let resolution = Vec2::new(svg.size.width(), svg.size.height());
                (new_scene, resolution)
            });
            builder.append(&scene_frag, None);
            params.resolution = Some(*resolution);
        }),
        config: crate::SceneConfig {
            animated: false,
            name,
        },
    }
}
