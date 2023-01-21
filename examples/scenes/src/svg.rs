use std::{
    fs::read_dir,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Ok, Result};
use vello::{SceneBuilder, SceneFragment};
use vello_svg::usvg;

use crate::{ExampleScene, SceneSet};

pub fn scene_from_files(files: &[PathBuf]) -> Result<SceneSet> {
    scene_from_files_inner(files, || ())
}

pub fn default_scene(command: impl FnOnce() -> clap::Command) -> Result<SceneSet> {
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/")
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
        let subcmd = command.find_subcommand_mut("download").unwrap();
        println!(
            "No test files have been downloaded. Consider downloading some using the subcommand:"
        );
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
            for file in read_dir(path)? {
                count += 1;
                let entry = file?;
                if let Some(extension) = Path::new(&entry.file_name()).extension() {
                    if extension == "svg" {
                        scenes.push(example_scene_of(entry.path()))
                    }
                }
            }
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
        function: Box::new(move |builder, _| {
            let scene_frag = cached_scene.get_or_insert_with(|| {
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
                new_scene
            });
            builder.append(&scene_frag, None);
        }),
        config: crate::SceneConfig {
            animated: false,
            name,
        },
    }
}
