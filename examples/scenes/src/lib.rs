pub mod download;
mod simple_text;
mod svg;
mod test_scenes;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};
use download::Download;
pub use simple_text::SimpleText;
pub use svg::{default_scene, scene_from_files};
pub use test_scenes::test_scenes;

use vello::SceneBuilder;

pub struct SceneParams<'a> {
    pub time: f64,
    pub text: &'a mut SimpleText,
}

pub struct SceneConfig {
    // TODO: This is currently unused
    pub animated: bool,
    pub name: String,
}

pub struct ExampleScene {
    pub function: Box<dyn FnMut(&mut SceneBuilder, &mut SceneParams)>,
    pub config: SceneConfig,
}

pub struct SceneSet {
    pub scenes: Vec<ExampleScene>,
}

#[derive(Args, Debug)]
pub struct Arguments {
    #[arg(short = 't', long)]
    test_scenes: bool,
    svgs: Option<Vec<PathBuf>>,
    #[clap(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Download(Download),
}

impl Arguments {
    pub fn select_scene_set(&self, command: impl FnOnce() -> clap::Command) -> Result<SceneSet> {
        if let Some(command) = &self.command {
            command.action()?
        }
        if self.test_scenes {
            Ok(test_scenes())
        } else if let Some(svgs) = &self.svgs {
            scene_from_files(&svgs)
        } else {
            default_scene(command)
        }
    }
}

impl Command {
    fn action(&self) -> Result<()> {
        match self {
            Command::Download(download) => download.action(),
        }
    }
}
