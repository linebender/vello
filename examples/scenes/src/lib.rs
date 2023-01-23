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

use vello::{kurbo::Vec2, SceneBuilder};

pub struct SceneParams<'a> {
    pub time: f64,
    pub text: &'a mut SimpleText,
    pub resolution: Option<Vec2>,
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
/// Shared config for scene selection
pub struct Arguments {
    #[arg(help_heading = "Scene Selection")]
    #[arg(short = 't', long, global(false))]
    test_scenes: bool,
    #[arg(help_heading = "Scene Selection", global(false))]
    /// The svg files to render
    svgs: Option<Vec<PathBuf>>,
    #[clap(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Download SVG files for testing. By default, downloads a set of files from wikipedia
    Download(Download),
}

impl Arguments {
    pub fn select_scene_set(
        &self,
        command: impl FnOnce() -> clap::Command,
    ) -> Result<Option<SceneSet>> {
        if let Some(command) = &self.command {
            command.action()?;
            Ok(None)
        } else {
            if self.test_scenes {
                Ok(test_scenes())
            } else if let Some(svgs) = &self.svgs {
                scene_from_files(&svgs)
            } else {
                default_scene(command)
            }
            .map(Some)
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
