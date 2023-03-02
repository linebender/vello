pub mod download;
mod simple_text;
#[cfg(not(target_arch = "wasm32"))]
mod svg;
mod test_scenes;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Args, Subcommand};
use download::Download;
pub use simple_text::SimpleText;
#[cfg(not(target_arch = "wasm32"))]
pub use svg::{default_scene, scene_from_files};
pub use test_scenes::test_scenes;

use vello::{kurbo::Vec2, peniko::Color, SceneBuilder};

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
    #[arg(long, global(false))]
    /// Whether to use the test scenes created by code
    test_scenes: bool,
    #[arg(help_heading = "Scene Selection", global(false))]
    /// The svg files paths to render
    svgs: Option<Vec<PathBuf>>,
    #[arg(help_heading = "Render Parameters")]
    #[arg(long, global(false))]
    /// The base color applied as the blend background to the rasterizer.
    /// Format is CSS style hexidecimal (#RGB, #RGBA, #RRGGBB, #RRGGBBAA) or
    /// an SVG color name such as "aliceblue"
    base_color: Option<String>,
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
            // There is no file access on WASM
            #[cfg(target_arch = "wasm32")]
            return Ok(Some(test_scenes()));
            #[cfg(not(target_arch = "wasm32"))]
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

    pub fn get_base_color(&self) -> Result<Option<Color>> {
        self.base_color.as_ref().map_or_else(
            || Ok(None),
            |s| {
                Color::parse(&s)
                    .ok_or_else(|| anyhow!("malformed color: {}", s))
                    .map(Some)
            },
        )
    }
}

impl Command {
    fn action(&self) -> Result<()> {
        match self {
            Command::Download(download) => download.action(),
        }
    }
}
