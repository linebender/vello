pub mod download;
mod images;
mod mmark;
mod simple_text;
mod svg;
mod test_scenes;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Args, Subcommand};
use download::Download;
pub use images::ImageCache;
pub use simple_text::SimpleText;
pub use svg::{default_scene, scene_from_files};
pub use test_scenes::test_scenes;

use vello::{kurbo::Vec2, peniko::Color, SceneBuilder};

pub struct SceneParams<'a> {
    pub time: f64,
    /// Whether blocking should be limited
    /// Will not change between runs
    // TODO: Just never block/handle this automatically?
    pub interactive: bool,
    pub text: &'a mut SimpleText,
    pub images: &'a mut ImageCache,
    pub resolution: Option<Vec2>,
    pub base_color: Option<vello::peniko::Color>,
    pub complexity: usize,
}

pub struct SceneConfig {
    // TODO: This is currently unused
    pub animated: bool,
    pub name: String,
}

pub struct ExampleScene {
    pub function: Box<dyn TestScene>,
    pub config: SceneConfig,
}

pub trait TestScene {
    fn render(&mut self, sb: &mut SceneBuilder, params: &mut SceneParams);
}

impl<F: FnMut(&mut SceneBuilder, &mut SceneParams)> TestScene for F {
    fn render(&mut self, sb: &mut SceneBuilder, params: &mut SceneParams) {
        self(sb, params);
    }
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
    #[arg(long, global(false), value_parser = parse_color)]
    /// The base color applied as the blend background to the rasterizer.
    /// Format is CSS style hexidecimal (#RGB, #RGBA, #RRGGBB, #RRGGBBAA) or
    /// an SVG color name such as "aliceblue"
    pub base_color: Option<Color>,
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
        #[allow(unused)] command: impl FnOnce() -> clap::Command,
    ) -> Result<Option<SceneSet>> {
        if let Some(command) = &self.command {
            command.action()?;
            Ok(None)
        } else {
            // There is no file access on WASM, and on Android we haven't set up the assets
            // directory.
            // TODO: Upload the assets directory on Android
            // Therefore, only render the `test_scenes` (including one SVG example)
            #[cfg(any(target_arch = "wasm32", target_os = "android"))]
            return Ok(Some(test_scenes()));
            #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
            if self.test_scenes {
                Ok(test_scenes())
            } else if let Some(svgs) = &self.svgs {
                scene_from_files(svgs)
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

fn parse_color(s: &str) -> Result<Color> {
    Color::parse(s).ok_or(anyhow!("'{s}' is not a valid color"))
}
