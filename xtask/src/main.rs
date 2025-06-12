// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Xtask utilities for Vello, currently only integrates Kompari

use clap::Parser;
use kompari::DirDiffConfig;
use kompari_tasks::args::{Command as KompariCommand, ReportArgs};
use kompari_tasks::{Actions, Args, Task};
use std::path::Path;
use std::process::Command;

struct ActionsImpl();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
/// Top-level command line parser for xtask
pub struct Cli {
    /// The possible commands in this CLI.
    /// This enables (future) global flags to be added to this struct
    #[clap(subcommand)]
    pub command: CliCommand,
}

#[derive(Parser, Debug)]
/// Top-level xtask command
pub enum CliCommand {
    /// Commands related to snapshots (current vs. reference)
    SnapshotsCpu(Args),
    /// Commands related to snapshots (current vs. reference)
    SnapshotsGpu(Args),
    /// Commands related to comparisons (cpu vs. gpu)
    Comparisons(ComparisonsArgs),
}

#[derive(Parser, Debug)]
/// CLI parser for comparisons
pub struct ComparisonsArgs {
    /// Command
    #[clap(subcommand)]
    pub command: ComparisonsCommand,
}

#[derive(Parser, Debug)]
/// Command for comparisons,
/// in comparisons there is no ground truth images, so no other command then "report" makes sense.
pub enum ComparisonsCommand {
    /// Create report with differences between cpu/gpu versions
    Report(ReportArgs),
}

impl Actions for ActionsImpl {
    fn generate_all_tests(&self) -> kompari::Result<()> {
        let cargo = std::env::var("CARGO").unwrap();
        Command::new(&cargo)
            .arg("nextest")
            .arg("run")
            .env("VELLO_TEST_GENERATE_ALL", "1")
            .status()?;
        Ok(())
    }
}

fn snapshots_command(dir: &str, args: Args) -> kompari::Result<()> {
    let tests_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("vello_tests");

    let snapshots_path = tests_path.join("snapshots");
    let current_path = tests_path.join("current").join(dir);

    let mut diff_config = DirDiffConfig::new(snapshots_path, current_path);
    diff_config.set_ignore_right_missing(true);
    let actions = ActionsImpl();
    let mut task = Task::new(diff_config, Box::new(actions));
    task.run(&args)?;
    Ok(())
}

fn comparisons_command(args: ComparisonsArgs) -> kompari::Result<()> {
    let tests_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("vello_tests")
        .join("comparisons");

    let cpu_path = tests_path.join("cpu");
    std::fs::create_dir_all(&cpu_path)?;
    let gpu_path = tests_path.join("gpu");
    std::fs::create_dir_all(&gpu_path)?;

    let diff_config = DirDiffConfig::new(cpu_path, gpu_path);
    let actions = ActionsImpl();
    let mut task = Task::new(diff_config, Box::new(actions));
    task.report_config().set_left_title("cpu");
    task.report_config().set_right_title("gpu");
    match args.command {
        ComparisonsCommand::Report(args) => {
            task.run(&Args {
                command: KompariCommand::Report(args),
            })?;
        }
    }
    Ok(())
}

fn main() -> kompari::Result<()> {
    let args = Cli::parse();
    match args.command {
        CliCommand::SnapshotsCpu(args) => snapshots_command("cpu", args),
        CliCommand::SnapshotsGpu(args) => snapshots_command("gpu", args),
        CliCommand::Comparisons(args) => comparisons_command(args),
    }
}
