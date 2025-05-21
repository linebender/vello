// Copyright 2025 the Parley Authors
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
    /// Command
    #[clap(subcommand)]
    pub command: CliCommand,
}

#[derive(Parser, Debug)]
/// Top-level xtask command
pub enum CliCommand {
    /// Commands related to snapshots (current vs. reference)
    Snapshots(Args),
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
/// because only report command makes, we do not reuse `kompari_tasks::Args`
pub enum ComparisonsCommand {
    /// Create report with differences between cpu/gpu versions
    Report(ReportArgs),
}

impl Actions for ActionsImpl {
    fn generate_all_tests(&self) -> kompari::Result<()> {
        let cargo = std::env::var("CARGO").unwrap();
        Command::new(&cargo)
            .arg("test")
            .env("VELLO_TEST_GENERATE_ALL", "1")
            .status()?;
        Ok(())
    }
}

fn snapshots_command(args: Args) -> kompari::Result<()> {
    let tests_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("vello_tests");

    let snapshots_path = tests_path.join("snapshots");
    let current_path = tests_path.join("current");

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
    let gpu_path = tests_path.join("gpu");

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
        CliCommand::Snapshots(args) => snapshots_command(args),
        CliCommand::Comparisons(args) => comparisons_command(args),
    }
}
