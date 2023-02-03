use std::{
    io::Seek,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use byte_unit::Byte;
use clap::Args;
use std::io::Read;
mod default_downloads;

#[derive(Args, Debug)]
pub(crate) struct Download {
    #[clap(long)]
    /// Directory to download the files into
    #[clap(default_value_os_t = default_directory())]
    pub directory: PathBuf,
    /// Set of files to download. Use `name@url` format to specify a file prefix
    downloads: Option<Vec<String>>,
    /// Whether to automatically install the default set of files
    #[clap(long)]
    auto: bool,
    /// The size limit for each individual file (ignored if the default files are downloaded)
    #[clap(long, default_value = "10 MB")]
    size_limit: Byte,
}

fn default_directory() -> PathBuf {
    let mut result = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("assets");
    result.push("downloads");
    result
}

impl Download {
    pub fn action(&self) -> Result<()> {
        let mut to_download = vec![];
        if let Some(downloads) = &self.downloads {
            to_download = downloads
                .iter()
                .map(|it| Self::parse_download(&it))
                .collect();
        } else {
            let mut accepted = self.auto;
            let downloads = default_downloads::default_downloads()
                .into_iter()
                .filter(|it| {
                    let file = it.file_path(&self.directory);
                    !file.exists()
                })
                .collect::<Vec<_>>();
            if !accepted {
                if downloads.len() != 0 {
                    println!(
                        "Would you like to download a set of default svg files? These files are:"
                    );
                    for download in &downloads {
                        let builtin = download.builtin.as_ref().unwrap();
                        println!(
                            "{} ({}) under license {} from {}",
                            download.name,
                            byte_unit::Byte::from_bytes(builtin.expected_size.into())
                                .get_appropriate_unit(false),
                            builtin.license,
                            builtin.info
                        );
                    }

                    // For rustfmt, split prompt into its own line
                    const PROMPT: &str =
                "Would you like to download a set of default svg files, as explained above?";
                    accepted = dialoguer::Confirm::new()
                        .with_prompt(PROMPT)
                        .wait_for_newline(true)
                        .interact()?;
                } else {
                    println!("Nothing to download! All default downloads already created");
                }
            }
            if accepted {
                to_download = downloads;
            }
        }
        for (index, download) in to_download.iter().enumerate() {
            println!(
                "{index}: Downloading {} from {}",
                download.name, download.url
            );
            download.fetch(&self.directory, self.size_limit)?
        }
        println!("{} downloads complete", to_download.len());
        Ok(())
    }

    fn parse_download(value: &str) -> SVGDownload {
        if let Some(at_index) = value.find('@') {
            let name = &value[0..at_index];
            let url = &value[at_index + 1..];
            SVGDownload {
                name: name.to_string(),
                url: url.to_string(),
                builtin: None,
            }
        } else {
            let end_index = value.rfind(".svg").unwrap_or(value.len());
            let url_with_name = &value[0..end_index];
            let name = url_with_name
                .rfind('/')
                .map(|v| &url_with_name[v + 1..])
                .unwrap_or(url_with_name);
            SVGDownload {
                name: name.to_string(),
                url: value.to_string(),
                builtin: None,
            }
        }
    }
}

struct SVGDownload {
    name: String,
    url: String,
    builtin: Option<BuiltinSvgProps>,
}

impl SVGDownload {
    fn file_path(&self, directory: &Path) -> PathBuf {
        directory.join(&self.name).with_extension("svg")
    }

    fn fetch(&self, directory: &Path, size_limit: Byte) -> Result<()> {
        let mut size_limit = size_limit.get_bytes().try_into()?;
        let mut limit_exact = false;
        if let Some(builtin) = &self.builtin {
            size_limit = builtin.expected_size;
            limit_exact = true;
        }
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&self.file_path(directory))
            .context("Creating file")?;
        let mut reader = ureq::get(&self.url).call()?.into_reader();

        std::io::copy(
            // ureq::into_string() has a limit of 10MiB so we must use the reader
            &mut (&mut reader).take(size_limit),
            &mut file,
        )?;
        if reader.read_exact(&mut [0]).is_ok() {
            bail!("Size limit exceeded");
        }
        if limit_exact {
            if file
                .seek(std::io::SeekFrom::Current(0))
                .context("Checking file limit")?
                != size_limit
            {
                bail!("Builtin downloaded file was not as expected");
            }
        }
        Ok(())
    }
}

struct BuiltinSvgProps {
    expected_size: u64,
    license: &'static str,
    info: &'static str,
}
