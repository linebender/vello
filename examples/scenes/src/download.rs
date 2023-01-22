use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;
use std::io::Read;
mod default_downloads;

#[derive(Args, Debug)]
pub(crate) struct Download {
    #[clap(long)]
    pub directory: Option<PathBuf>,
    downloads: Option<Vec<String>>,
    #[clap(long)]
    auto: bool,
    #[clap(long, default_value_t = 10_000_000)]
    size_limit: u64,
}

fn default_directory() -> Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../assets/downloads")
        .canonicalize()?)
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
            let downloads = default_downloads::default_downloads();
            if !accepted {
                println!("Would you like to download a set of default svg files? These files are:");
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
            }
            if accepted {
                to_download = downloads;
            }
        }
        let directory = &self.directory.clone().unwrap_or(default_directory()?);
        for (index, download) in to_download.iter().enumerate() {
            println!(
                "{index}: Downloading {} from {}",
                download.name, download.url
            );
            download.fetch(&directory, self.size_limit)?
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
    fn fetch(&self, directory: &Path, size_limit: u64) -> Result<()> {
        // ureq::into_string() has a limit of 10MiB so let's use the reader directly:
        let mut size_limit = size_limit;
        let mut buf: Vec<u8> = Vec::new();
        if let Some(builtin) = &self.builtin {
            size_limit = builtin.expected_size;
            buf.reserve(size_limit.try_into()?);
        }
        ureq::get(&self.url)
            .call()?
            .into_reader()
            .take(size_limit)
            .read_to_end(&mut buf)?;
        let body: String = String::from_utf8_lossy(&buf).to_string();

        std::fs::write(directory.join(&self.name).with_extension(".svg"), &body)?;
        Ok(())
    }
}

struct BuiltinSvgProps {
    expected_size: u64,
    license: &'static str,
    info: &'static str,
}
