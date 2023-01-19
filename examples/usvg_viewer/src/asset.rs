use anyhow::{bail, Result};
use generic_array::GenericArray;
use hex_literal::hex;
use sha2::{Digest, Sha256};
use std::env::temp_dir;
use std::io::{self, Cursor, Read};
use std::path::PathBuf;

pub struct SvgAsset {
    url: &'static str,
    sha256sum: [u8; 32],
    pub license: &'static str,
    pub size: u128,
}

impl SvgAsset {
    pub fn local_path(&self) -> PathBuf {
        let arr = GenericArray::from(self.sha256sum);
        temp_dir()
            .join(format!("vello-asset-{:x}", arr))
            .with_extension("svg")
    }

    pub fn fetched(&self) -> bool {
        let resource_local_path = self.local_path();
        if let Ok(contents) = std::fs::read_to_string(&resource_local_path) {
            if Sha256::digest(contents)[..] == self.sha256sum {
                return true;
            }
        }
        false
    }

    pub fn fetch(&self) -> Result<()> {
        // ureq::into_string() has a limit of 10MiB so let's use the reader directly:
        let mut buf: Vec<u8> = Vec::with_capacity(self.size as usize);
        ureq::get(self.url)
            .call()?
            .into_reader()
            .take(self.size as u64)
            .read_to_end(&mut buf)?;
        let body: String = String::from_utf8_lossy(&buf).to_string();

        if Sha256::digest(&body)[..] != self.sha256sum {
            bail!(format!("Invalid sha256 hash for resource: {}", self.url))
        }

        std::fs::write(self.local_path(), &body)?;
        Ok(())
    }
}

pub const ASSETS: &[SvgAsset] = &[
    SvgAsset {
        url: "https://upload.wikimedia.org/wikipedia/commons/d/d2/Grand_Arms_of_Francis_II%2C_Holy_Roman_Emperor-Personal_%281804-1806%29.svg",
        sha256sum: hex!("240a7bb124cd4686f61fae132fdf762c07733c0be488aae845fde1845c76a41c"),
        license: "Public Domain",
        size: 9171285,
    },
    SvgAsset {
        url: "https://upload.wikimedia.org/wikipedia/commons/7/72/Political_Map_of_the_World_%28august_2013%29.svg",
        sha256sum: hex!("57956f10ed0ad3b1bea1e6c74cc7b386e42c99d87720a87c323d07f18c15d349"),
        license: "Public Domain",
        size: 12771150,
    },
    // SvgAsset {
    //     url: "https://upload.wikimedia.org/wikipedia/commons/c/c2/Garter_King_of_Arms.svg",
    //     sha256sum: hex!("6718e2c7cb3831cdebce9c7e1151fc397e1b774d6391b773654b86b2699e64a1"),
    //     license: "Public Domain",
    //     size: 13535073,
    // },
];
