use anyhow::{bail, Result};
use generic_array::GenericArray;
use hex_literal::hex;
use sha2::{Digest, Sha256};
use std::env::temp_dir;
use std::io::Read;
use std::path::PathBuf;

pub struct SvgAsset {
    _source: &'static str,
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
    // DANGER: Zooming in on this image crashes my computer. @lemmih 2023-01-20.
    // SvgAsset {
    //     _source: "https://commons.wikimedia.org/wiki/File:American_Legion_Seal_SVG.svg",
    //     url: "https://upload.wikimedia.org/wikipedia/commons/c/cf/American_Legion_Seal_SVG.svg",
    //     sha256sum: hex!("b990f047a274b463a75433ddb9c917e90067615bba5ad8373a3f77753c6bb5e1"),
    //     license: "Public Domain",
    //     size: 10849279,
    // },

    SvgAsset {
        _source: "https://commons.wikimedia.org/wiki/File:CIA_WorldFactBook-Political_world.svg",
        url: "https://upload.wikimedia.org/wikipedia/commons/7/72/Political_Map_of_the_World_%28august_2013%29.svg",
        sha256sum: hex!("57956f10ed0ad3b1bea1e6c74cc7b386e42c99d87720a87c323d07f18c15d349"),
        license: "Public Domain",
        size: 12771150,
    },

    SvgAsset {
        _source: "https://commons.wikimedia.org/wiki/File:World_-_time_zones_map_(2014).svg",
        url: "https://upload.wikimedia.org/wikipedia/commons/c/c6/World_-_time_zones_map_%282014%29.svg",
        sha256sum: hex!("0cfecd5cdeadc51eb06f60c75207a769feb5b63abe20e4cd6c0d9fea30e07563"),
        license: "Public Domain",
        size: 5235172,
    },

    SvgAsset {
        _source: "https://commons.wikimedia.org/wiki/File:Coat_of_arms_of_Poland-official.svg",
        url: "https://upload.wikimedia.org/wikipedia/commons/3/3e/Coat_of_arms_of_Poland-official.svg",
        sha256sum: hex!("59b4d0e29adcd7ec6a7ab50af5796f1d13afc0334a6d4bd4d4099a345b0e3066"),
        license: "Public Domain",
        size: 10747708,
    },

    SvgAsset {
        _source: "https://commons.wikimedia.org/wiki/File:Coat_of_arms_of_the_Kingdom_of_Yugoslavia.svg",
        url: "https://upload.wikimedia.org/wikipedia/commons/5/58/Coat_of_arms_of_the_Kingdom_of_Yugoslavia.svg",
        sha256sum: hex!("2b1084dee535985eb241b14c9a5260129efe4c415c66dafa548b81117842d3e3"),
        license: "Public Domain",
        size: 12795806,
    },

    // This SVG renders poorly
    // SvgAsset {
    //     _source: "https://commons.wikimedia.org/wiki/File:Map_of_the_World_Oceans_-_January_2015.svg",
    //     url: "https://upload.wikimedia.org/wikipedia/commons/d/db/Map_of_the_World_Oceans_-_January_2015.svg",
    //     sha256sum: hex!("c8b0b13a577092bafa38b48b2fed28a1a26a91d237f4808444fa4bfee423c330"),
    //     license: "Public Domain",
    //     size: 10804504,
    // },
];
