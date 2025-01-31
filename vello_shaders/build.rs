// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build step.

// These modules are also included in the main crate, where the items are reachable
#[allow(warnings, reason = "Checked elsewhere")]
#[path = "src/compile/mod.rs"]
mod compile;
#[allow(warnings, reason = "Checked elsewhere")]
#[path = "src/types.rs"]
mod types;

use std::env;
use std::fmt::Write;
use std::path::Path;

use compile::ShaderInfo;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("shaders.rs");

    println!("cargo:rerun-if-changed={}", compile::shader_dir().display());

    let mut shaders = match ShaderInfo::from_default() {
        Ok(s) => s,
        Err(err) => {
            let formatted = err.to_string();
            for line in formatted.lines() {
                println!("cargo:warning={line}");
            }
            return;
        }
    };

    // Drop the HashMap and sort by name so that we get deterministic order.
    let mut shaders = shaders.drain().collect::<Vec<_>>();
    shaders.sort_by(|x, y| x.0.cmp(&y.0));
    let mut buf = String::default();
    write_types(&mut buf, &shaders).unwrap();
    write_shaders(&mut buf, &shaders).unwrap();
    std::fs::write(dest_path, &buf).unwrap();
}

fn write_types(buf: &mut String, shaders: &[(String, ShaderInfo)]) -> Result<(), std::fmt::Error> {
    writeln!(buf, "pub struct Shaders<'a> {{")?;
    for (name, _) in shaders {
        writeln!(buf, "    pub {name}: ComputeShader<'a>,")?;
    }
    writeln!(buf, "}}")?;
    Ok(())
}

fn write_shaders(
    buf: &mut String,
    shaders: &[(String, ShaderInfo)],
) -> Result<(), std::fmt::Error> {
    writeln!(buf, "mod generated {{")?;
    writeln!(buf, "    use super::*;")?;
    writeln!(buf, "    use BindType::*;")?;
    writeln!(buf, "    pub const SHADERS: Shaders<'static> = Shaders {{")?;
    for (name, info) in shaders {
        let bind_tys = info
            .bindings
            .iter()
            .map(|binding| binding.ty)
            .collect::<Vec<_>>();
        let wg_bufs = &info.workgroup_buffers;
        writeln!(buf, "        {name}: ComputeShader {{")?;
        writeln!(buf, "            name: Cow::Borrowed({:?}),", name)?;
        writeln!(
            buf,
            "            workgroup_size: {:?},",
            info.workgroup_size
        )?;
        writeln!(buf, "            bindings: Cow::Borrowed(&{:?}),", bind_tys)?;
        writeln!(
            buf,
            "            workgroup_buffers: Cow::Borrowed(&{:?}),",
            wg_bufs
        )?;
        if cfg!(feature = "wgsl") {
            let indices = info
                .bindings
                .iter()
                .map(|binding| binding.location.1)
                .collect::<Vec<_>>();
            writeln!(buf, "            wgsl: WgslSource {{")?;
            writeln!(
                buf,
                "                code: Cow::Borrowed({:?}),",
                info.source
            )?;
            writeln!(
                buf,
                "                binding_indices : Cow::Borrowed(&{:?}),",
                indices
            )?;
            writeln!(buf, "            }},")?;
        }
        if cfg!(feature = "msl") {
            write_msl(buf, info)?;
        }
        writeln!(buf, "        }},")?;
    }
    writeln!(buf, "    }};")?;
    writeln!(buf, "}}")?;
    Ok(())
}

#[cfg(not(feature = "msl"))]
fn write_msl(_: &mut String, _: &ShaderInfo) -> Result<(), std::fmt::Error> {
    Ok(())
}

#[cfg(feature = "msl")]
fn write_msl(buf: &mut String, info: &ShaderInfo) -> Result<(), std::fmt::Error> {
    let mut index_iter = compile::msl::BindingIndexIterator::default();
    let indices = info
        .bindings
        .iter()
        .map(|binding| index_iter.next(binding.ty))
        .collect::<Vec<_>>();
    writeln!(buf, "            msl: MslSource {{")?;
    writeln!(
        buf,
        "                code: Cow::Borrowed({:?}),",
        compile::msl::translate(info).unwrap()
    )?;
    writeln!(
        buf,
        "                binding_indices : Cow::Borrowed(&{:?}),",
        indices
    )?;
    writeln!(buf, "            }},")?;
    Ok(())
}
