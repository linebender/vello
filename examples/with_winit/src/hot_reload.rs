// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::time::Duration;

use anyhow::Result;
use notify_debouncer_mini::notify::*;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult};

pub fn hot_reload(mut f: impl FnMut() -> Option<()> + Send + 'static) -> Result<impl Sized> {
    let mut debouncer = new_debouncer(
        Duration::from_millis(500),
        move |res: DebounceEventResult| match res {
            Ok(_) => f().unwrap(),
            Err(e) => println!("Hot reloading file watching failed: {e:?}"),
        },
    )?;

    debouncer.watcher().watch(
        vello_shaders::compile::shader_dir().as_path(),
        // We currently don't support hot reloading the imports, so don't recurse into there
        RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
}
