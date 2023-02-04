use std::{path::Path, time::Duration};

use anyhow::Result;
use notify_debouncer_mini::{new_debouncer, notify::*, DebounceEventResult};

pub(crate) fn hot_reload(mut f: impl FnMut() -> Option<()> + Send + 'static) -> Result<impl Sized> {
    let mut debouncer = new_debouncer(
        Duration::from_millis(500),
        None,
        move |res: DebounceEventResult| match res {
            Ok(_) => f().unwrap(),
            Err(errors) => errors.iter().for_each(|e| println!("Error {:?}", e)),
        },
    )?;

    debouncer.watcher().watch(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../shader")
            .canonicalize()?,
        // We currently don't support hot reloading the imports, so don't recurse into there
        RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
}
