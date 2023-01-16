use std::{path::Path, time::Duration};

use notify_debouncer_mini::{new_debouncer, notify::*, DebounceEventResult};

pub(crate) fn hot_reload(mut f: impl FnMut() -> Option<()> + Send + 'static) -> impl Sized {
    let mut debouncer = new_debouncer(
        Duration::from_millis(500),
        None,
        move |res: DebounceEventResult| match res {
            Ok(_) => f().unwrap(),
            Err(errors) => errors.iter().for_each(|e| println!("Error {:?}", e)),
        },
    )
    .unwrap();

    debouncer
        .watcher()
        .watch(
            dbg!(&Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../shader")
                .canonicalize()
                .unwrap()),
            // We currently don't support hot reloading the imports, so don't recurse into there
            RecursiveMode::NonRecursive,
        )
        .expect("Could watch shaders directory");
    debouncer
}
