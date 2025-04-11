// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Code related to managing wgpu pipeline caches, especially on Android

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use vello::wgpu::{self, AdapterInfo, Device, Features, PipelineCache};
use winit::event_loop::EventLoop;

#[cfg(target_os = "android")]
fn get_cache_directory_android<T>(event_loop: &EventLoop<T>) -> anyhow::Result<PathBuf> {
    use std::path::PathBuf;

    use anyhow::Context;
    use winit::platform::android::EventLoopExtAndroid;

    let app = event_loop.android_app();
    let app_jobject = unsafe { jni::objects::JObject::from_raw(app.activity_as_ptr().cast()) };
    // If we got a null VM, we can't pass up
    let jvm = unsafe { jni::JavaVM::from_raw(app.vm_as_ptr().cast()).context("Making VM")? };
    let mut env = jvm.attach_current_thread().context("Attaching to thread")?;
    let res = env
        .call_method(app_jobject, "getCacheDir", "()Ljava/io/File;", &[])
        .context("Calling GetCacheDir")?;
    let file = res.l().context("Converting to JObject")?;
    let directory_path = env
        .call_method(file, "getAbsolutePath", "()Ljava/lang/String;", &[])
        .context("Calling `getAbsolutePath`")?;
    let string = directory_path.l().context("Converting to a string")?.into();
    let string = env
        .get_string(&string)
        .context("Converting into a Rust string")?;
    let string: String = string.into();
    let dir = PathBuf::from(string).join("vello");
    if !dir.exists() {
        std::fs::create_dir(&dir).context("Creating pipeline cache directory")?;
    }
    // TODO: Also get the quota. This appears to be more involved, requiring a worker thread and being asynchronous
    Ok(dir)
}

pub(crate) fn get_cache_directory<T>(
    _event_loop: &EventLoop<T>,
) -> anyhow::Result<Option<PathBuf>> {
    #[cfg(target_os = "android")]
    {
        return get_cache_directory_android(_event_loop).map(Some);
    }
    #[expect(
        clippy::allow_attributes,
        reason = "Doesn't apply if a platform has a cache dir"
    )]
    #[allow(
        unreachable_code,
        reason = "Fallback path if none of the short-circuits apply"
    )]
    return Ok(None);
}

/// Load a pipeline cache for the given device, or create one if not present.
///
/// # Safety
///
/// The directory should only have been written into by [`write_pipeline_cache`].
///
/// We recognise that this is an impossible standard, but that's because Vulkan
/// makes UB unavoidable around invalid pipeline caches, so just do your best.
pub(crate) unsafe fn load_pipeline_cache(
    device: &Device,
    adapter_info: &AdapterInfo,
    directory: &Path,
) -> anyhow::Result<Option<(PipelineCache, PathBuf)>> {
    if !device.features().contains(Features::PIPELINE_CACHE) {
        return Ok(None);
    }
    let cache_key = wgpu::util::pipeline_cache_key(adapter_info).ok_or_else(|| {
        anyhow::anyhow!(
            "Adapter info indicates that pipeline cache is not supported, but device has feature."
        )
    })?;
    let cache_file = directory.join(cache_key);
    let data = match std::fs::read(&cache_file) {
        Ok(data) => {
            log::info!("Successfully loaded pipeline cache");
            Some(data)
        }
        Err(e) if e.kind() == ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    let cache = unsafe {
        // Safety: Met by the fact that `write_pipeline_cache` wrote only the data from
        // PipelineCache::get_data.
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("with_winit.pipeline_cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    Ok(Some((cache, cache_file)))
}

pub(crate) fn write_pipeline_cache(path: &Path, cache: &PipelineCache) -> anyhow::Result<()> {
    let Some(data) = cache.get_data() else {
        return Ok(());
    };
    let new_path = path.with_extension(".new");
    let () = std::fs::write(&new_path, data)?;
    if let Err(e) = std::fs::rename(&new_path, path) {
        let () = std::fs::remove_file(&new_path)?;
        // Return the failure from renaming
        Err(e.into())
    } else {
        Ok(())
    }
}
